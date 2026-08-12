# ================================================================
# Sparse Bayesian Neural Networks for Breast-Cancer Prediction
# Simulation verification + WDBC real-data verification + optional NKI
# Designed for Google Colab / standard Python 3 environment.
# ================================================================

import os, math, json, random, warnings, shutil, zipfile, platform
from pathlib import Path
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from scipy.special import expit, logsumexp
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, log_loss, brier_score_loss, accuracy_score,
    precision_recall_curve, average_precision_score, roc_curve,
    confusion_matrix
)

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from IPython.display import display
except Exception:
    display = print

# ------------------------------
# Configuration
# ------------------------------
SEED = 20260812
FAST_MODE = True
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE = torch.float32

# Optional NKI analysis. Upload CSV/XLSX to Colab and set NKI_FILE.
# Example: NKI_FILE = '/content/NKI_cleaned.xlsx'
NKI_FILE = None
NKI_TIME_COL = None
NKI_EVENT_COL = None
NKI_DROP_COLS = []

OUTDIR = Path('/content/sbnn_verification_outputs') if Path('/content').exists() else Path('./sbnn_verification_outputs')
OUTDIR.mkdir(parents=True, exist_ok=True)

if FAST_MODE:
    EPOCHS_SIM_CLASS = 220
    EPOCHS_SIM_SURV = 240
    EPOCHS_REAL = 160
    MC_PRED = 80
    OUTER_FOLDS = 3
    INNER_FOLDS = 2
    N_BOOT = 150
else:
    EPOCHS_SIM_CLASS = 700
    EPOCHS_SIM_SURV = 800
    EPOCHS_REAL = 550
    MC_PRED = 400
    OUTER_FOLDS = 5
    INNER_FOLDS = 3
    N_BOOT = 1000

# ------------------------------
# Reproducibility / utilities
# ------------------------------
def set_all_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_all_seeds()

print('='*78)
print('Sparse Bayesian Neural Network verification suite')
print('='*78)
print('Device:', DEVICE)
print('FAST_MODE:', FAST_MODE)
print('Output directory:', OUTDIR.resolve())
print('Python:', platform.python_version())
print('NumPy:', np.__version__, '| pandas:', pd.__version__, '| torch:', torch.__version__)

plt.rcParams.update({
    'figure.dpi': 120,
    'savefig.dpi': 220,
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'legend.fontsize': 9,
    'figure.figsize': (7.2, 5.2)
})


def show_table(df, name, index=False):
    path = OUTDIR / f'{name}.csv'
    df.to_csv(path, index=index)
    print(f'\n--- {name} ---')
    display(df)
    return path


def save_show(fig, name):
    path = OUTDIR / f'{name}.png'
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    return path


def sigmoid_np(x):
    return expit(np.clip(x, -40, 40))


def clip_probs(p):
    return np.clip(np.asarray(p), 1e-7, 1 - 1e-7)


def ece_score(y, p, n_bins=10):
    y = np.asarray(y).astype(int)
    p = clip_probs(p)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def calibration_intercept_slope(y, p):
    p = clip_probs(p)
    logitp = np.log(p / (1 - p)).reshape(-1, 1)
    try:
        mod = LogisticRegression(C=1e6, solver='lbfgs', max_iter=2000)
        mod.fit(logitp, y)
        return float(mod.intercept_[0]), float(mod.coef_[0, 0])
    except Exception:
        return np.nan, np.nan


def class_metrics(y, p, threshold=0.5):
    p = clip_probs(p)
    pred = (p >= threshold).astype(int)
    ci, cs = calibration_intercept_slope(y, p)
    return {
        'AUC': roc_auc_score(y, p),
        'LogLoss': log_loss(y, p),
        'Brier': brier_score_loss(y, p),
        'Accuracy': accuracy_score(y, pred),
        'ECE': ece_score(y, p),
        'CalIntercept': ci,
        'CalSlope': cs
    }


def bootstrap_metric_ci(y, p, metric_name, B=200, seed=SEED):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    p = np.asarray(p)
    vals = []
    n = len(y)
    for _ in range(B):
        ix = rng.integers(0, n, n)
        yy, pp = y[ix], p[ix]
        if len(np.unique(yy)) < 2 and metric_name == 'AUC':
            continue
        if metric_name == 'AUC': vals.append(roc_auc_score(yy, pp))
        elif metric_name == 'LogLoss': vals.append(log_loss(yy, clip_probs(pp)))
        elif metric_name == 'Brier': vals.append(brier_score_loss(yy, pp))
    if not vals:
        return (np.nan, np.nan)
    return tuple(np.quantile(vals, [0.025, 0.975]))


# ================================================================
# Variational building blocks
# ================================================================
class VariationalLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_std=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_std = float(prior_std)
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_rho = nn.Parameter(torch.empty(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_mu)
        nn.init.constant_(self.weight_rho, -1.5)
        nn.init.zeros_(self.bias_mu)
        nn.init.constant_(self.bias_rho, -1.5)

    @staticmethod
    def _sigma(rho):
        return F.softplus(rho) + 1e-6

    def forward(self, x, sample=True):
        if sample:
            ws = self._sigma(self.weight_rho)
            bs = self._sigma(self.bias_rho)
            w = self.weight_mu + ws * torch.randn_like(self.weight_mu)
            b = self.bias_mu + bs * torch.randn_like(self.bias_mu)
        else:
            w, b = self.weight_mu, self.bias_mu
        return F.linear(x, w, b)

    def kl_components(self):
        ps2 = self.prior_std ** 2
        ws = self._sigma(self.weight_rho)
        bs = self._sigma(self.bias_rho)
        prior_sd = torch.tensor(self.prior_std, device=ws.device, dtype=ws.dtype)
        klw = torch.log(prior_sd / ws) + (ws.pow(2) + self.weight_mu.pow(2)) / (2 * ps2) - 0.5
        klb = torch.log(prior_sd / bs) + (bs.pow(2) + self.bias_mu.pow(2)) / (2 * ps2) - 0.5
        return klw, klb

    def kl(self):
        klw, klb = self.kl_components()
        return klw.sum() + klb.sum()


def bernoulli_kl(q, p):
    eps = 1e-7
    q = torch.clamp(q, eps, 1-eps)
    p = torch.tensor(float(p), dtype=q.dtype, device=q.device)
    return (q * (torch.log(q) - torch.log(p)) +
            (1-q) * (torch.log(1-q) - torch.log(1-p))).sum()


class SparseBayesClassifier(nn.Module):
    def __init__(self, p, hidden=20, prior_inclusion=0.20, prior_std=0.5, temperature=0.35):
        super().__init__()
        self.p = p
        self.prior_inclusion = float(prior_inclusion)
        self.temperature = float(temperature)
        self.gate_logits = nn.Parameter(torch.zeros(p))
        self.fc1 = VariationalLinear(p, hidden, prior_std)
        self.fc2 = VariationalLinear(hidden, 1, prior_std)

    def pips(self):
        return torch.sigmoid(self.gate_logits)

    def sample_gate(self, hard=False, temperature=None):
        q = self.pips()
        if hard:
            return torch.bernoulli(q)
        tau = self.temperature if temperature is None else temperature
        u = torch.rand_like(q).clamp(1e-6, 1-1e-6)
        logistic_noise = torch.log(u) - torch.log1p(-u)
        return torch.sigmoid((self.gate_logits + logistic_noise) / tau)

    def forward(self, x, sample=True, hard_gate=False, temperature=None):
        z = self.sample_gate(hard=hard_gate, temperature=temperature) if sample else self.pips()
        h = torch.tanh(self.fc1(x * z, sample=sample))
        return self.fc2(h, sample=sample).squeeze(-1)

    def kl(self):
        # Spike-and-slab-consistent factorization: when z_j=0 the outgoing first-layer
        # slab weights for feature j are irrelevant, so their Gaussian KL is weighted
        # by q(z_j=1). Biases and downstream weights always contribute.
        klw, klb = self.fc1.kl_components()
        q = self.pips()
        first = (klw * q[None, :]).sum() + klb.sum()
        return first + self.fc2.kl() + bernoulli_kl(q, self.prior_inclusion)


class GaussianBayesClassifier(nn.Module):
    def __init__(self, p, hidden=20, prior_std=0.5):
        super().__init__()
        self.fc1 = VariationalLinear(p, hidden, prior_std)
        self.fc2 = VariationalLinear(hidden, 1, prior_std)

    def forward(self, x, sample=True):
        h = torch.tanh(self.fc1(x, sample=sample))
        return self.fc2(h, sample=sample).squeeze(-1)

    def kl(self):
        return self.fc1.kl() + self.fc2.kl()


class SparseBayesSurvival(nn.Module):
    def __init__(self, p, K, hidden=20, prior_inclusion=0.15, prior_std=0.5, temperature=0.35):
        super().__init__()
        self.p = p
        self.K = K
        self.prior_inclusion = float(prior_inclusion)
        self.temperature = float(temperature)
        self.gate_logits = nn.Parameter(torch.zeros(p))
        self.fc1 = VariationalLinear(p, hidden, prior_std)
        self.risk = VariationalLinear(hidden, 1, prior_std)
        # Deterministic flexible baseline interval logits; weak L2 penalty below.
        self.alpha = nn.Parameter(torch.linspace(-2.2, -1.0, K))

    def pips(self):
        return torch.sigmoid(self.gate_logits)

    def sample_gate(self, hard=False, temperature=None):
        q = self.pips()
        if hard:
            return torch.bernoulli(q)
        tau = self.temperature if temperature is None else temperature
        u = torch.rand_like(q).clamp(1e-6, 1-1e-6)
        noise = torch.log(u) - torch.log1p(-u)
        return torch.sigmoid((self.gate_logits + noise) / tau)

    def forward(self, x, sample=True, hard_gate=False, temperature=None):
        z = self.sample_gate(hard=hard_gate, temperature=temperature) if sample else self.pips()
        h = torch.tanh(self.fc1(x * z, sample=sample))
        r = self.risk(h, sample=sample).squeeze(-1)
        return r[:, None] + self.alpha[None, :]

    def kl(self):
        # Same feature-gated spike-and-slab KL construction as in classification.
        klw, klb = self.fc1.kl_components()
        q = self.pips()
        first = (klw * q[None, :]).sum() + klb.sum()
        baseline_pen = 0.5 * (self.alpha / 5.0).pow(2).sum()
        return first + self.risk.kl() + bernoulli_kl(q, self.prior_inclusion) + baseline_pen


class GaussianBayesSurvival(nn.Module):
    def __init__(self, p, K, hidden=10, prior_std=0.5):
        super().__init__()
        self.K = K
        self.fc1 = VariationalLinear(p, hidden, prior_std)
        self.risk = VariationalLinear(hidden, 1, prior_std)
        self.alpha = nn.Parameter(torch.linspace(-2.2, -1.0, K))

    def forward(self, x, sample=True):
        h = torch.tanh(self.fc1(x, sample=sample))
        r = self.risk(h, sample=sample).squeeze(-1)
        return r[:, None] + self.alpha[None, :]

    def kl(self):
        baseline_pen = 0.5 * (self.alpha / 5.0).pow(2).sum()
        return self.fc1.kl() + self.risk.kl() + baseline_pen


# ================================================================
# Training / prediction
# ================================================================
def _to_tensor(x):
    return torch.as_tensor(np.asarray(x), dtype=DTYPE, device=DEVICE)


def train_sparse_classifier(X, y, hidden=20, prior_inclusion=.2, prior_std=0.5,
                            epochs=300, lr=8e-3, seed=SEED, verbose=False,
                            X_val=None, y_val=None):
    set_all_seeds(seed)
    X_t, y_t = _to_tensor(X), _to_tensor(y)
    model = SparseBayesClassifier(X.shape[1], hidden, prior_inclusion, prior_std).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_state, best_score, patience, bad = None, np.inf, max(40, epochs//6), 0
    use_validation = X_val is not None
    trace = []
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        tau = max(0.25, 1.0 - 0.75 * ep/max(1, int(.65*epochs)))
        logits = model(X_t, sample=True, hard_gate=False, temperature=tau)
        nll = F.binary_cross_entropy_with_logits(logits, y_t, reduction='mean')
        kl_warm = min(1.0, (ep+1) / max(1, int(.30*epochs)))
        loss = nll + kl_warm * model.kl()/len(X)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        trace.append(float(loss.detach().cpu()))

        if use_validation and (ep % 5 == 0 or ep == epochs-1):
            model.eval()
            with torch.no_grad():
                pv = torch.sigmoid(model(_to_tensor(X_val), sample=False)).cpu().numpy()
            score = log_loss(y_val, clip_probs(pv))
            if ep >= int(.30*epochs):
                if score < best_score - 1e-5:
                    best_score = score
                    best_state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
                    bad = 0
                else:
                    bad += 1
                if bad >= patience and ep > int(.55*epochs):
                    break
    if use_validation and best_state is not None:
        model.load_state_dict(best_state)
    if verbose:
        print(f'SBNN classifier trained: epochs={len(trace)}' + (f', best validation={best_score:.4f}' if np.isfinite(best_score) else ''))
    return model, trace


def train_gaussian_bnn_classifier(X, y, hidden=20, prior_std=0.5, epochs=300, lr=8e-3,
                                  seed=SEED, X_val=None, y_val=None):
    set_all_seeds(seed)
    X_t, y_t = _to_tensor(X), _to_tensor(y)
    model = GaussianBayesClassifier(X.shape[1], hidden, prior_std).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_state, best_score, patience, bad = None, np.inf, max(40, epochs//6), 0
    trace=[]
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        logits = model(X_t, sample=True)
        nll = F.binary_cross_entropy_with_logits(logits, y_t, reduction='mean')
        kl_warm = min(1.0, (ep+1) / max(1, int(.30*epochs)))
        loss = nll + kl_warm * model.kl()/len(X)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0); opt.step()
        trace.append(float(loss.detach().cpu()))
        if X_val is not None and (ep % 5 == 0 or ep == epochs-1):
            model.eval()
            with torch.no_grad(): pv=torch.sigmoid(model(_to_tensor(X_val), sample=False)).cpu().numpy()
            score=log_loss(y_val, clip_probs(pv))
            if ep >= int(.30*epochs):
                if score < best_score - 1e-5:
                    best_score=score; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; bad=0
                else: bad += 1
                if bad >= patience and ep > int(.55*epochs): break
    if X_val is not None and best_state is not None: model.load_state_dict(best_state)
    return model, trace


@torch.no_grad()
def predict_sparse_classifier(model, X, M=120, seed=SEED):
    set_all_seeds(seed)
    model.eval(); Xt=_to_tensor(X); draws=[]
    for _ in range(M):
        draws.append(torch.sigmoid(model(Xt, sample=True, hard_gate=True)).cpu().numpy())
    draws=np.asarray(draws)
    mean=draws.mean(0)
    pred_ent=-(clip_probs(mean)*np.log(clip_probs(mean)) + (1-clip_probs(mean))*np.log(1-clip_probs(mean)))
    each_ent=-(clip_probs(draws)*np.log(clip_probs(draws)) + (1-clip_probs(draws))*np.log(1-clip_probs(draws)))
    mi=np.maximum(pred_ent - each_ent.mean(0), 0)
    lo,hi=np.quantile(draws,[.025,.975],axis=0)
    return {'mean':mean,'draws':draws,'entropy':pred_ent,'mi':mi,'lo':lo,'hi':hi}


@torch.no_grad()
def predict_gaussian_classifier(model, X, M=120, seed=SEED):
    set_all_seeds(seed)
    model.eval(); Xt=_to_tensor(X); draws=[]
    for _ in range(M): draws.append(torch.sigmoid(model(Xt, sample=True)).cpu().numpy())
    draws=np.asarray(draws); mean=draws.mean(0)
    return {'mean':mean,'draws':draws}


def make_time_cuts(time, event, K=8):
    time=np.asarray(time,float); event=np.asarray(event,int)
    evt=time[event==1]
    if len(evt)<3: raise ValueError('Too few events to build discrete-time intervals.')
    probs=np.linspace(0,1,K+1)[1:-1]
    inner=np.unique(np.quantile(evt, probs))
    upper=max(time.max()*1.001, evt.max()*1.001)
    cuts=np.unique(np.r_[0.0, inner, upper])
    if len(cuts)<4:
        cuts=np.linspace(0, upper, min(K,4)+1)
    return cuts


def surv_targets(time, event, cuts):
    time=np.asarray(time,float); event=np.asarray(event,int)
    K=len(cuts)-1
    end=cuts[1:]
    m=np.searchsorted(end, time, side='left')
    m=np.clip(m,0,K-1)
    Y=np.zeros((len(time),K),dtype=np.float32)
    M=np.zeros_like(Y)
    for i,mi in enumerate(m):
        M[i,:mi+1]=1.0
        if event[i]==1: Y[i,mi]=1.0
    return Y,M


def train_sparse_survival(X,time,event,cuts,hidden=20,prior_inclusion=.15,prior_std=0.5,
                          epochs=350,lr=8e-3,seed=SEED,verbose=False,
                          X_val=None,time_val=None,event_val=None):
    set_all_seeds(seed)
    Y,M=surv_targets(time,event,cuts)
    Xt,Yt,Mt=_to_tensor(X),_to_tensor(Y),_to_tensor(M)
    model=SparseBayesSurvival(X.shape[1],len(cuts)-1,hidden,prior_inclusion,prior_std).to(DEVICE)
    opt=torch.optim.Adam(model.parameters(),lr=lr)
    best_state,best_score,patience,bad=None,np.inf,max(45,epochs//6),0
    trace=[]
    if X_val is not None:
        Yv,Mv=surv_targets(time_val,event_val,cuts); Xv=_to_tensor(X_val); Yv=_to_tensor(Yv); Mv=_to_tensor(Mv)
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        tau=max(.25,1.0-.75*ep/max(1,int(.65*epochs)))
        logits=model(Xt,sample=True,hard_gate=False,temperature=tau)
        bce=F.binary_cross_entropy_with_logits(logits,Yt,reduction='none')
        nll=(bce*Mt).sum()/len(X)
        kl_warm=min(1.0,(ep+1)/max(1,int(.30*epochs)))
        loss=nll+kl_warm*model.kl()/len(X)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),10.0); opt.step()
        trace.append(float(loss.detach().cpu()))
        if X_val is not None and (ep%5==0 or ep==epochs-1):
            model.eval()
            with torch.no_grad():
                lv=model(Xv,sample=False)
                bv=F.binary_cross_entropy_with_logits(lv,Yv,reduction='none')
                score=float((bv*Mv).sum().cpu()/len(X_val))
            if ep >= int(.30*epochs):
                if score<best_score-1e-5:
                    best_score=score; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; bad=0
                else: bad+=1
                if bad>=patience and ep>int(.55*epochs): break
    if X_val is not None and best_state is not None: model.load_state_dict(best_state)
    if verbose: print(f'SBNN survival trained: epochs={len(trace)}' + (f', best validation={best_score:.4f}' if np.isfinite(best_score) else ''))
    return model,trace


def train_gaussian_survival(X,time,event,cuts,hidden=10,prior_std=0.5,epochs=350,lr=8e-3,seed=SEED):
    set_all_seeds(seed)
    Y,M=surv_targets(time,event,cuts)
    Xt,Yt,Mt=_to_tensor(X),_to_tensor(Y),_to_tensor(M)
    model=GaussianBayesSurvival(X.shape[1],len(cuts)-1,hidden,prior_std).to(DEVICE)
    opt=torch.optim.Adam(model.parameters(),lr=lr)
    trace=[]
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        logits=model(Xt,sample=True)
        bce=F.binary_cross_entropy_with_logits(logits,Yt,reduction='none')
        nll=(bce*Mt).sum()/len(X)
        kl_warm=min(1.0,(ep+1)/max(1,int(.30*epochs)))
        loss=nll+kl_warm*model.kl()/len(X)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),10.0); opt.step()
        trace.append(float(loss.detach().cpu()))
    return model,trace


@torch.no_grad()
def predict_gaussian_survival(model,X,M=120,seed=SEED):
    set_all_seeds(seed); model.eval(); Xt=_to_tensor(X); surv=[]
    for _ in range(M):
        haz=torch.sigmoid(model(Xt,sample=True))
        surv.append(torch.cumprod(1-haz,dim=1).cpu().numpy())
    surv=np.asarray(surv)
    return {'surv_mean':surv.mean(0),'surv_draws':surv}


@torch.no_grad()
def predict_sparse_survival(model,X,M=120,seed=SEED):
    set_all_seeds(seed); model.eval(); Xt=_to_tensor(X); surv=[]; risks=[]
    for _ in range(M):
        logits=model(Xt,sample=True,hard_gate=True)
        haz=torch.sigmoid(logits)
        s=torch.cumprod(1-haz,dim=1)
        surv.append(s.cpu().numpy()); risks.append((1-s).cpu().numpy())
    surv=np.asarray(surv); risks=np.asarray(risks)
    return {
        'surv_mean':surv.mean(0), 'risk_mean':risks.mean(0),
        'surv_draws':surv, 'risk_draws':risks,
        'surv_lo':np.quantile(surv,.025,axis=0), 'surv_hi':np.quantile(surv,.975,axis=0),
        'risk_sd':risks.std(0)
    }


# ================================================================
# Censoring-aware metrics and Cox ridge benchmark
# ================================================================
class KaplanMeier:
    def __init__(self, time, event):
        time=np.asarray(time,float); event=np.asarray(event,int)
        order=np.argsort(time); time=time[order]; event=event[order]
        uniq=np.unique(time)
        surv=1.0; times=[]; vals=[]
        n=len(time)
        for t in uniq:
            at_risk=np.sum(time>=t)
            d=np.sum((time==t)&(event==1))
            if at_risk>0 and d>0: surv*=1-d/at_risk
            times.append(t); vals.append(surv)
        self.times=np.asarray(times); self.vals=np.asarray(vals)

    def predict(self,t,left=False):
        arr=np.atleast_1d(t).astype(float)
        out=[]
        for x in arr:
            if left:
                j=np.searchsorted(self.times,x,side='left')-1
            else:
                j=np.searchsorted(self.times,x,side='right')-1
            out.append(1.0 if j<0 else float(self.vals[j]))
        out=np.asarray(out)
        return out[0] if np.ndim(t)==0 else out


def censoring_km(train_time,train_event):
    return KaplanMeier(train_time,1-np.asarray(train_event,int))


def uno_c_index(test_time,test_event,risk,train_time,train_event,tau=None):
    t=np.asarray(test_time,float); e=np.asarray(test_event,int); r=np.asarray(risk,float)
    G=censoring_km(train_time,train_event)
    if tau is None: tau=np.quantile(np.asarray(train_time)[np.asarray(train_event)==1],.9)
    num=den=0.0
    for i in range(len(t)):
        if e[i]!=1 or t[i]>tau: continue
        g=max(float(G.predict(t[i],left=True)),1e-4)
        w=1/(g*g)
        js=np.where(t>t[i])[0]
        if len(js)==0: continue
        diff=r[i]-r[js]
        num += w*(np.sum(diff>0)+0.5*np.sum(diff==0))
        den += w*len(js)
    return np.nan if den==0 else num/den


def ipcw_brier(test_time,test_event,surv_pred,eval_times,train_time,train_event):
    t=np.asarray(test_time,float); e=np.asarray(test_event,int); S=np.asarray(surv_pred,float)
    G=censoring_km(train_time,train_event)
    scores=[]
    for k,u in enumerate(eval_times):
        err=0.0
        for i in range(len(t)):
            if t[i]<=u and e[i]==1:
                g=max(float(G.predict(t[i],left=True)),1e-4)
                err += (S[i,k]**2)/g
            elif t[i]>u:
                g=max(float(G.predict(u,left=False)),1e-4)
                err += ((1-S[i,k])**2)/g
            # censored before u: zero IPCW contribution
        scores.append(err/len(t))
    return np.asarray(scores)


def integrated_brier(eval_times,brier):
    eval_times=np.asarray(eval_times,float); brier=np.asarray(brier,float)
    if len(eval_times)<2: return float(np.nanmean(brier))
    return float(np.trapezoid(brier,eval_times)/(eval_times[-1]-eval_times[0]))


def fit_cox_ridge(X,time,event,penalty=1.0,maxiter=250):
    X=np.asarray(X,float); time=np.asarray(time,float); event=np.asarray(event,int)
    p=X.shape[1]; uniq_event=np.unique(time[event==1])
    def fg(beta):
        xb=np.clip(X@beta,-30,30); ex=np.exp(xb)
        nll=0.0; grad=np.zeros(p)
        for tt in uniq_event:
            ev=(time==tt)&(event==1); d=ev.sum(); risk=time>=tt
            denom=ex[risk].sum()+1e-12
            nll -= xb[ev].sum() - d*np.log(denom)
            grad -= X[ev].sum(axis=0) - d*(X[risk]*ex[risk,None]).sum(axis=0)/denom
        nll += .5*penalty*np.dot(beta,beta)
        grad += penalty*beta
        return nll,grad
    res=minimize(lambda b: fg(b),np.zeros(p),jac=True,method='L-BFGS-B',options={'maxiter':maxiter})
    beta=res.x
    xb=np.clip(X@beta,-30,30); ex=np.exp(xb)
    event_times=[]; increments=[]
    for tt in uniq_event:
        d=np.sum((time==tt)&(event==1)); denom=ex[time>=tt].sum()+1e-12
        event_times.append(tt); increments.append(d/denom)
    return {'beta':beta,'event_times':np.asarray(event_times),'haz_inc':np.asarray(increments),'success':res.success}


def cox_predict_survival(model,X,eval_times):
    risk=np.exp(np.clip(np.asarray(X)@model['beta'],-30,30))
    cum=np.cumsum(model['haz_inc'])
    H0=[]
    for u in eval_times:
        j=np.searchsorted(model['event_times'],u,side='right')-1
        H0.append(0.0 if j<0 else cum[j])
    H0=np.asarray(H0)
    return np.exp(-risk[:,None]*H0[None,:])


# ================================================================
# Simulation generators
# ================================================================
def simulate_sparse_classification(n=650,p=60,seed=SEED):
    rng=np.random.default_rng(seed); X=rng.normal(size=(n,p))
    eta=(1.25*X[:,0]-1.05*X[:,1]+1.10*X[:,2]*X[:,3]+1.15*np.sin(X[:,4])-0.70*(X[:,5]**2-1))
    eta += rng.normal(0,.25,n)
    pr=sigmoid_np(eta); y=rng.binomial(1,pr)
    true=np.zeros(p,int); true[:6]=1
    return X,y,true


def simulate_sparse_survival(n=560,p=60,seed=SEED+1):
    rng=np.random.default_rng(seed); X=rng.normal(size=(n,p))
    eta=.75*X[:,0]-.65*X[:,1]+.65*X[:,2]*X[:,3]+.75*np.sin(X[:,4])-.45*(X[:,5]**2-1)
    shape=1.35; base=.020
    u=rng.uniform(size=n)
    T=(-np.log(u)/(base*np.exp(np.clip(eta,-3,3))))**(1/shape)
    # Choose censoring scale adaptively to obtain a realistic moderate censoring fraction.
    C=rng.exponential(scale=np.quantile(T,.70),size=n)
    time=np.minimum(T,C); event=(T<=C).astype(int)
    true=np.zeros(p,int); true[:6]=1
    return X,time,event,true


# ================================================================
# Plotting helpers
# ================================================================
def plot_roc_curves(y,preds,title,name):
    fig,ax=plt.subplots()
    for label,p in preds.items():
        fpr,tpr,_=roc_curve(y,p); auc=roc_auc_score(y,p)
        ax.plot(fpr,tpr,lw=2,label=f'{label} (AUC={auc:.3f})')
    ax.plot([0,1],[0,1],'--',lw=1,color='gray')
    ax.set(xlabel='False-positive rate',ylabel='True-positive rate',title=title)
    ax.legend(frameon=False); ax.grid(alpha=.25)
    return save_show(fig,name)


def plot_calibration(y,preds,title,name,n_bins=8):
    fig,ax=plt.subplots()
    ax.plot([0,1],[0,1],'--',lw=1,color='gray',label='Ideal')
    for label,p in preds.items():
        p=np.asarray(p); qs=np.unique(np.quantile(p,np.linspace(0,1,n_bins+1)))
        xs=[]; ys=[]
        for a,b in zip(qs[:-1],qs[1:]):
            m=(p>=a)&(p< b if b<qs[-1] else p<=b)
            if m.sum()>=3: xs.append(p[m].mean()); ys.append(np.asarray(y)[m].mean())
        ax.plot(xs,ys,marker='o',label=label)
    ax.set(xlabel='Mean predicted probability',ylabel='Observed fraction',title=title,xlim=(0,1),ylim=(0,1))
    ax.legend(frameon=False); ax.grid(alpha=.25)
    return save_show(fig,name)


def plot_pips(pips,true=None,feature_names=None,title='Posterior inclusion probabilities',name='pips'):
    pips=np.asarray(pips); order=np.argsort(-pips); top=order[:min(30,len(order))]
    labels=[str(i) if feature_names is None else str(feature_names[i]) for i in top]
    fig,ax=plt.subplots(figsize=(8,max(4,0.23*len(top))))
    y=np.arange(len(top)); bars=ax.barh(y,pips[top]); ax.invert_yaxis()
    ax.set_yticks(y); ax.set_yticklabels(labels); ax.set_xlim(0,1)
    ax.set_xlabel('Variational posterior inclusion probability'); ax.set_title(title); ax.grid(axis='x',alpha=.25)
    if true is not None:
        for b,j in zip(bars,top):
            if true[j]==1: b.set_hatch('///')
        ax.text(.99,.02,'Hatched = truly active',transform=ax.transAxes,ha='right',va='bottom',fontsize=9)
    return save_show(fig,name)


def risk_coverage_curve(y,p,uncertainty,metric='error'):
    y=np.asarray(y); p=np.asarray(p); u=np.asarray(uncertainty)
    order=np.argsort(u); covs=np.linspace(.25,1,16); vals=[]
    for c in covs:
        m=max(2,int(round(c*len(y)))); ix=order[:m]
        if metric=='error': vals.append(np.mean((p[ix]>=.5)!=y[ix]))
        else: vals.append(log_loss(y[ix],clip_probs(p[ix])))
    return covs,np.asarray(vals)


def plot_risk_coverage(y,p,uncertainty,title,name):
    fig,ax=plt.subplots(); c,e=risk_coverage_curve(y,p,uncertainty,'error')
    ax.plot(c,e,marker='o',lw=2)
    ax.set(xlabel='Coverage (fraction not abstained)',ylabel='Classification error',title=title)
    ax.grid(alpha=.25)
    return save_show(fig,name)


def plot_brier_curves(eval_times,curves,title,name):
    fig,ax=plt.subplots()
    for label,v in curves.items(): ax.plot(eval_times,v,marker='o',label=label)
    ax.set(xlabel='Time',ylabel='IPCW Brier score',title=title)
    ax.legend(frameon=False); ax.grid(alpha=.25)
    return save_show(fig,name)


# ================================================================
# 1. Sparse nonlinear classification simulation
# ================================================================
def run_classification_simulation():
    print('\n\n'+'='*78+'\n1) SPARSE NONLINEAR CLASSIFICATION SIMULATION\n'+'='*78)
    X,y,true=simulate_sparse_classification()
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.30,stratify=y,random_state=SEED)
    sc=StandardScaler().fit(Xtr); Xtr=sc.transform(Xtr); Xte=sc.transform(Xte)

    # Proposed sparse BNN
    sbnn,trace=train_sparse_classifier(Xtr,ytr,hidden=10,prior_inclusion=.12,epochs=EPOCHS_SIM_CLASS,seed=SEED,verbose=True)
    ps=predict_sparse_classifier(sbnn,Xte,M=MC_PRED,seed=SEED)['mean']
    pred_obj=predict_sparse_classifier(sbnn,Xte,M=MC_PRED,seed=SEED+3)
    pips=sbnn.pips().detach().cpu().numpy()

    # Gaussian BNN comparator
    gbnn,_=train_gaussian_bnn_classifier(Xtr,ytr,hidden=10,epochs=EPOCHS_SIM_CLASS,seed=SEED+11)
    pg=predict_gaussian_classifier(gbnn,Xte,M=MC_PRED,seed=SEED+11)['mean']

    # Elastic-net logistic benchmark
    lr=LogisticRegression(penalty='elasticnet',solver='saga',l1_ratio=.5,C=1.0,max_iter=3000,random_state=SEED)
    lr.fit(Xtr,ytr); pl=lr.predict_proba(Xte)[:,1]
    rf=RandomForestClassifier(n_estimators=350,min_samples_leaf=3,max_features='sqrt',random_state=SEED,n_jobs=-1)
    rf.fit(Xtr,ytr); pr=rf.predict_proba(Xte)[:,1]

    rows=[]
    for label,pred in [('Sparse BNN (proposed)',ps),('Gaussian BNN',pg),('Elastic-net logistic',pl),('Random forest',pr)]:
        r={'Method':label}; r.update(class_metrics(yte,pred)); rows.append(r)
    tab=pd.DataFrame(rows).sort_values('LogLoss')
    show_table(tab,'sim_classification_metrics')

    sel=pd.DataFrame({
        'Feature':np.arange(len(pips)), 'TrueActive':true.astype(bool), 'PIP':pips
    }).sort_values('PIP',ascending=False)
    auc_sel=roc_auc_score(true,pips); ap_sel=average_precision_score(true,pips)
    rec=pd.DataFrame([{'PIP_AUROC':auc_sel,'PIP_AUPRC':ap_sel,'MeanPIP_Active':pips[true==1].mean(),'MeanPIP_Noise':pips[true==0].mean()}])
    show_table(rec,'sim_classification_feature_recovery')
    show_table(sel.head(30),'sim_classification_top_pips')

    plot_roc_curves(yte,{'Sparse BNN':ps,'Gaussian BNN':pg,'Elastic-net logistic':pl,'Random forest':pr},
                    'Simulation: ROC comparison','sim_classification_roc')
    plot_calibration(yte,{'Sparse BNN':ps,'Gaussian BNN':pg,'Elastic-net logistic':pl,'Random forest':pr},
                     'Simulation: calibration','sim_classification_calibration')
    plot_pips(pips,true=true,title='Simulation: sparse-BNN posterior inclusion probabilities',name='sim_classification_pips')
    plot_risk_coverage(yte,pred_obj['mean'],pred_obj['mi'],'Simulation: uncertainty-based selective prediction','sim_classification_risk_coverage')

    fig,ax=plt.subplots(); ax.plot(trace); ax.set(xlabel='Epoch',ylabel='Negative ELBO objective',title='Simulation: sparse-BNN training trace'); ax.grid(alpha=.25)
    save_show(fig,'sim_classification_training_trace')
    return {'metrics':tab,'pips':pips}


# ================================================================
# 2. Sparse censored-survival simulation
# ================================================================
def run_survival_simulation():
    print('\n\n'+'='*78+'\n2) SPARSE CENSORED-SURVIVAL SIMULATION\n'+'='*78)
    X,time,event,true=simulate_sparse_survival()
    ixtr,ixte=train_test_split(np.arange(len(time)),test_size=.30,stratify=event,random_state=SEED)
    Xtr,Xte=X[ixtr],X[ixte]; ttr,tte=time[ixtr],time[ixte]; etr,ete=event[ixtr],event[ixte]
    sc=StandardScaler().fit(Xtr); Xtr=sc.transform(Xtr); Xte=sc.transform(Xte)
    cuts=make_time_cuts(ttr,etr,K=8)

    sbnn,trace=train_sparse_survival(Xtr,ttr,etr,cuts,hidden=10,prior_inclusion=.12,epochs=EPOCHS_SIM_SURV,seed=SEED,verbose=True)
    pp=predict_sparse_survival(sbnn,Xte,M=MC_PRED,seed=SEED)
    pips=sbnn.pips().detach().cpu().numpy()
    endpoints=cuts[1:]
    # Restrict evaluation to stable support, avoiding extremely sparse tail risk sets.
    tau=np.quantile(ttr[etr==1],.85)
    keep=endpoints<=tau
    eval_times=endpoints[keep]
    S_sbnn=pp['surv_mean'][:,keep]
    risk_h=1-S_sbnn[:,-1]
    c_sbnn=uno_c_index(tte,ete,risk_h,ttr,etr,tau=eval_times[-1])
    bs_sbnn=ipcw_brier(tte,ete,S_sbnn,eval_times,ttr,etr)
    ibs_sbnn=integrated_brier(eval_times,bs_sbnn)

    # Gaussian-prior Bayesian survival NN ablation: same nonlinear architecture, no sparse feature gates.
    gb_surv,_=train_gaussian_survival(Xtr,ttr,etr,cuts,hidden=10,epochs=EPOCHS_SIM_SURV,seed=SEED+91)
    gp=predict_gaussian_survival(gb_surv,Xte,M=MC_PRED,seed=SEED+92)
    S_gb=gp['surv_mean'][:,keep]
    risk_gb=1-S_gb[:,-1]
    c_gb=uno_c_index(tte,ete,risk_gb,ttr,etr,tau=eval_times[-1])
    bs_gb=ipcw_brier(tte,ete,S_gb,eval_times,ttr,etr); ibs_gb=integrated_brier(eval_times,bs_gb)

    # Penalized Cox benchmark.
    cox=fit_cox_ridge(Xtr,ttr,etr,penalty=1.0,maxiter=180)
    S_cox=cox_predict_survival(cox,Xte,eval_times)
    risk_cox=1-S_cox[:,-1]
    c_cox=uno_c_index(tte,ete,risk_cox,ttr,etr,tau=eval_times[-1])
    bs_cox=ipcw_brier(tte,ete,S_cox,eval_times,ttr,etr); ibs_cox=integrated_brier(eval_times,bs_cox)

    tab=pd.DataFrame([
        {'Method':'Sparse Bayesian discrete-time survival NN (proposed)','Uno_C':c_sbnn,'IBS':ibs_sbnn},
        {'Method':'Gaussian-prior Bayesian survival NN','Uno_C':c_gb,'IBS':ibs_gb},
        {'Method':'Ridge Cox proportional hazards','Uno_C':c_cox,'IBS':ibs_cox}
    ])
    show_table(tab,'sim_survival_metrics')
    rec=pd.DataFrame([{'PIP_AUROC':roc_auc_score(true,pips),'PIP_AUPRC':average_precision_score(true,pips),
                       'MeanPIP_Active':pips[true==1].mean(),'MeanPIP_Noise':pips[true==0].mean(),
                       'CensoringFraction':1-event.mean()}])
    show_table(rec,'sim_survival_feature_recovery')
    sel=pd.DataFrame({'Feature':np.arange(len(pips)),'TrueActive':true.astype(bool),'PIP':pips}).sort_values('PIP',ascending=False)
    show_table(sel.head(30),'sim_survival_top_pips')

    plot_brier_curves(eval_times,{'Sparse Bayesian survival NN':bs_sbnn,'Gaussian Bayesian survival NN':bs_gb,'Ridge Cox':bs_cox},
                      'Simulation: censoring-aware Brier curves','sim_survival_brier_curves')
    plot_pips(pips,true=true,title='Simulation survival: posterior inclusion probabilities',name='sim_survival_pips')

    # Representative posterior survival curves for 6 test individuals spanning risk.
    ranks=np.argsort(risk_h); chosen=np.unique(np.linspace(0,len(ranks)-1,6).astype(int)); ids=ranks[chosen]
    fig,ax=plt.subplots()
    for i in ids:
        ax.step(endpoints,pp['surv_mean'][i],where='post',label=f'test {i}')
        ax.fill_between(endpoints,pp['surv_lo'][i],pp['surv_hi'][i],step='post',alpha=.12)
    ax.set(xlabel='Time',ylabel='Posterior survival probability',title='Simulation: posterior survival curves with 95% intervals',ylim=(0,1))
    ax.legend(frameon=False,ncol=2); ax.grid(alpha=.25)
    save_show(fig,'sim_survival_posterior_curves')

    fig,ax=plt.subplots(); ax.plot(trace); ax.set(xlabel='Epoch',ylabel='Negative ELBO objective',title='Survival simulation: training trace'); ax.grid(alpha=.25)
    save_show(fig,'sim_survival_training_trace')
    return {'metrics':tab,'pips':pips}


# ================================================================
# 3. WDBC real-data verification with leakage-safe outer CV
# ================================================================
def tune_sbnn_inner(X,y,seed):
    # Lightweight nested tuning: only two prespecified candidates in FAST_MODE.
    candidates=[(6,.18),(10,.18)] if FAST_MODE else [(6,.10),(10,.10),(10,.20),(14,.20)]
    skf=StratifiedKFold(n_splits=INNER_FOLDS,shuffle=True,random_state=seed)
    scores=[]
    inner_epochs=max(80,int(EPOCHS_REAL*.60))
    for hidden,pi0 in candidates:
        fold=[]
        for fi,(a,b) in enumerate(skf.split(X,y)):
            sc=StandardScaler().fit(X[a]); Xa=sc.transform(X[a]); Xb=sc.transform(X[b])
            m,_=train_sparse_classifier(Xa,y[a],hidden=hidden,prior_inclusion=pi0,epochs=inner_epochs,
                                        seed=seed+100*fi,X_val=Xb,y_val=y[b])
            pp=predict_sparse_classifier(m,Xb,M=max(50,MC_PRED//2),seed=seed+10+fi)['mean']
            fold.append(log_loss(y[b],clip_probs(pp)))
        scores.append((np.mean(fold),hidden,pi0))
    return min(scores,key=lambda z:z[0])


def run_wdbc_real_data():
    print('\n\n'+'='*78+'\n3) REAL DATA: WISCONSIN DIAGNOSTIC BREAST CANCER (WDBC)\n'+'='*78)
    dat=load_breast_cancer()
    X=np.asarray(dat.data,float)
    # sklearn target: 0=malignant, 1=benign. Paper target uses 1=malignant.
    y=(dat.target==0).astype(int)
    names=np.asarray(dat.feature_names)
    print(f'n={len(y)}, p={X.shape[1]}, malignant={y.sum()}, benign={(1-y).sum()}')

    outer=StratifiedKFold(n_splits=OUTER_FOLDS,shuffle=True,random_state=SEED)
    method_names=['Sparse BNN (proposed)','Gaussian BNN','Elastic-net logistic','Random forest']
    oof={m:np.full(len(y),np.nan) for m in method_names}
    mi_oof=np.full(len(y),np.nan)
    pip_mat=[]; fold_rows=[]; hyper_rows=[]

    for fold,(tr,te) in enumerate(outer.split(X,y),1):
        print(f'\nOuter fold {fold}/{OUTER_FOLDS}')
        best_score,hidden,pi0=tune_sbnn_inner(X[tr],y[tr],SEED+fold)
        print(f'  selected SBNN hidden={hidden}, prior inclusion={pi0:.2f}, inner logloss={best_score:.4f}')
        hyper_rows.append({'Fold':fold,'Hidden':hidden,'PriorInclusion':pi0,'InnerLogLoss':best_score})
        sc=StandardScaler().fit(X[tr]); Xtr=sc.transform(X[tr]); Xte=sc.transform(X[te])

        sbnn,_=train_sparse_classifier(Xtr,y[tr],hidden=hidden,prior_inclusion=pi0,epochs=EPOCHS_REAL,
                                       seed=SEED+1000+fold)
        pobj=predict_sparse_classifier(sbnn,Xte,M=MC_PRED,seed=SEED+2000+fold)
        oof['Sparse BNN (proposed)'][te]=pobj['mean']; mi_oof[te]=pobj['mi']
        pip_mat.append(sbnn.pips().detach().cpu().numpy())

        gbnn,_=train_gaussian_bnn_classifier(Xtr,y[tr],hidden=hidden,epochs=EPOCHS_REAL,seed=SEED+3000+fold)
        oof['Gaussian BNN'][te]=predict_gaussian_classifier(gbnn,Xte,M=MC_PRED,seed=SEED+4000+fold)['mean']

        # Inner-CV elastic-net logistic via a pipeline: scaler is re-estimated in each inner fold.
        pipe=Pipeline([('sc',StandardScaler()),('lr',LogisticRegression(penalty='elasticnet',solver='saga',l1_ratio=.5,max_iter=4000,random_state=SEED+fold))])
        grid=GridSearchCV(pipe,{'lr__C':[.1,1,10]},cv=StratifiedKFold(n_splits=INNER_FOLDS,shuffle=True,random_state=SEED+fold),
                          scoring='neg_log_loss',n_jobs=1)
        grid.fit(X[tr],y[tr]); oof['Elastic-net logistic'][te]=grid.predict_proba(X[te])[:,1]

        rf=RandomForestClassifier(n_estimators=450,min_samples_leaf=2,max_features='sqrt',class_weight='balanced',random_state=SEED+fold,n_jobs=-1)
        rf.fit(X[tr],y[tr]); oof['Random forest'][te]=rf.predict_proba(X[te])[:,1]

        for m in method_names:
            rr={'Fold':fold,'Method':m}; rr.update(class_metrics(y[te],oof[m][te])); fold_rows.append(rr)

    foldtab=pd.DataFrame(fold_rows)
    show_table(foldtab,'wdbc_outer_fold_metrics')
    show_table(pd.DataFrame(hyper_rows),'wdbc_selected_hyperparameters')

    pooled=[]
    for m in method_names:
        met=class_metrics(y,oof[m]); row={'Method':m}; row.update(met)
        for metric in ['AUC','LogLoss','Brier']:
            lo,hi=bootstrap_metric_ci(y,oof[m],metric,N_BOOT,SEED+7)
            row[f'{metric}_CI_L']=lo; row[f'{metric}_CI_U']=hi
        pooled.append(row)
    pooltab=pd.DataFrame(pooled).sort_values('LogLoss')
    show_table(pooltab,'wdbc_pooled_oof_metrics')

    pmat=np.vstack(pip_mat); meanpip=pmat.mean(0); sdpip=pmat.std(0)
    stability=pd.DataFrame({'Feature':names,'MeanPIP':meanpip,'SD_PIP':sdpip,'SelectionFreq_PIP_gt_0.5':(pmat>.5).mean(0)})
    stability=stability.sort_values('MeanPIP',ascending=False)
    show_table(stability,'wdbc_feature_stability')

    plot_roc_curves(y,{m:oof[m] for m in method_names},'WDBC: pooled out-of-fold ROC','wdbc_roc_oof')
    plot_calibration(y,{m:oof[m] for m in method_names},'WDBC: pooled out-of-fold calibration','wdbc_calibration_oof')
    plot_pips(meanpip,feature_names=names,title='WDBC: mean posterior inclusion probability across outer folds',name='wdbc_mean_pips')
    plot_risk_coverage(y,oof['Sparse BNN (proposed)'],mi_oof,'WDBC: epistemic-uncertainty risk–coverage curve','wdbc_risk_coverage')

    # Confusion matrix at fixed 0.5 is descriptive; no claim of optimized clinical threshold.
    cm=confusion_matrix(y,(oof['Sparse BNN (proposed)']>=.5).astype(int))
    cmtab=pd.DataFrame(cm,index=['True benign','True malignant'],columns=['Pred benign','Pred malignant'])
    show_table(cmtab.reset_index().rename(columns={'index':'Observed'}),'wdbc_sbnn_confusion_matrix')
    fig,ax=plt.subplots(figsize=(5,4)); im=ax.imshow(cm,cmap='Blues')
    for i in range(2):
        for j in range(2): ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=13)
    ax.set_xticks([0,1]); ax.set_xticklabels(['Benign','Malignant']); ax.set_yticks([0,1]); ax.set_yticklabels(['Benign','Malignant'])
    ax.set(xlabel='Predicted',ylabel='Observed',title='WDBC: Sparse BNN OOF confusion matrix (threshold 0.5)')
    save_show(fig,'wdbc_sbnn_confusion_matrix_plot')

    # Predictive interval width vs error indicator: a transparent uncertainty diagnostic.
    # Regenerate one global plot from OOF MI (available for all test patients).
    err=((oof['Sparse BNN (proposed)']>=.5).astype(int)!=y).astype(int)
    ud=pd.DataFrame({'MutualInformation':mi_oof,'Error':err})
    bins=pd.qcut(ud['MutualInformation'],q=min(6,len(np.unique(mi_oof))),duplicates='drop')
    uu=ud.groupby(bins,observed=True).agg(MeanMI=('MutualInformation','mean'),ErrorRate=('Error','mean'),N=('Error','size')).reset_index(drop=True)
    show_table(uu,'wdbc_uncertainty_error_bins')
    fig,ax=plt.subplots(); ax.plot(uu['MeanMI'],uu['ErrorRate'],marker='o'); ax.set(xlabel='Mean mutual information',ylabel='Observed error rate',title='WDBC: error rate versus epistemic uncertainty'); ax.grid(alpha=.25)
    save_show(fig,'wdbc_uncertainty_vs_error')

    return {'metrics':pooltab,'oof':oof,'meanpip':meanpip,'features':names}


# ================================================================
# 4. Optional NKI real-data module
# ================================================================
def load_table(path):
    path=str(path)
    if path.lower().endswith(('.xlsx','.xls')): return pd.read_excel(path)
    if path.lower().endswith('.csv'): return pd.read_csv(path)
    if path.lower().endswith(('.tsv','.txt')): return pd.read_csv(path,sep=None,engine='python')
    raise ValueError('NKI_FILE must be CSV, TSV, XLS, or XLSX.')


def infer_survival_columns(df,time_col=None,event_col=None):
    cols=list(df.columns); low={c:str(c).strip().lower() for c in cols}
    if time_col is None:
        pats=['survival','surv_time','time','follow','followup','dmfs','distant','rfs','os_time','event_time']
        cand=[c for c in cols if any(p in low[c] for p in pats) and pd.api.types.is_numeric_dtype(df[c])]
        # Prefer columns explicitly containing time/follow/survival.
        pref=[c for c in cand if any(p in low[c] for p in ['time','follow','survival'])]
        cand=pref if pref else cand
        if len(cand)==1: time_col=cand[0]
    if event_col is None:
        pats=['event','status','death','metast','relapse','censor']
        cand=[]
        for c in cols:
            if any(p in low[c] for p in pats):
                vals=pd.Series(df[c]).dropna().unique()
                if len(vals)<=5: cand.append(c)
        if len(cand)==1: event_col=cand[0]
    return time_col,event_col


def normalize_event(v):
    s=pd.Series(v)
    if pd.api.types.is_numeric_dtype(s):
        vals=sorted(pd.unique(s.dropna()))
        if set(vals).issubset({0,1}): return s.astype(int).to_numpy()
        if len(vals)==2:
            return (s==vals[-1]).astype(int).to_numpy()
    ss=s.astype(str).str.strip().str.lower()
    event_words={'1','event','dead','death','yes','metastasis','relapse','failure','deceased'}
    censor_words={'0','censored','censor','alive','no','disease-free','disease free'}
    if ss.isin(event_words|censor_words).all(): return ss.isin(event_words).astype(int).to_numpy()
    raise ValueError('Could not safely map event column to 0/1. Recode it explicitly before running.')


def run_nki_if_available():
    global NKI_FILE
    print('\n\n'+'='*78+'\n4) OPTIONAL REAL DATA: NKI TRANSCRIPTOMIC SURVIVAL\n'+'='*78)
    if NKI_FILE is None:
        # Conservative auto-detection only by filename, never by guessing arbitrary data files.
        roots=[Path('/content'),Path('.')]
        hits=[]
        for root in roots:
            if root.exists():
                for pat in ['*NKI*.csv','*NKI*.xlsx','*nki*.csv','*nki*.xlsx','*Netherlands*.csv','*Netherlands*.xlsx']:
                    hits += list(root.glob(pat))
        hits=list(dict.fromkeys(map(str,hits)))
        if len(hits)==1:
            NKI_FILE=hits[0]; print('Auto-detected NKI file:',NKI_FILE)
        else:
            print('NKI analysis skipped: no unique NKI file was configured/detected.')
            print('To run it, upload the verified cleaned file and set:')
            print("  NKI_FILE='/content/your_file.xlsx'")
            print("  NKI_TIME_COL='your_time_column'")
            print("  NKI_EVENT_COL='your_event_indicator_column'  # 1=event, 0=censored")
            return None

    df=load_table(NKI_FILE)
    print('Loaded:',NKI_FILE,'shape=',df.shape)
    tc,ec=infer_survival_columns(df,NKI_TIME_COL,NKI_EVENT_COL)
    if tc is None or ec is None:
        print('\nColumns available:')
        print(list(df.columns))
        raise ValueError('Could not uniquely and safely infer NKI time/event columns. Set NKI_TIME_COL and NKI_EVENT_COL explicitly.')
    print('Using time column:',tc,'| event column:',ec)

    t=pd.to_numeric(df[tc],errors='coerce').to_numpy(float); e=normalize_event(df[ec])
    drop=set([tc,ec]+list(NKI_DROP_COLS))
    Xdf=df.drop(columns=[c for c in drop if c in df.columns]).select_dtypes(include=[np.number]).copy()
    valid=np.isfinite(t)&np.isfinite(e)
    Xdf=Xdf.loc[valid].reset_index(drop=True); t=t[valid]; e=e[valid]
    # Median imputation uses training data later; here retain NaN.
    print(f'Analysis rows={len(t)}, numeric candidate predictors={Xdf.shape[1]}, events={e.sum()}, censoring={1-e.mean():.3f}')
    if len(t)<80 or e.sum()<20: raise ValueError('NKI sample/event count is too small after cleaning; inspect endpoint coding.')

    tr,te=train_test_split(np.arange(len(t)),test_size=.30,stratify=e,random_state=SEED)
    # Training-only median imputation and unsupervised variance filter.
    med=Xdf.iloc[tr].median(axis=0)
    Xtr=Xdf.iloc[tr].fillna(med).to_numpy(float); Xte=Xdf.iloc[te].fillna(med).to_numpy(float)
    names=np.asarray(Xdf.columns)
    var=np.var(Xtr,axis=0)
    keep=np.where(var>1e-12)[0]
    # For speed/stability in p>>n, retain highest-variance inputs only, fitted on training fold.
    maxp=300 if FAST_MODE else 600
    if len(keep)>maxp: keep=keep[np.argsort(var[keep])[-maxp:]]
    Xtr,Xte=Xtr[:,keep],Xte[:,keep]; names=names[keep]
    sc=StandardScaler().fit(Xtr); Xtr=sc.transform(Xtr); Xte=sc.transform(Xte)
    cuts=make_time_cuts(t[tr],e[tr],K=8)

    sbnn,_=train_sparse_survival(Xtr,t[tr],e[tr],cuts,hidden=8,prior_inclusion=max(.02,min(.15,12/Xtr.shape[1])),
                                 epochs=EPOCHS_REAL+80,seed=SEED+55,verbose=True)
    pp=predict_sparse_survival(sbnn,Xte,M=MC_PRED,seed=SEED+56)
    endpoints=cuts[1:]; tau=np.quantile(t[tr][e[tr]==1],.80); mask=endpoints<=tau
    ev= endpoints[mask]; S=pp['surv_mean'][:,mask]; risk=1-S[:,-1]
    c=uno_c_index(t[te],e[te],risk,t[tr],e[tr],tau=ev[-1]); bs=ipcw_brier(t[te],e[te],S,ev,t[tr],e[tr]); ibs=integrated_brier(ev,bs)

    # Gaussian-prior Bayesian survival NN ablation on the identical retained predictors.
    gb,_=train_gaussian_survival(Xtr,t[tr],e[tr],cuts,hidden=8,epochs=EPOCHS_REAL+80,seed=SEED+57)
    gp=predict_gaussian_survival(gb,Xte,M=MC_PRED,seed=SEED+58); Sg=gp['surv_mean'][:,mask]
    rg=1-Sg[:,-1]; cg=uno_c_index(t[te],e[te],rg,t[tr],e[tr],tau=ev[-1]); bsg=ipcw_brier(t[te],e[te],Sg,ev,t[tr],e[tr]); ibsg=integrated_brier(ev,bsg)

    # Ridge Cox benchmark with training-only standardized same predictors.
    cox=fit_cox_ridge(Xtr,t[tr],e[tr],penalty=2.0,maxiter=200); Sc=cox_predict_survival(cox,Xte,ev)
    rc=1-Sc[:,-1]; cc=uno_c_index(t[te],e[te],rc,t[tr],e[tr],tau=ev[-1]); bsc=ipcw_brier(t[te],e[te],Sc,ev,t[tr],e[tr]); ibsc=integrated_brier(ev,bsc)
    tab=pd.DataFrame([
        {'Method':'Sparse Bayesian discrete-time survival NN (proposed)','Uno_C':c,'IBS':ibs},
        {'Method':'Gaussian-prior Bayesian survival NN','Uno_C':cg,'IBS':ibsg},
        {'Method':'Ridge Cox proportional hazards','Uno_C':cc,'IBS':ibsc}
    ])
    show_table(tab,'nki_holdout_metrics')
    pips=sbnn.pips().detach().cpu().numpy(); stab=pd.DataFrame({'Feature':names,'PIP':pips}).sort_values('PIP',ascending=False)
    show_table(stab.head(50),'nki_top_pips')
    plot_brier_curves(ev,{'Sparse Bayesian survival NN':bs,'Gaussian Bayesian survival NN':bsg,'Ridge Cox':bsc},'NKI: censoring-aware Brier curves','nki_brier_curves')
    plot_pips(pips,feature_names=names,title='NKI: posterior inclusion probabilities (training-only variance-filtered inputs)',name='nki_pips')
    return {'metrics':tab,'pips':pips,'features':names}


# ================================================================
# Final audit / archive
# ================================================================
def write_manifest():
    manifest={
        'seed':SEED,'fast_mode':FAST_MODE,'device':str(DEVICE),'mc_prediction_draws':MC_PRED,
        'outer_folds_wdbc':OUTER_FOLDS,'inner_folds_wdbc':INNER_FOLDS,
        'notes':[
            'WDBC target recoded so 1=malignant, matching manuscript notation.',
            'All WDBC scaling for the proposed model is fitted only on the corresponding training fold.',
            'Sparse BNN feature gates use a Concrete relaxation in training and Bernoulli posterior draws at prediction.',
            'Survival simulation uses a right-censoring-aware discrete-time hazard likelihood.',
            'Uno-style IPCW concordance and IPCW Brier weights use censoring estimated from training data.',
            'NKI is not analyzed unless a verified local file is supplied or uniquely auto-detected by NKI-like filename.',
            'NKI optional variance filtering is unsupervised and fitted only on the training split.'
        ]
    }
    with open(OUTDIR/'run_manifest.json','w') as f: json.dump(manifest,f,indent=2)


def archive_outputs():
    # Copy this script into outputs when possible.
    try:
        thisfile=Path(__file__)
        if thisfile.exists(): shutil.copy2(thisfile,OUTDIR/thisfile.name)
    except Exception: pass
    zip_path=Path(str(OUTDIR)+'.zip')
    if zip_path.exists(): zip_path.unlink()
    shutil.make_archive(str(OUTDIR),'zip',root_dir=OUTDIR)
    print('\n'+'='*78)
    print('ALL ANALYSES COMPLETED')
    print('='*78)
    print('Saved output folder:',OUTDIR.resolve())
    print('ZIP archive:',zip_path.resolve())
    if Path('/content').exists():
        print('\nColab download command:')
        print("from google.colab import files; files.download('/content/sbnn_verification_outputs.zip')")
    return zip_path


if __name__=='__main__':
    results={}
    results['sim_classification']=run_classification_simulation()
    results['sim_survival']=run_survival_simulation()
    results['wdbc']=run_wdbc_real_data()
    try:
        results['nki']=run_nki_if_available()
    except Exception as ex:
        print('\nNKI module stopped safely:',repr(ex))
        print('No NKI numerical results were fabricated. Correct the NKI configuration and rerun that module.')
    write_manifest()
    archive_outputs()
