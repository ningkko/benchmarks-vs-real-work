"""Suspension main model: logit version and two-way clustering (developer and repository).

Logit: merged within 7 days on treated x P0/P1/P3, one dummy per treated developer, one dummy per calendar day;
comparison developers share one intercept (their role is to identify the day effects). Treated developers whose
pull requests are all merged or all unmerged carry no within-developer information and are dropped (their dummy
would be infinite). Reported on the probability scale: the average over the remaining treated pull requests of the
predicted change when a phase indicator is switched on against P2; reversal = mean of the P1 and P3 changes.
Standard errors clustered by developer, delta method with a numerical gradient.
Two-way clustering: the linear model of twfe.py with V = V_developer + V_repository - V_developer x repository.

usage: python3 -I robust2.py BASE
"""
import os, sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
import statsmodels.api as sm
BASE = sys.argv[1]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ci, load_suspension, demean
D = load_suspension(BASE)

# ---- two-way clustering on the linear model
Xcols = ["T_P0", "T_P1", "T_P3"]
w = pd.Series(np.ones(len(D)), index=D.index)
keys = [D.user, D.day]
yt = demean(D["merged_7d"].astype(float), w, keys).values
Xt = np.column_stack([demean(D[c], w, keys) for c in Xcols])
bread = np.linalg.pinv(Xt.T @ Xt); b = bread @ (Xt.T @ yt); e = yt - Xt @ b
N, K = len(D), len(Xcols)


def V_for(groups):
    meat = np.zeros((K, K))
    for _, idx in pd.Series(np.arange(N)).groupby(groups):
        s = (Xt[idx.values] * e[idx.values][:, None]).sum(0); meat += np.outer(s, s)
    G = pd.Series(groups).nunique()
    return bread @ meat @ bread * (G / (G - 1)) * ((N - 1) / (N - K))


Vd = V_for(D.user.values); Vr = V_for(D.repo.values); Vdr = V_for((D.user + "|" + D.repo).values)
V2 = Vd + Vr - Vdr
c = np.array([0, .5, .5])
print(f"linear, CR1 by developer: reversal {ci(c @ b, np.sqrt(c @ Vd @ c))}; P0 placebo {ci(b[0], np.sqrt(Vd[0, 0]))}")
print(f"linear, two-way developer and repository: reversal {ci(c @ b, np.sqrt(max(c @ V2 @ c, 0)))}; P0 placebo {ci(b[0], np.sqrt(max(V2[0, 0], 0)))}")

# ---- logit
L = D.copy()
tr = L.treated == 1
var = L[tr].groupby("user").merged_7d.agg(["min", "max"])
keep_dev = set(var[var["min"] != var["max"]].index)
L = L[(~tr) | L.user.isin(keep_dev)].reset_index(drop=True)
dev_d = pd.get_dummies(L.user.where(L.treated == 1, "_comparison"), prefix="dev", drop_first=True, dtype=float)
day_d = pd.get_dummies(L.day, prefix="day", drop_first=True, dtype=float)
X = pd.concat([L[Xcols].astype(float), dev_d, day_d], axis=1)
X.insert(0, "const", 1.0)
y = L.merged_7d.astype(float).values
m = sm.Logit(y, X.values).fit(disp=0, method="newton", maxiter=200, cov_type="cluster", cov_kwds={"groups": pd.factorize(L.user)[0]})
beta, Vb = m.params, m.cov_params()
Xv = X.values; T = (L.treated == 1).values
pos = {col: X.columns.get_loc(col) for col in Xcols}


def ame(bv):
    xt = Xv[T].copy()
    base = xt.copy(); base[:, [pos[c_] for c_ in Xcols]] = 0.0
    p2 = 1 / (1 + np.exp(-(base @ bv)))
    out = []
    for c_ in Xcols:
        on = base.copy(); on[:, pos[c_]] = 1.0
        out.append((1 / (1 + np.exp(-(on @ bv))) - p2).mean())
    return np.array(out)


a = ame(beta)
eps = 1e-6
J = np.column_stack([(ame(beta + eps * np.eye(len(beta))[j]) - ame(beta - eps * np.eye(len(beta))[j])) / (2 * eps) for j in range(len(beta))])
Va = J @ Vb @ J.T
print(f"logit (treated-developer and day dummies; {len(keep_dev)} treated developers with variation in the outcome), "
      f"average marginal effects: reversal {ci(c @ a, np.sqrt(c @ Va @ c))}; P0 placebo {ci(a[0], np.sqrt(Va[0, 0]))}")
