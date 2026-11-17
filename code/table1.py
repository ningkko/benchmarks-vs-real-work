"""Table 1 (baseline balance) and entropy balancing on the all-pull-request data.

usage: python3 -I table1.py BASE
writes BASE/table1.csv, BASE/table1_weighted_model.csv, BASE/table1.md
"""
import os, json, sys, numpy as np, pandas as pd, warnings
from scipy.optimize import minimize
warnings.filterwarnings("ignore")
BASE = sys.argv[1]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import prep, ci, phase, load_suspension, fit
D = load_suspension(BASE)
treated297 = set(D[D.treated == 1].user)


def load(path):
    raw = [json.loads(l) for l in open(path)]
    df = prep(raw)
    t = {r["html_url"]: (r.get("title") or "") for r in raw if "html_url" in r}
    bl = {r["html_url"]: (r.get("body_len_full") if r.get("body_len_full") is not None else len(r.get("body") or "")) for r in raw if "html_url" in r}
    df["title_feat"] = df.html_url.map(t).str.strip().str.lower().str.startswith("feat").astype(float)
    df["body_len"] = df.html_url.map(bl).astype(float)
    return phase(df).copy()


A = load(f"{BASE}/subst/treated_all.jsonl"); A = A[A.user.isin(treated297)].copy(); A["treated"] = 1
B = load(f"{BASE}/subst/control_all.jsonl"); B = B[~B.user.isin(treated297)].copy(); B["treated"] = 0
X = pd.concat([A, B], ignore_index=True).drop_duplicates("html_url")
for c in ["merged_7d", "merged_1h", "closed_unmerged", "own_repo"]:
    X[c] = X[c].astype(float)
X["comments"] = pd.to_numeric(X.comments, errors="coerce").fillna(0).astype(float)
ASSOC = ["OWNER", "MEMBER", "COLLABORATOR", "CONTRIBUTOR", "NONE"]
X["assoc"] = X.assoc.where(X.assoc.isin(ASSOC), "OTHER")
for a in ASSOC + ["OTHER"]:
    X[f"assoc_{a}"] = (X.assoc == a).astype(float)
ASSOC_ROWS = [a for a in ASSOC + ["OTHER"] if X[f"assoc_{a}"].sum() > 0]


def wmean(v, w):
    return np.sum(v * w) / np.sum(w)


def wvar(v, w):
    m = wmean(v, w); return np.sum(w * (v - m) ** 2) / np.sum(w)


def smd(a, b, wa=None, wb=None):
    wa = np.ones(len(a)) if wa is None else wa; wb = np.ones(len(b)) if wb is None else wb
    s = np.sqrt((wvar(a, wa) + wvar(b, wb)) / 2)
    return (wmean(a, wa) - wmean(b, wb)) / s if s > 0 else 0.0


def wquant(v, w, q):
    o = np.argsort(v); v, w = np.asarray(v)[o], np.asarray(w)[o]; cw = np.cumsum(w) / w.sum()
    return v[np.searchsorted(cw, q)]


def describe(g, w=None):
    """g: pull requests of one group; w: per pull request weights (developer weight) or None."""
    w = np.ones(len(g)) if w is None else np.asarray(w)
    dev = g.assign(_w=w).groupby("user").agg(n=("html_url", "size"), w=("_w", "first"))
    out = {"developers": len(dev) if w is None or np.all(w == 1) else dev.w.sum() ** 2 / (dev.w ** 2).sum(),
           "prs_per_dev_median": wquant(dev.n.values, dev.w.values, .5), "prs_per_dev_q1": wquant(dev.n.values, dev.w.values, .25),
           "prs_per_dev_q3": wquant(dev.n.values, dev.w.values, .75), "pull_requests": len(g)}
    for c in ["merged_7d", "merged_1h", "closed_unmerged", "own_repo", "title_feat"] + [f"assoc_{a}" for a in ASSOC_ROWS]:
        out[c] = wmean(g[c].values, w) * 100
    out["comments"] = wmean(g.comments.values, w)
    out["body_len_median"] = wquant(g.body_len.fillna(0).values, w, .5)
    return out, dev


def smds(g1, g0, w1=None, w0=None):
    w1 = np.ones(len(g1)) if w1 is None else np.asarray(w1); w0 = np.ones(len(g0)) if w0 is None else np.asarray(w0)
    d1 = g1.assign(_w=w1).groupby("user").agg(n=("html_url", "size"), w=("_w", "first"))
    d0 = g0.assign(_w=w0).groupby("user").agg(n=("html_url", "size"), w=("_w", "first"))
    s = {"prs_per_dev_median": smd(np.log1p(d1.n.values), np.log1p(d0.n.values), d1.w.values, d0.w.values)}
    for c in ["merged_7d", "merged_1h", "closed_unmerged", "own_repo", "title_feat"] + [f"assoc_{a}" for a in ASSOC_ROWS]:
        s[c] = smd(g1[c].values, g0[c].values, w1, w0)
    s["comments"] = smd(np.log1p(g1.comments.values), np.log1p(g0.comments.values), w1, w0)
    s["body_len_median"] = smd(np.log1p(g1.body_len.fillna(0).values), np.log1p(g0.body_len.fillna(0).values), w1, w0)
    return s


T0 = X[(X.treated == 1) & (X.phase == "P0")]; C0 = X[(X.treated == 0) & (X.phase == "P0")]

# ---- entropy balancing at the developer level, exact on three P0 moments
tgt_m7 = T0.merged_7d.mean(); tgt_own = T0.own_repo.mean(); tgt_n = T0.groupby("user").size().mean()
cd = C0.groupby("user").agg(n=("html_url", "size"), m7=("merged_7d", "sum"), own=("own_repo", "sum"))
G = np.column_stack([cd.m7 - tgt_m7 * cd.n, cd.own - tgt_own * cd.n, cd.n - tgt_n]).astype(float)
Gs = G / G.std(0)
dual = lambda lam: np.log(np.exp(Gs @ lam).sum())
grad = lambda lam: (np.exp(Gs @ lam)[:, None] * Gs).sum(0) / np.exp(Gs @ lam).sum()
res = minimize(dual, np.zeros(3), jac=grad, method="BFGS", options={"gtol": 1e-12, "maxiter": 5000})
wd = np.exp(Gs @ res.x); wd = wd / wd.sum() * len(wd)
cd["w"] = wd
print(f"entropy balancing: converged {res.success}; max |weighted moment| = {np.abs((wd[:, None] * G).sum(0) / wd.sum()).max():.2e}; "
      f"effective n {wd.sum()**2/(wd**2).sum():.1f} of {len(wd)}; max weight {wd.max():.2f}")
wC0 = C0.user.map(cd.w).values

colsA = {}
colsA["Treated"], _ = describe(T0)
colsA["Comparison"], _ = describe(C0)
colsA["Comparison, weighted"], _ = describe(C0, wC0)
smdA = smds(T0, C0); smdAw = smds(T0, C0, None, wC0)
colsB = {p: describe(X[(X.treated == 1) & (X.phase == p)])[0] for p in ["P0", "P1", "P2", "P3"]}
P2 = X[(X.treated == 1) & (X.phase == "P2")]
smdB = {p: smds(X[(X.treated == 1) & (X.phase == p)], P2) for p in ["P0", "P1", "P3"]}

LAB = [("developers", "Developers with at least one pull request"), ("pull_requests", "Pull requests"),
       ("prs_per_dev_median", "Pull requests per developer, median [IQR]"), ("merged_7d", "Merged within 7 days (%)"),
       ("merged_1h", "Merged within 1 hour (%)"), ("closed_unmerged", "Closed without merge (%)"),
       ("comments", "Comments per pull request, mean"), ("own_repo", "Opened in the author's own repository (%)")] + \
      [(f"assoc_{a}", f"Author association: {a} (%)") for a in ASSOC_ROWS] + \
      [("title_feat", "Title starts with \"feat\" (%)"), ("body_len_median", "Description length, median characters")]


def fmtv(col, k):
    if k == "prs_per_dev_median":
        return f"{col[k]:.0f} [{col['prs_per_dev_q1']:.0f}-{col['prs_per_dev_q3']:.0f}]"
    if k in ("developers",):
        return f"{col[k]:.0f}" if float(col[k]).is_integer() else f"{col[k]:.1f} (effective)"
    if k == "pull_requests":
        return f"{col[k]:,}"
    if k == "comments":
        return f"{col[k]:.2f}"
    if k == "body_len_median":
        return f"{col[k]:.0f}"
    return f"{col[k]:.1f}"


rows = []
for k, lab in LAB:
    r = {"variable": lab}
    for nm, col in colsA.items():
        r[f"A: {nm}"] = fmtv(col, k)
    r["A: SMD"] = f"{smdA[k]:+.2f}" if k in smdA else ""
    r["A: SMD, weighted"] = f"{smdAw[k]:+.2f}" if k in smdAw else ""
    for p in ["P0", "P1", "P2", "P3"]:
        r[f"B: treated {p}"] = fmtv(colsB[p], k)
    for p in ["P0", "P1", "P3"]:
        r[f"B: SMD {p} vs P2"] = f"{smdB[p][k]:+.2f}" if k in smdB[p] else ""
    rows.append(r)
T1 = pd.DataFrame(rows)
T1.to_csv(f"{BASE}/table1.csv", index=False)

# ---- refit the suspension model on the all-pull-request data, unweighted and weighted
keep_c = set(cd.index)
D = X[(X.treated == 1) | X.user.isin(keep_c)].reset_index(drop=True)
D["day"] = D.created_at.dt.strftime("%Y-%m-%d")
for p in ["P0", "P1", "P3"]:
    D[f"T_{p}"] = ((D.treated == 1) & (D.phase == p)).astype(float)
c = np.array([0, .5, .5]); mres = []
wt_eb = np.where(D.treated == 1, 1.0, D.user.map(cd.w).fillna(0).values)
for nm, w in [("unweighted (all-pull-request model, comparison developers with P0 pull requests)", np.ones(len(D))),
              ("entropy-balanced comparison", wt_eb)]:
    b, se, V = fit(D, "merged_7d", ["T_P0", "T_P1", "T_P3"], w)
    est, s = c @ b, np.sqrt(c @ V @ c)
    mres.append(dict(model=nm, reversal=est * 100, ci_low=(est - 1.96 * s) * 100, ci_high=(est + 1.96 * s) * 100,
                     placebo_P0=b[0] * 100, placebo_low=(b[0] - 1.96 * se[0]) * 100, placebo_high=(b[0] + 1.96 * se[0]) * 100,
                     treated_prs=int((D.treated == 1).sum()), comparison_prs=int((D.treated == 0).sum()),
                     comparison_devs=int(D[D.treated == 0].user.nunique())))
    print(nm, f"reversal {ci(est, s)}; placebo {ci(b[0], se[0])}")
M = pd.DataFrame(mres); M.to_csv(f"{BASE}/table1_weighted_model.csv", index=False)

eb_line = (f"Entropy balancing: {len(wd)} comparison developers with P0 pull requests; effective number {wd.sum()**2/(wd**2).sum():.1f}; "
           f"largest weight {wd.max():.2f} (weights scaled to mean 1). Balanced exactly on the P0 share merged within 7 days, "
           f"the P0 share in the author's own repository and P0 pull requests per developer.")
md = ["# Table 1. Baseline balance and treated developers by phase", "",
      "All pull requests the developers opened (tagged or not), May 26 to July 15, 2026; the 297 treated developers of the main design and the sampled comparison developers. Panel A uses P0 (before the Fable 5 release) only. SMD = standardized difference (difference in means over the pooled SD; pull requests per developer, comments and description length on the log(1+x) scale). Panel B SMDs compare each phase with P2, the suspension.", "",
      "## Panel A. Treated vs comparison, P0", "",
      "| Variable | Treated | Comparison | Comparison, weighted | SMD | SMD, weighted |", "|---|---|---|---|---|---|"]
for r in rows:
    md.append(f"| {r['variable']} | {r['A: Treated']} | {r['A: Comparison']} | {r['A: Comparison, weighted']} | {r['A: SMD']} | {r['A: SMD, weighted']} |")
md += ["", eb_line, "", "## Panel B. Treated developers by phase", "",
       "| Variable | P0 | P1 | P2 | P3 | SMD P0 vs P2 | SMD P1 vs P2 | SMD P3 vs P2 |", "|---|---|---|---|---|---|---|---|"]
for r in rows:
    md.append(f"| {r['variable']} | {r['B: treated P0']} | {r['B: treated P1']} | {r['B: treated P2']} | {r['B: treated P3']} | {r['B: SMD P0 vs P2']} | {r['B: SMD P1 vs P2']} | {r['B: SMD P3 vs P2']} |")
md += ["", "## Suspension model with entropy-balanced weights", "",
       "Seven-day merging on all pull requests; developer and calendar-day fixed effects; CR1 by developer; percentage points.", "",
       "| Model | Reversal contrast [95% CI] | P0 placebo [95% CI] | Treated PRs | Comparison PRs (developers) |", "|---|---|---|---|---|"]
for r in mres:
    md.append(f"| {r['model']} | {r['reversal']:+.1f} [{r['ci_low']:+.1f}, {r['ci_high']:+.1f}] | {r['placebo_P0']:+.1f} [{r['placebo_low']:+.1f}, {r['placebo_high']:+.1f}] | {r['treated_prs']:,} | {r['comparison_prs']:,} ({r['comparison_devs']}) |")
open(f"{BASE}/table1.md", "w").write("\n".join(md) + "\n")
print("written")
