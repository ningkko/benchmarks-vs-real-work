"""The suspension design on every pull request the developers opened, tagged or not.
Uses the all-pull-request collection (subst/treated_all.jsonl, subst/control_all.jsonl) and the same model as twfe.py.

usage: python3 -I allprs.py BASE
"""
import os, json, sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
BASE = sys.argv[1]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import prep, ci, phase, load_suspension, fit
D = load_suspension(BASE)
treated297 = set(D[D.treated == 1].user)
D_main = D

A = prep([json.loads(l) for l in open(f"{BASE}/subst/treated_all.jsonl")])
B = prep([json.loads(l) for l in open(f"{BASE}/subst/control_all.jsonl")])
A = phase(A).copy(); B = phase(B).copy()
A = A[A.user.isin(treated297)].copy(); A["treated"] = 1
B = B[~B.user.isin(treated297)].copy(); B["treated"] = 0
D = pd.concat([A, B], ignore_index=True).drop_duplicates("html_url")
D["day"] = D.created_at.dt.strftime("%Y-%m-%d")
for p in ["P0", "P1", "P3"]:
    D[f"T_{p}"] = ((D.treated == 1) & (D.phase == p)).astype(float)
D["fable_pr"] = ((D.treated == 1) & (D.model == "Fable 5")).astype(float)
D["merged_7d"] = D.merged_7d.astype(float)
print(f"all pull requests: treated {int(D.treated.sum())} from {D[D.treated==1].user.nunique()} developers; "
      f"comparison {int((D.treated==0).sum())} from {D[D.treated==0].user.nunique()} developers; days {D.day.nunique()}")
tr = D[D.treated == 1]
print("treated, by phase: " + "; ".join(f"{p} n={int((tr.phase==p).sum())}, merged-7d {tr[tr.phase==p].merged_7d.mean()*100:.1f}%, "
      f"Fable share {tr[tr.phase==p].model.eq('Fable 5').mean()*100:.1f}%, any Claude line {(tr[tr.phase==p].model!='untagged').mean()*100:.1f}%" for p in ["P0", "P1", "P2", "P3"]))
share = {p: float(tr[tr.phase == p].model.eq("Fable 5").mean()) for p in ["P1", "P2", "P3"]}
dose = ((share["P1"] - share["P2"]) + (share["P3"] - share["P2"])) / 2
n_user = D.groupby("user").html_url.transform("size")
c = np.array([0, .5, .5])
for wname, wt in [("per pull request", np.ones(len(D))), ("per developer", 1.0 / n_user.values)]:
    b, se, V = fit(D, "merged_7d", ["T_P0", "T_P1", "T_P3"], wt)
    r = c @ b; s = np.sqrt(c @ V @ c)
    print(f"[{wname}] P0 placebo {ci(b[0], se[0])}; P1 {ci(b[1], se[1])}; P3 {ci(b[2], se[2])}; reversal {ci(r, s)}; "
          f"per Fable 5 pull request on this base (dose {dose*100:.1f} points) {ci(r / dose, s / dose)}")
# developer-level means, treated only: merge share in P1+P3 minus P2, developers active in both
m = tr.groupby(["user", "phase"]).merged_7d.mean().unstack()
both = m[m.P2.notna() & (m.P1.notna() | m.P3.notna())]
d = both[["P1", "P3"]].mean(axis=1) - both.P2
rng = np.random.default_rng(11); bs = [rng.choice(d.values, len(d)).mean() for _ in range(2000)]
print(f"developer-level mean, Fable phases minus suspension, treated developers with pull requests in both: {d.mean()*100:+.1f} "
      f"[{np.percentile(bs,2.5)*100:+.1f}, {np.percentile(bs,97.5)*100:+.1f}], n={len(d)}")
