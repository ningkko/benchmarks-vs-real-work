"""Merged pull requests reverted within 30 days.
A revert is a pull request in the same repository with "revert" in the title that quotes the original title
or names the original number, opened within 30 days after the merge.

usage: python3 -I revert.py BASE
"""
import os, sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
BASE = sys.argv[1]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ci, load_suspension, fit, EVENTS, fe_ols, pool, load_reverts, mark, switchers
D = load_suspension(BASE)
D_all = D.copy()


print("== suspension sample")
R = load_reverts(f"{BASE}/revert/reverts_suspension.jsonl")
print(f"   revert-titled pull requests collected: {len(R)} in {R.repo.nunique()} repositories")
T_ = D_all[(D_all.treated == 1)].copy()
T_["merged_at"] = pd.to_datetime(T_.merged_at, utc=True)
T_["reverted_30d"] = mark(T_, R)
mg = T_[T_.merged_at.notna()]
print(f"   treated merged pull requests {len(mg)}; reverted within 30 days {int(mg.reverted_30d.sum())} ({mg.reverted_30d.mean()*100:.2f}%)")
print("   by phase:", {p: f"{mg[mg.phase==p].reverted_30d.mean()*100:.2f}% (n={int((mg.phase==p).sum())})" for p in ["P0", "P1", "P2", "P3"]})
print("   by model:", {m: f"{mg[mg.model==m].reverted_30d.mean()*100:.2f}% (n={int((mg.model==m).sum())})" for m in ["Fable 5", "Opus 4.8"]})
D_all["reverted_30d"] = 0.0
D_all.loc[T_.index, "reverted_30d"] = T_.reverted_30d.astype(float)
D_all["merged_not_reverted"] = ((D_all.merged_7d == 1) & (D_all.reverted_30d == 0)).astype(float)
D = D_all[(D_all.treated == 0) | D_all.merged_at.notna()].reset_index(drop=True)
D["merged_at"] = D.merged_at  # unchanged
b, se, V = fit(D, "reverted_30d", ["T_P0", "T_P1", "T_P3"], np.ones(len(D)))
c = np.array([0, .5, .5]); b2, se2, _ = fit(D, "reverted_30d", ["fable_pr"], np.ones(len(D)))
print(f"   reverted within 30 days (merged pull requests; control rows carry the day effects with reverted = 0): "
      f"reversal {ci(c @ b, np.sqrt(c @ V @ c))}, as used {ci(b2[0], se2[0])}")
D = D_all.reset_index(drop=True)
b, se, V = fit(D, "merged_not_reverted", ["T_P0", "T_P1", "T_P3"], np.ones(len(D)))
b2, se2, _ = fit(D, "merged_not_reverted", ["fable_pr"], np.ones(len(D)))
print(f"   merged within 7 days and not reverted within 30: reversal {ci(c @ b, np.sqrt(c @ V @ c))}, as used {ci(b2[0], se2[0])}")

import os
if os.path.exists(f"{BASE}/revert/reverts_events.jsonl"):
    print("\n== upgrade events")
    R_events = load_reverts(f"{BASE}/revert/reverts_events.jsonl")
    print(f"   revert-titled pull requests collected: {len(R_events)} in {R_events.repo.nunique()} repositories")
    rows = []
    for k, d, old, new, f_old, f_new in EVENTS:
        if k not in [3, 4, 6, 7, 8]:
            continue
        a, b_, sw = switchers(BASE, d, old, new, f_old, f_new)
        a["new"] = 0.0; b_["new"] = 1.0
        E = pd.concat([a[a.user.isin(sw)], b_[b_.user.isin(sw)]], ignore_index=True)
        E["merged_at"] = pd.to_datetime(E.merged_at, utc=True)
        E["reverted_30d"] = mark(E, R_events).astype(float)
        E["y"] = ((E.merged_7d == 1) & (E.reverted_30d == 0)).astype(float)
        m = E[E.merged_at.notna()]
        rate = m.groupby("new").reverted_30d.mean() * 100
        bb, V = fe_ols(E, "y", ["new"])
        rows.append(dict(event=k, prs=len(E), merged=len(m), rev_before=rate.get(0.0, np.nan), rev_after=rate.get(1.0, np.nan),
                         n_before=int((m.new == 0).sum()), n_after=int((m.new == 1).sum()),
                         k_before=int(m[m.new == 0].reverted_30d.sum()), k_after=int(m[m.new == 1].reverted_30d.sum()),
                         est=bb[0] * 100, se=np.sqrt(V[0, 0]) * 100))
    Rv = pd.DataFrame(rows)
    print(Rv.round(2).to_string(index=False))
    print(f"   reverted within 30 days, all five events: before the switch {int(Rv.k_before.sum())} of {int(Rv.n_before.sum())} merged, "
          f"after {int(Rv.k_after.sum())} of {int(Rv.n_after.sum())}")
    p = pool(Rv.est, Rv.se)
    print(f"   merged within 7 days and not reverted within 30, five events pooled: {p['fe']:+.1f} [{p['fe']-1.96*p['fe_se']:+.1f}, {p['fe']+1.96*p['fe_se']:+.1f}]")
