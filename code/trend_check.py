"""Suspension reversal contrast with a linear trend for treated developers.
The phases are not evenly spaced, so the mean of the P1 and P3 terms does not remove a treated-group trend exactly.

usage: python3 -I trend_check.py BASE
"""
import os, sys
import numpy as np
BASE = sys.argv[1]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import T, ci, PHASES, load_suspension, fit
D = load_suspension(BASE)

mid = {}
for nm, a, b in PHASES:
    s = D[(D.treated == 1) & (D.phase == nm)]
    mid[nm] = s.created_at.mean()
print("treated PR-weighted phase midpoints:", {k: v.strftime("%Y-%m-%d %H:%M") for k, v in mid.items()})
d1 = (mid["P2"] - mid["P1"]).total_seconds() / 86400
d3 = (mid["P3"] - mid["P2"]).total_seconds() / 86400
print(f"days from P1 to P2: {d1:.1f}; from P2 to P3: {d3:.1f}")

D["t_weeks"] = (D.created_at - T("2026-05-26")).dt.total_seconds() / (7 * 86400)
D["trend_tr"] = D.treated * D.t_weeks
for y in ["merged_7d", "closed_unmerged"]:
    b, se, V = fit(D, y, ["T_P0", "T_P1", "T_P3", "trend_tr"], np.ones(len(D)))
    c = np.array([0, 0.5, 0.5, 0]); r = c @ b; rse = np.sqrt(c @ V @ c)
    print(f"{y}: reversal contrast with treated-group linear trend {ci(r, rse)}; trend per week {ci(b[3], se[3])}")
    b0, se0, V0 = fit(D, y, ["T_P0", "T_P1", "T_P3"], np.ones(len(D)))
    c0 = np.array([0, 0.5, 0.5]); print(f"{y}: without the trend {ci(c0 @ b0, np.sqrt(c0 @ V0 @ c0))}")

# per-use scaling. Availability contrasts are intention-to-treat; take-up is the Fable-tagged share by phase.
share = {p: float(D[(D.treated == 1) & (D.phase == p)].fable_pr.mean()) for p in ["P0", "P1", "P2", "P3"]}
print("Fable-tagged share of treated PRs by phase:", {k: round(v * 100, 1) for k, v in share.items()})
dose = ((share["P1"] - share["P2"]) + (share["P3"] - share["P2"])) / 2
b0, se0, V0 = fit(D, "merged_7d", ["T_P0", "T_P1", "T_P3"], np.ones(len(D)))
c0 = np.array([0, 0.5, 0.5]); r0 = c0 @ b0; s0 = np.sqrt(c0 @ V0 @ c0)
print(f"merged_7d: reversal contrast {ci(r0, s0)}; mean take-up difference {dose*100:.1f} points; "
      f"per-use (Wald) {ci(r0 / dose, s0 / dose)}")
for p, j in [("P1", 1), ("P3", 2)]:
    d = share[p] - share["P2"]
    print(f"   {p} term {ci(b0[j], se0[j])} / take-up difference {d*100:.1f} -> per-use {ci(b0[j] / d, se0[j] / d)}")
