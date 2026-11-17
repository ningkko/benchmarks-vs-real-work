"""Two-way fixed-effects event study around the Fable 5 suspension.

Outcome ~ treated-phase indicators + developer FE + calendar-day FE.
Reference category: treated developers during the suspension (P2) and all control PRs.
Controls (Opus 4.8 users who never used Fable) only identify the day effects.
Standard errors clustered by developer (CR1). Two weightings: per PR, and per developer (1/n PRs).
usage: python3 -I twfe.py BASE
"""
import json, os, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_suspension, fit, ci

BASE = sys.argv[1]
D = load_suspension(BASE)

res = {}
n_user = D.groupby("user").html_url.transform("size")
for wname, wt in [("per PR", np.ones(len(D))), ("per developer", 1.0 / n_user.values)]:
    for y in ["merged_7d", "closed_unmerged"]:
        b, se, V = fit(D, y, ["T_P0", "T_P1", "T_P3"], wt)
        txt = "  ".join(f"{c} {ci(bb, s)}" for c, bb, s in zip(["P0 (before release)", "P1 (Fable on)", "P3 (Fable back)"], b, se))
        print(f"[{wname}] {y:16s} vs suspension: {txt}")
        # reversal contrast: mean of the two Fable-on phases minus the suspension (cancels a linear trend)
        cvec = np.array([0, 0.5, 0.5]); aba = cvec @ b; aba_se = np.sqrt(cvec @ V @ cvec)
        print(f"[{wname}] {y:16s} reversal contrast (P1+P3)/2 - P2: {ci(aba, aba_se)}")
        b2, se2, _ = fit(D, y, ["fable_pr"], wt)
        print(f"[{wname}] {y:16s} Fable-tagged PR (as used): {ci(b2[0], se2[0])}")
        res[f"{wname}|{y}"] = {"P0": [b[0], se[0]], "P1": [b[1], se[1]], "P3": [b[2], se[2]], "aba": [aba, aba_se], "asused": [b2[0], se2[0]]}
json.dump(res, open(f"{BASE}/twfe_numbers.json", "w"), indent=1)
