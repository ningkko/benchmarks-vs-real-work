"""Weighting check: cap each developer's weight at 20 or 50 pull requests, for the suspension model (twfe)
or the February model (twfe_feb).

usage: python3 -I sens_cap.py BASE twfe|twfe_feb
"""
import os, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_suspension, load_february, fit, ci

BASE, which = sys.argv[1], sys.argv[2]
D = load_suspension(BASE) if which == "twfe" else load_february(BASE)
n_user = D.groupby("user").html_url.transform("size")
top = D.user.value_counts()
print("top developers' share of PRs:", round(top.iloc[0] / len(D) * 100, 1), round(top.iloc[:5].sum() / len(D) * 100, 1), "% (top1, top5)")
for cap in [20, 50]:
    wt = np.minimum(1.0, cap / n_user.values)
    if which == "twfe":
        b, se, V = fit(D, "merged_7d", ["T_P0", "T_P1", "T_P3"], wt)
        c = np.array([0, .5, .5])
        print(f"cap {cap}: reversal {ci(c @ b, np.sqrt(c @ V @ c))}  P0 {ci(b[0], se[0])}")
        b2, se2, _ = fit(D, "merged_7d", ["fable_pr"], wt)
        print(f"cap {cap}: as used {ci(b2[0], se2[0])}")
    else:
        b, se, V = fit(D, "merged_7d", ["new_opus", "new_sonnet", "opus_family"], wt)
        print(f"cap {cap}: Opus {ci(b[0], se[0])}  Sonnet {ci(b[1], se[1])}")
