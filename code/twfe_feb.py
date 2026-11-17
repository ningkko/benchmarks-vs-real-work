"""February upgrades (exploratory): two-way fixed effects (developer and calendar day) on all Opus and Sonnet
4.5 and 4.6 tagged pull requests. Exposure: the pull request names the 4.6 version of its family. Day effects
come from developers on the other family or not yet switched (Opus switched Feb 5, Sonnet Feb 17).
Standard errors clustered by developer.

usage: python3 -I twfe_feb.py BASE
"""
import os, sys, json, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_february, fit, ci

BASE = sys.argv[1]
D = load_february(BASE)
n_user = D.groupby("user").html_url.transform("size")
out = {}
for wname, wt in [("per PR", np.ones(len(D))), ("per developer", 1.0 / n_user.values)]:
    for y in ["merged_7d", "closed_unmerged"]:
        b, se, V = fit(D, y, ["new_opus", "new_sonnet", "opus_family"], wt)
        print(f"[{wname}] {y:16s} Opus 4.6 vs 4.5 {ci(b[0], se[0])} | Sonnet 4.6 vs 4.5 {ci(b[1], se[1])} | Opus vs Sonnet family {ci(b[2], se[2])}")
        out[f"{wname}|{y}"] = {"opus": [b[0], se[0]], "sonnet": [b[1], se[1]], "family": [b[2], se[2]]}
json.dump(out, open(f"{BASE}/twfe_feb_numbers.json", "w"), indent=1)
