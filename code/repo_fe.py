"""As-used contrast with developer-by-repository fixed effects (same developer, same repository)."""
import os, sys, numpy as np, warnings
warnings.filterwarnings("ignore")
BASE = sys.argv[1]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ci, load_suspension, fit
D = load_suspension(BASE)
D_all = D.copy()
for label, mask in [("all repos", np.ones(len(D_all), bool)), ("other people's repos", ~D_all.own_repo.values)]:
    D = D_all[mask].copy()
    D["user"] = D["user"] + "|" + D["repo"]      # developer x repository as the unit
    D = D.reset_index(drop=True)
    for y in ["merged_7d", "closed_unmerged"]:
        b2, se2, _ = fit(D, y, ["fable_pr"], np.ones(len(D)))
        print(f"{label:22s} {y:16s} as used, developer x repo FE: {ci(b2[0], se2[0])}  (units {D.user.nunique()}, PRs {len(D)})")
