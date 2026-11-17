"""Split by who can merge: author without write access (someone else must merge) vs author with write access."""
import os, sys, numpy as np, warnings
warnings.filterwarnings("ignore")
BASE = sys.argv[1]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ci, load_suspension, fit
D = load_suspension(BASE)
D_all = D.copy()
D_all["external"] = ~D_all.assoc.isin(["OWNER", "MEMBER", "COLLABORATOR"])
t = D_all[D_all.treated == 1]
print("treated PRs by author association:", t.assoc.value_counts().to_dict())
print("external share of treated PRs by phase:", t.groupby("phase").external.mean().mul(100).round(1).to_dict())
print("merged within 7 days, treated P2: external %.1f%%, write access %.1f%%" % (
    t[(t.phase == "P2") & t.external].merged_7d.mean() * 100, t[(t.phase == "P2") & ~t.external].merged_7d.mean() * 100))
for label, mask in [("write access (can self-merge)", ~D_all.external.values), ("no write access (someone else merges)", D_all.external.values)]:
    for unit in ["developer", "developer x repo"]:
        D = D_all[mask].copy()
        if unit == "developer x repo":
            D["user"] = D["user"] + "|" + D["repo"]
        D = D.reset_index(drop=True)
        nt = int(D.treated.sum()); dt_ = D[D.treated == 1].user.nunique()
        out = []
        for y in ["merged_7d", "closed_unmerged"]:
            b, se, V = fit(D, y, ["T_P0", "T_P1", "T_P3"], np.ones(len(D)))
            c = np.array([0, .5, .5])
            b2, se2, _ = fit(D, y, ["fable_pr"], np.ones(len(D)))
            out.append(f"{y}: reversal {ci(c @ b, np.sqrt(c @ V @ c))}, as used {ci(b2[0], se2[0])}")
        print(f"\n[{label}] units={unit}; treated PRs {nt}, treated units {dt_}")
        for o in out: print("   ", o)
# how many treated developers contribute Fable and Opus 4.8 PRs in external repos
e = D_all[(D_all.treated == 1) & D_all.external & D_all.model.isin(["Fable 5", "Opus 4.8"])]
g = e.groupby(["user", "model"]).size().unstack(fill_value=0)
both = g[(g.get("Fable 5", 0) > 0) & (g.get("Opus 4.8", 0) > 0)]
print("\nexternal: treated developers with both Fable and Opus 4.8 PRs:", len(both), "PRs:", int(both.values.sum()))
print(e.groupby("model").agg(prs=("html_url", "size"), devs=("user", "nunique"), merged7=("merged_7d", "mean"), closed=("closed_unmerged", "mean")).round(3))
