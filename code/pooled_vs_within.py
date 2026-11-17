"""July-August (after restoration), all tagged Fable 5 and Opus 4.8 PRs on sampled days:
pooled difference vs within-developer and within-developer-and-repo differences (day fixed effects)."""
import os, json, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
BASE = sys.argv[1]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import prep, T, fit, ci
rng = np.random.default_rng(11)
rows = []
for f in ["weekly_fable.jsonl", "weekly_opus48.jsonl"]:
    rows += [json.loads(l) for l in open(f"{BASE}/data_fable/{f}")]
D = prep(rows)
D = D[D.model.isin(["Fable 5", "Opus 4.8"]) & (D.created_at >= T("2026-07-02"))].reset_index(drop=True)
D["day"] = D.created_at.dt.strftime("%Y-%m-%d")
D["fable_pr"] = (D.model == "Fable 5").astype(float)
D["external"] = ~D.assoc.isin(["OWNER", "MEMBER", "COLLABORATOR"])
f, o = D[D.model == "Fable 5"], D[D.model == "Opus 4.8"]
fu = {u: x.merged_7d.values for u, x in f.groupby("user")}; ou = {u: x.merged_7d.values for u, x in o.groupby("user")}
bs = [(np.concatenate([fu[k] for k in rng.choice(list(fu), len(fu))]).mean() - np.concatenate([ou[k] for k in rng.choice(list(ou), len(ou))]).mean()) * 100 for _ in range(2000)]
print(f"Jul 8 - Aug 19, {D.day.nunique()} days: Fable {len(f)} PRs / {f.user.nunique()} devs, merged7 {f.merged_7d.mean()*100:.1f}%; Opus 4.8 {len(o)} PRs / {o.user.nunique()} devs, merged7 {o.merged_7d.mean()*100:.1f}%")
print(f"pooled difference {(f.merged_7d.mean()-o.merged_7d.mean())*100:+.1f} [{np.percentile(bs,2.5):+.1f}, {np.percentile(bs,97.5):+.1f}]")
print(f"closed without merge: Fable {f.closed_unmerged.mean()*100:.1f}% vs Opus 4.8 {o.closed_unmerged.mean()*100:.1f}%")
print(f"own repo share: Fable {f.own_repo.mean()*100:.1f}% vs Opus 4.8 {o.own_repo.mean()*100:.1f}%; lacks write access: Fable {f.external.mean()*100:.1f}% vs {o.external.mean()*100:.1f}%")
D_keep = D.copy()
# day fixed effects only (pooled, adjusted for day)
b, se, _ = fit(D, "merged_7d", ["fable_pr"], np.ones(len(D)), fe=("day",))
print("pooled, day fixed effects only (clustered by developer):", ci(b[0], se[0]))
both = set(f.user) & set(o.user)
print("developers with both models:", len(both))
b, se, _ = fit(D, "merged_7d", ["fable_pr"], np.ones(len(D)))
print("within developer (+ day FE):", ci(b[0], se[0]))
D["user"] = D["user"] + "|" + D["repo"]
b, se, _ = fit(D, "merged_7d", ["fable_pr"], np.ones(len(D)))
print("within developer and repository (+ day FE):", ci(b[0], se[0]))

print("\n-- by who can merge, Jul 8 - Aug 19 sample --")
D_base = D_keep.copy()
for label, m in [("author has write access", ~D_base.external), ("author lacks write access", D_base.external)]:
    D = D_base[m.values].reset_index(drop=True)
    f2, o2 = D[D.model == "Fable 5"], D[D.model == "Opus 4.8"]
    b0, se0, _ = fit(D, "merged_7d", ["fable_pr"], np.ones(len(D)), fe=("day",))
    b1, se1, _ = fit(D, "merged_7d", ["fable_pr"], np.ones(len(D)))
    D["user"] = D["user"] + "|" + D["repo"]
    b2, se2, _ = fit(D, "merged_7d", ["fable_pr"], np.ones(len(D)))
    print(f"{label:26s} PRs {len(D)} (Fable {len(f2)}, Opus {len(o2)}); merged7 {f2.merged_7d.mean()*100:.1f} vs {o2.merged_7d.mean()*100:.1f}; pooled+day {ci(b0[0],se0[0])}; within dev {ci(b1[0],se1[0])}; within dev x repo {ci(b2[0],se2[0])}")
