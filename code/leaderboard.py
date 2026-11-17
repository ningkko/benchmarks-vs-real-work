"""Pooled ('leaderboard') comparison of Fable 5 vs Opus 4.8 merge rates, one complete day per week.

usage: python3 -I leaderboard.py BASE
Differences are Fable 5 minus Opus 4.8, percentage points; 95% CI from 2,000 bootstrap resamples of developers.
"""
import os, json, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
BASE = sys.argv[1]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import prep
rng = np.random.default_rng(11)
rows = []
for f in ["weekly_fable.jsonl", "weekly_opus48.jsonl"]:
    rows += [json.loads(l) for l in open(f"{BASE}/data_fable/{f}")]
W = prep(rows)
W = W[W.model.isin(["Fable 5", "Opus 4.8"])].copy()
W["date"] = W.created_at.dt.strftime("%Y-%m-%d")
W["external"] = ~W.assoc.isin(["OWNER", "MEMBER", "COLLABORATOR"])
out = []
for d, g in W.groupby("date"):
    f, o = g[g.model == "Fable 5"], g[g.model == "Opus 4.8"]
    if len(f) < 50 or len(o) < 50:
        continue
    diff = (f.merged_7d.mean() - o.merged_7d.mean()) * 100
    fu = {u: x.merged_7d.values for u, x in f.groupby("user")}; ou = {u: x.merged_7d.values for u, x in o.groupby("user")}
    fk, ok = list(fu), list(ou)
    bs = []
    for _ in range(2000):
        a = np.concatenate([fu[k] for k in rng.choice(fk, len(fk))]); b = np.concatenate([ou[k] for k in rng.choice(ok, len(ok))])
        bs.append((a.mean() - b.mean()) * 100)
    devw = (f.groupby("user").merged_7d.mean().mean() - o.groupby("user").merged_7d.mean().mean()) * 100
    out.append(dict(date=d, n_fable=len(f), dev_fable=f.user.nunique(), n_opus=len(o), dev_opus=o.user.nunique(),
                    merged_fable=round(f.merged_7d.mean() * 100, 1), merged_opus=round(o.merged_7d.mean() * 100, 1),
                    diff=round(diff, 1), lo=round(np.percentile(bs, 2.5), 1), hi=round(np.percentile(bs, 97.5), 1),
                    devw_diff=round(devw, 1),
                    own_fable=round(f.own_repo.mean() * 100, 1), own_opus=round(o.own_repo.mean() * 100, 1),
                    ext_fable=round(f.external.mean() * 100, 1), ext_opus=round(o.external.mean() * 100, 1)))
R = pd.DataFrame(out)
pd.set_option("display.width", 250)
print(R.to_string(index=False))
R.to_csv(f"{BASE}/leaderboard_series.csv", index=False)
both = set(W[W.model == "Fable 5"].user) & set(W[W.model == "Opus 4.8"].user)
print("\ndevelopers in both groups on these days:", len(both), "of", W.user.nunique())
