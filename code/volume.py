"""Post-hoc check E6: how many pull requests switchers open around an upgrade, against stayers
(planned before the data were collected).

usage: python3 -I volume.py BASE
"""
import json, sys, os, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
BASE = sys.argv[1]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import T, prep, pool
DAY = pd.Timedelta(days=1)

groups = json.load(open(f"{BASE}/volume/groups.json"))
rows = []
for k, g in sorted(groups.items(), key=lambda kv: int(kv[0])):
    recs = [json.loads(l) for l in open(f"{BASE}/volume/v{k}.jsonl")]
    P = prep([r for r in recs if not r.get("_done")]).drop_duplicates("html_url")
    d0 = T(g["date"])
    P["win"] = np.where((P.created_at >= d0 - 7 * DAY) & (P.created_at < d0), "before",
               np.where((P.created_at >= d0 + DAY) & (P.created_at < d0 + 8 * DAY), "after", ""))
    P = P[P.win != ""]
    res = {}
    for grp in ["switchers", "stayers"]:
        users = pd.Index(g[grp])
        n = P[P.user.isin(users)].groupby(["user", "win"]).size().unstack("win").reindex(users).fillna(0)
        m = P[P.user.isin(users)].groupby(["user", "win"]).merged_7d.sum().unstack("win").reindex(users).fillna(0)
        for c in ["before", "after"]:
            n[c] = n[c] if c in n.columns else 0.0
            m[c] = m[c] if c in m.columns else 0.0
        ok = (n.before > 0) & (n.after > 0)
        res[grp] = dict(dev=len(users), dropped=int((~ok).sum()), n_before=n.before.mean(), n_after=n.after.mean(),
                        d_open=np.log(n.after[ok]) - np.log(n.before[ok]),
                        d_merged=np.log1p(m.after) - np.log1p(m.before))
    s, c = res["switchers"], res["stayers"]
    out = dict(event=int(k), switchers=s["dev"], stayers=c["dev"], sw_dropped=s["dropped"], st_dropped=c["dropped"],
               sw_before=s["n_before"], sw_after=s["n_after"], st_before=c["n_before"], st_after=c["n_after"])
    for y in ["d_open", "d_merged"]:
        a, b = s[y], c[y]
        out[f"{y}_est"] = a.mean() - b.mean()
        out[f"{y}_se"] = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        out[f"{y}_sw"] = a.mean(); out[f"{y}_st"] = b.mean()
    rows.append(out)

R = pd.DataFrame(rows)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(R.round(3).to_string(index=False))
R.to_csv(f"{BASE}/volume/volume_estimates.csv", index=False)
for y, lab in [("d_open", "pull requests opened"), ("d_merged", "pull requests merged within 7 days")]:
    p = pool(R[f"{y}_est"], R[f"{y}_se"])
    lo, hi = p["fe"] - 1.96 * p["fe_se"], p["fe"] + 1.96 * p["fe_se"]
    pct = lambda v: (np.exp(v) - 1) * 100
    print(f"{lab}: switchers minus stayers, log change {p['fe']:+.3f} [{lo:+.3f}, {hi:+.3f}] p={p['p_fe']:.4f} "
          f"= {pct(p['fe']):+.1f}% [{pct(lo):+.1f}%, {pct(hi):+.1f}%] | RE {p['re']:+.3f} I2={p['I2']:.2f} k={p['k']}")
    w = 1 / R[f"{y}_se"] ** 2
    print(f"   switchers' own log change {np.average(R[f'{y}_sw'], weights=w):+.3f}, stayers' {np.average(R[f'{y}_st'], weights=w):+.3f} (same weights)")
