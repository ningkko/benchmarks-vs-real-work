"""Post-hoc checks of what might explain the upgrade results (listed before
they were run). Same switchers, stayers and fixed effects as calendar_check.py.

usage: python3 -I posthoc.py BASE
"""
import sys, os, warnings
import numpy as np, pandas as pd
from scipy import stats
warnings.filterwarnings("ignore")
BASE = sys.argv[1]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import EVENTS, INCLUDED, STRICT, WRITE, pool, twfe, build, wtwfe, ratio_gain, fmt_pool

built = {}
for k, d, old, new, f_old, f_new in EVENTS:
    if k in INCLUDED:
        built[k] = (build(BASE, k, d, old, new, f_old, f_new), old, new)

# ---- E1: room to move (switchers' pull requests in the week before) ----
print("== E1 room to move: switchers' PRs in the week before each upgrade")
e1 = []
for k, (D, old, new) in built.items():
    b = D[(D.treat == 1) & (D.after == 0)]
    allm = b.groupby("user").merged_7d.min()
    e1.append(dict(event=k, prs=len(b), switchers=b.user.nunique(), merged7=b.merged_7d.mean() * 100,
                   merged1h=b.merged_1h.mean() * 100, write=b.assoc.isin(WRITE).mean() * 100,
                   all_merged_devs=(allm == 1).mean() * 100))
E1 = pd.DataFrame(e1)
print(E1.round(1).to_string(index=False))
tot = pd.concat([D[(D.treat == 1) & (D.after == 0)] for D, _, _ in built.values()])
allm_all = pd.concat([D[(D.treat == 1) & (D.after == 0)].groupby("user").merged_7d.min() for D, _, _ in built.values()])
print("all six: PRs %d, merged within 7 days %.1f%%, within 1 hour %.1f%%, write access %.1f%%; switchers who merged every PR that week %.1f%% of %d"
      % (len(tot), tot.merged_7d.mean() * 100, tot.merged_1h.mean() * 100, tot.assoc.isin(WRITE).mean() * 100,
         (allm_all == 1).mean() * 100, len(allm_all)))

# ---- E2 / E3: other outcomes, calendar-controlled ----
rows = []
OUT = [("blen", "description length (characters)", 1.0), ("files", "files named in the description", 1.0),
       ("is_feat", "title starts with feat (points)", 100.0), ("merged_1h", "merged within 1 hour (points)", 100.0),
       ("closed_unmerged", "closed without merge (points)", 100.0), ("comments", "comments per PR", 1.0)]
for y, lab, sc in OUT:
    est, se, evs = [], [], []
    for k, (D, old, new) in built.items():
        Dy = D[D[y].notna()]
        if Dy[Dy.treat == 1].after.nunique() < 2 or Dy[Dy.treat == 0].after.nunique() < 2:
            continue
        b, V = twfe(Dy, y, ["sw_after"])
        est.append(b[0]); se.append(np.sqrt(V[0, 0])); evs.append(k)
        rows.append(dict(check="E2/E3", outcome=y, event=k, est=b[0] * sc, se=np.sqrt(V[0, 0]) * sc,
                         sw_before=Dy[(Dy.treat == 1) & (Dy.after == 0)][y].mean() * sc))
    p = pool(est, se)
    print(f"== {lab}: calendar-controlled change {fmt_pool(p, sc, 2 if sc == 1.0 else 1)} events {evs}")

# ---- E4: developers with room (an unmerged PR in the week before) ----
print("== E4 developers with at least one PR not merged within 7 days in the week before")
e4 = []
for k, (D, old, new) in built.items():
    b = D[D.after == 0]
    room = set(b.groupby("user").merged_7d.min().loc[lambda s: s == 0].index)
    Dr = D[D.user.isin(room)]
    if Dr[Dr.treat == 1].after.nunique() < 2 or Dr[Dr.treat == 0].after.nunique() < 2:
        continue
    bb, V = twfe(Dr, "merged_7d", ["sw_after"])
    g = ratio_gain(old, new)
    bu = 1 - Dr[(Dr.treat == 1) & (Dr.after == 0)].merged_7d.mean()
    e4.append(dict(event=k, switchers=Dr[Dr.treat == 1].user.nunique(), stayers=Dr[Dr.treat == 0].user.nunique(),
                   sw_before=(1 - bu) * 100, est=bb[0] * 100, se=np.sqrt(V[0, 0]) * 100,
                   pred=(bu * g * 100) if g is not None else np.nan))
    rows.append(dict(check="E4", outcome="merged_7d", event=k, est=bb[0] * 100, se=np.sqrt(V[0, 0]) * 100,
                     sw_before=(1 - bu) * 100))
E4 = pd.DataFrame(e4)
print(E4.round(2).to_string(index=False))
p = pool(E4.est, E4.se)
print(f"   pooled, included events {list(E4.event)}: {fmt_pool(p)}")
h = E4[E4.event.isin(STRICT) & E4.pred.notna()]
p = pool(h.est, h.se)
pr = (h.pred / h.se ** 2).sum() / (1 / h.se ** 2).sum(); gap = p["fe"] - pr
print(f"   strict benchmark set {list(h.event)}: change {p['fe']:+.2f} vs prediction {pr:+.2f}; difference {gap:+.2f} "
      f"[{gap-1.96*p['fe_se']:+.2f}, {gap+1.96*p['fe_se']:+.2f}] p={2*(1-stats.norm.cdf(abs(gap/p['fe_se']))):.4f}")

# ---- E5: equal weight per developer ----
print("== E5 equal weight per developer (calendar-controlled 7-day merging)")
e5 = []
for k, (D, old, new) in built.items():
    w = 1.0 / D.groupby("user").html_url.transform("size").values
    bb, V = wtwfe(D, "merged_7d", ["sw_after"], w)
    g = ratio_gain(old, new)
    sb = D[(D.treat == 1) & (D.after == 0)]
    bu = 1 - sb.groupby("user").merged_7d.mean().mean()
    e5.append(dict(event=k, est=bb[0] * 100, se=np.sqrt(V[0, 0]) * 100, sw_before_devmean=(1 - bu) * 100,
                   pred=(bu * g * 100) if g is not None else np.nan))
    rows.append(dict(check="E5", outcome="merged_7d", event=k, est=bb[0] * 100, se=np.sqrt(V[0, 0]) * 100,
                     sw_before=(1 - bu) * 100))
E5 = pd.DataFrame(e5)
print(E5.round(2).to_string(index=False))
p = pool(E5.est, E5.se)
print(f"   pooled, included events: {fmt_pool(p)}")
h = E5[E5.event.isin(STRICT) & E5.pred.notna()]
p = pool(h.est, h.se)
pr = (h.pred / h.se ** 2).sum() / (1 / h.se ** 2).sum(); gap = p["fe"] - pr
print(f"   strict benchmark set {list(h.event)}: change {p['fe']:+.2f} vs prediction {pr:+.2f}; difference {gap:+.2f} "
      f"[{gap-1.96*p['fe_se']:+.2f}, {gap+1.96*p['fe_se']:+.2f}] p={2*(1-stats.norm.cdf(abs(gap/p['fe_se']))):.4f}")
pd.DataFrame(rows).to_csv(f"{BASE}/calendar/posthoc_estimates.csv", index=False)
