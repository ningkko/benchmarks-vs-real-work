"""Pre-specified replication across Claude Code model upgrades (analysis plan written 2026-11-02, 19:03 UTC, before these events were collected).

usage: python3 -I events.py BASE
For each event: switchers = developers with >=1 PR naming the older model in the 7 days before the release
and >=1 PR naming the newer model in the 7 days after (release day left out).
Model per event (linear probability, developer fixed effects, SE clustered by developer):
    merged_7d ~ new + external + new:external
Primary: interaction (new:external), pooled by inverse-variance weighting; DerSimonian-Laird as sensitivity.
Secondary: change within each group; overall change vs proportional-transfer prediction (SWE-Bench Pro).
"""
import sys, warnings, os
import numpy as np, pandas as pd
from scipy import stats
warnings.filterwarnings("ignore")
BASE = sys.argv[1]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import T, EVENTS, SWEPRO, load_event, fe_ols, pool

rows = []
for k, d, old, new, f_old, f_new in EVENTS:
    A, B = load_event(BASE, f_old), load_event(BASE, f_new)
    if A is None or B is None:
        print(f"event {k}: data missing"); continue
    d0 = T(d)
    a = A[(A.model == old) & (A.created_at >= d0 - pd.Timedelta(days=7)) & (A.created_at < d0)].copy()
    b = B[(B.model == new) & (B.created_at >= d0 + pd.Timedelta(days=1)) & (B.created_at < d0 + pd.Timedelta(days=8))].copy()
    sw = set(a.user) & set(b.user)
    a["new"] = 0.0; b["new"] = 1.0
    D = pd.concat([a[a.user.isin(sw)], b[b.user.isin(sw)]], ignore_index=True).drop_duplicates("html_url")
    D["external"] = (~D.assoc.isin(["OWNER", "MEMBER", "COLLABORATOR"])).astype(float)
    D["new_x_ext"] = D.new * D.external
    D["y"] = D.merged_7d.astype(float)
    ext_both = set(D[(D.external == 1) & (D.new == 0)].user) & set(D[(D.external == 1) & (D.new == 1)].user)
    bb, V = fe_ols(D, "y", ["new", "external", "new_x_ext"])
    c_ext = np.array([1, 0, 1])
    ch_ext, ch_ext_se = c_ext @ bb, np.sqrt(c_ext @ V @ c_ext)
    b_all, V_all = fe_ols(D, "y", ["new"])
    before_unmerged = 1 - D[D.new == 0].y.mean()
    pred = None
    if old in SWEPRO and new in SWEPRO:
        ratio = (100 - SWEPRO[new]) / (100 - SWEPRO[old])
        pred = before_unmerged * (1 - ratio) * 100
    r = dict(event=k, date=d, old=old, new=new, n_old=len(a), n_new=len(b), switchers=len(sw), prs=len(D),
             ext_prs=int(D.external.sum()), ext_switchers=len(ext_both),
             merged_before=D[D.new == 0].y.mean() * 100, merged_after=D[D.new == 1].y.mean() * 100,
             all_est=b_all[0] * 100, all_se=np.sqrt(V_all[0, 0]) * 100,
             self_est=bb[0] * 100, self_se=np.sqrt(V[0, 0]) * 100,
             ext_est=ch_ext * 100, ext_se=ch_ext_se * 100,
             inter_est=bb[2] * 100, inter_se=np.sqrt(V[2, 2]) * 100, pred=pred)
    rows.append(r)
R = pd.DataFrame(rows)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
show = R.copy()
for c in ["merged_before", "merged_after", "all_est", "all_se", "self_est", "self_se", "ext_est", "ext_se", "inter_est", "inter_se", "pred"]:
    show[c] = show[c].astype(float).round(1)
print(show.to_string(index=False))
R.to_csv(f"{BASE}/events/event_estimates.csv", index=False)

inc = R[R.ext_switchers >= 10]
print(f"\nevents meeting the pre-specified rule (>=10 switchers without write access in both windows): {list(inc.event)}")
for label, e, s in [("PRIMARY interaction (no write access minus write access)", "inter_est", "inter_se"),
                    ("change, author lacks write access", "ext_est", "ext_se"),
                    ("change, author has write access", "self_est", "self_se")]:
    p = pool(inc[e], inc[s])
    print(f"{label}: FE {p['fe']:+.2f} [{p['fe']-1.96*p['fe_se']:+.2f}, {p['fe']+1.96*p['fe_se']:+.2f}] p={p['p_fe']:.4f} | "
          f"RE {p['re']:+.2f} [{p['re']-1.96*p['re_se']:+.2f}, {p['re']+1.96*p['re_se']:+.2f}] p={p['p_re']:.4f} | I2={p['I2']:.2f} k={p['k']}")
p = pool(R.all_est, R.all_se)
print(f"change, all pull requests (all events): FE {p['fe']:+.2f} [{p['fe']-1.96*p['fe_se']:+.2f}, {p['fe']+1.96*p['fe_se']:+.2f}] p={p['p_fe']:.4f} | RE {p['re']:+.2f} I2={p['I2']:.2f}")
h3 = R[R.pred.notna()]
if len(h3):
    p = pool(h3.all_est, h3.all_se)
    pr = (h3.pred / h3.all_se ** 2).sum() / (1 / h3.all_se ** 2).sum()
    gap = p['fe'] - pr
    print(f"H3 events {list(h3.event)}: observed pooled {p['fe']:+.2f} (SE {p['fe_se']:.2f}) vs proportional-transfer prediction {pr:+.2f}; "
          f"difference {gap:+.2f} [{gap-1.96*p['fe_se']:+.2f}, {gap+1.96*p['fe_se']:+.2f}] p={2*(1-stats.norm.cdf(abs(gap/p['fe_se']))):.4f}")

print("\n-- strict reading of the reporting rule: events below the threshold left out of every pool --")
p = pool(inc.all_est, inc.all_se)
print(f"change, all pull requests (events {list(inc.event)}): FE {p['fe']:+.2f} [{p['fe']-1.96*p['fe_se']:+.2f}, {p['fe']+1.96*p['fe_se']:+.2f}] p={p['p_fe']:.4f} | RE {p['re']:+.2f} [{p['re']-1.96*p['re_se']:+.2f}, {p['re']+1.96*p['re_se']:+.2f}] I2={p['I2']:.2f}")
h3s = inc[inc.pred.notna()]
p = pool(h3s.all_est, h3s.all_se)
pr = (h3s.pred / h3s.all_se ** 2).sum() / (1 / h3s.all_se ** 2).sum()
gap = p['fe'] - pr
print(f"H3 events {list(h3s.event)}: observed pooled {p['fe']:+.2f} (SE {p['fe_se']:.2f}) vs prediction {pr:+.2f}; difference {gap:+.2f} "
      f"[{gap-1.96*p['fe_se']:+.2f}, {gap+1.96*p['fe_se']:+.2f}] p={2*(1-stats.norm.cdf(abs(gap/p['fe_se']))):.4f}")
print("switchers total", int(R.switchers.sum()), "| PRs total", int(R.prs.sum()), "| included events switchers", int(inc.switchers.sum()),
      "| included external switchers", int(inc.ext_switchers.sum()))
