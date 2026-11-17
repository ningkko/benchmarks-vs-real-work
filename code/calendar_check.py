"""Exploratory calendar control for the upgrade events (planned before the comparison data were
collected; not part of the pre-specified analysis).

For each event: switchers' pull requests (older model in the 7 days before the release, newer model in the 7 days
after) plus "stayers", developers whose pull requests name a comparison model in both windows and who never name
the event's older or newer model there. Seven-day merge on (switcher x after) with developer and calendar-day fixed
effects, SE clustered by developer (CR1), so the day effects come from the stayers.

usage: python3 -I calendar_check.py BASE
"""
import json, sys, os, warnings
import numpy as np, pandas as pd
from scipy import stats
warnings.filterwarnings("ignore")
BASE = sys.argv[1]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import T, prep, EVENTS, SWEPRO, DAY, load_event, fe_ols, pool, COMP, twfe

out = []
for k, d, old, new, f_old, f_new in EVENTS:
    if k not in COMP:
        continue
    A, B = load_event(BASE, f_old), load_event(BASE, f_new)
    d0 = T(d)
    a = A[(A.model == old) & (A.created_at >= d0 - 7 * DAY) & (A.created_at < d0)].copy()
    b = B[(B.model == new) & (B.created_at >= d0 + DAY) & (B.created_at < d0 + 8 * DAY)].copy()
    sw = set(a.user) & set(b.user)
    S_ = pd.concat([a[a.user.isin(sw)], b[b.user.isin(sw)]], ignore_index=True).drop_duplicates("html_url")
    S_["treat"] = 1.0
    cm, files = COMP[k]
    C = prep(sum([[json.loads(l) for l in open(f"{BASE}/{f}")] for f in files], []))
    C = C[C.model == cm].drop_duplicates("html_url")
    cb = C[(C.created_at >= d0 - 7 * DAY) & (C.created_at < d0)]
    ca = C[(C.created_at >= d0 + DAY) & (C.created_at < d0 + 8 * DAY)]
    stay = (set(cb.user) & set(ca.user)) - (set(a.user) | set(b.user))
    Cn = pd.concat([cb[cb.user.isin(stay)], ca[ca.user.isin(stay)]], ignore_index=True)
    Cn["treat"] = 0.0
    D = pd.concat([S_, Cn], ignore_index=True).drop_duplicates("html_url")
    D["after"] = (D.created_at >= d0 + DAY).astype(float)
    D["day"] = D.created_at.dt.strftime("%Y-%m-%d")
    D["y"] = D.merged_7d.astype(float)
    D["sw_after"] = D.treat * D.after
    bb, V = twfe(D, "y", ["sw_after"])
    St = D[D.treat == 0].copy()
    bs, Vs = fe_ols(St, "y", ["after"])
    Sw = D[D.treat == 1].copy()
    bw, Vw = fe_ols(Sw, "y", ["after"])
    pred = None
    if old in SWEPRO and new in SWEPRO:
        pred = (1 - Sw[Sw.after == 0].y.mean()) * (1 - (100 - SWEPRO[new]) / (100 - SWEPRO[old])) * 100
    out.append(dict(event=k, comparison=cm, switchers=len(sw), sw_prs=len(Sw), stayers=len(stay), stay_prs=len(St),
                    stay_before=St[St.after == 0].y.mean() * 100, stay_after=St[St.after == 1].y.mean() * 100,
                    sw_change=bw[0] * 100, sw_se=np.sqrt(Vw[0, 0]) * 100,
                    stay_change=bs[0] * 100, stay_se=np.sqrt(Vs[0, 0]) * 100,
                    did=bb[0] * 100, did_se=np.sqrt(V[0, 0]) * 100, pred=pred))

R = pd.DataFrame(out)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(R.round(2).to_string(index=False))
R.to_csv(f"{BASE}/calendar/calendar_estimates.csv", index=False)


def show(label, ev):
    h = R[R.event.isin(ev)]
    p = pool(h.did, h.did_se)
    print(f"{label} events {list(h.event)}: calendar-controlled change FE {p['fe']:+.2f} "
          f"[{p['fe']-1.96*p['fe_se']:+.2f}, {p['fe']+1.96*p['fe_se']:+.2f}] p={p['p_fe']:.4f} | RE {p['re']:+.2f} I2={p['I2']:.2f}")
    hp = h[h.pred.notna()]
    if len(hp) and len(hp) == len(h):
        pr = (hp.pred / hp.did_se ** 2).sum() / (1 / hp.did_se ** 2).sum()
        gap = p['fe'] - pr
        print(f"   vs proportional-transfer prediction {pr:+.2f}: difference {gap:+.2f} "
              f"[{gap-1.96*p['fe_se']:+.2f}, {gap+1.96*p['fe_se']:+.2f}] p={2*(1-stats.norm.cdf(abs(gap/p['fe_se']))):.4f}")
    ps = pool(h.stay_change, h.stay_se)
    print(f"   stayers' own change FE {ps['fe']:+.2f} [{ps['fe']-1.96*ps['fe_se']:+.2f}, {ps['fe']+1.96*ps['fe_se']:+.2f}]")


show("included (pre-specified inclusion rule)", [1, 3, 4, 6, 7, 8])
show("benchmark comparison, strict", [3, 4, 6, 7, 8])
show("benchmark comparison, lenient", [3, 4, 5, 6, 7, 8])
