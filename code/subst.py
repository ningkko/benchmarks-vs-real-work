"""Did treated developers leave Claude, or GitHub, during the suspension?
All pull requests (with or without a Claude co-author line) of the treated developers and of the sampled control
developers, May 26 to July 16, 2026, by suspension phase.

usage: python3 -I subst.py BASE
"""
import os, json, sys
import numpy as np, pandas as pd
BASE = sys.argv[1]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CLAUDE_LINE, prep, PHASES, phase

LEN = {nm: (b - a).total_seconds() / 86400 for nm, a, b in PHASES}


def load(grp):
    recs = [json.loads(l) for l in open(f"{BASE}/subst/{grp}_all.jsonl")]
    P = prep([r for r in recs if not r.get("_done")])
    P = phase(P)
    P = P[P.phase.isin(LEN)].copy()
    P["claude"] = P.body.fillna("").str.contains(CLAUDE_LINE).astype(int)
    logins = sorted({l.strip() for l in open(f"{BASE}/data_fable/{grp}_logins.txt") if l.strip()})
    return P, logins


rows = {}
for grp in ["treated", "control"]:
    P, logins = load(grp)
    idx = pd.MultiIndex.from_product([logins, list(LEN)], names=["user", "phase"])
    n_all = P.groupby(["user", "phase"]).size().reindex(idx, fill_value=0)
    n_cl = P[P.claude == 1].groupby(["user", "phase"]).size().reindex(idx, fill_value=0)
    days = pd.Series([LEN[p] for _, p in idx], index=idx)
    rate = (n_all / days).unstack("phase")
    rate_cl = (n_cl / days).unstack("phase")
    rows[grp] = dict(rate=rate, rate_cl=rate_cl, n_all=n_all.unstack("phase"), n_cl=n_cl.unstack("phase"))
    print(f"== {grp}: {len(logins)} developers, {len(P)} pull requests May 26 to Jul 15 (phases only), {int(P.claude.sum())} with a Claude line")
    for p in LEN:
        a = n_all.xs(p, level="phase")
        c = n_cl.xs(p, level="phase")
        print(f"   {p}: {int((a > 0).sum())} developers with any PR, {int((c > 0).sum())} with a Claude PR; "
              f"{int(a.sum())} PRs, {int(c.sum())} Claude ({c.sum() / max(a.sum(), 1) * 100:.1f}%); "
              f"PRs per developer-day {a.sum() / len(logins) / LEN[p]:.3f}, Claude {c.sum() / len(logins) / LEN[p]:.3f}")

print("\n== difference-in-differences of log(1 + pull requests per day), treated minus control, phase vs P2")
for lab, key in [("all pull requests", "rate"), ("Claude-tagged pull requests", "rate_cl")]:
    t, c = np.log1p(rows["treated"][key]), np.log1p(rows["control"][key])
    for p in ["P0", "P1", "P3"]:
        dt_ = t[p] - t["P2"]; dc = c[p] - c["P2"]
        est = dt_.mean() - dc.mean()
        se = np.sqrt(dt_.var(ddof=1) / len(dt_) + dc.var(ddof=1) / len(dc))
        pct = lambda v: (np.exp(v) - 1) * 100
        print(f"   {lab}, {p} vs P2: {est:+.3f} [{est - 1.96 * se:+.3f}, {est + 1.96 * se:+.3f}] "
              f"= {pct(est):+.1f}% [{pct(est - 1.96 * se):+.1f}%, {pct(est + 1.96 * se):+.1f}%]; "
              f"treated own change {dt_.mean():+.3f}, control {dc.mean():+.3f}")

print("\n== treated developers active in P2: any PR vs Claude PR")
na, nc = rows["treated"]["n_all"], rows["treated"]["n_cl"]
print(f"   any PR in P2 {int((na['P2'] > 0).sum())}; Claude PR in P2 {int((nc['P2'] > 0).sum())}; "
      f"any PR in P2 but no Claude PR {int(((na['P2'] > 0) & (nc['P2'] == 0)).sum())}")
print(f"   among developers with any PR in P1 and P2: Claude share of their PRs P1 {nc['P1'][(na['P1'] > 0) & (na['P2'] > 0)].sum() / na['P1'][(na['P1'] > 0) & (na['P2'] > 0)].sum() * 100:.1f}%, "
      f"P2 {nc['P2'][(na['P1'] > 0) & (na['P2'] > 0)].sum() / na['P2'][(na['P1'] > 0) & (na['P2'] > 0)].sum() * 100:.1f}%")
