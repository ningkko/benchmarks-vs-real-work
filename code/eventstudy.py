"""Event study by period for the suspension design.
Same model as twfe.py (developer and calendar-day fixed effects, CR1 by developer), with the treated phase
indicators replaced by treated x period indicators. Periods follow the phase boundaries so that no bin mixes
Fable-available days with suspension days: the week of June 22-28 (inside the suspension) is the reference.

usage: python3 -I eventstudy.py BASE
"""
import os, sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
BASE = sys.argv[1]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import T, load_suspension, fit
D = load_suspension(BASE)

BINS = [("May 26-Jun 1", T("2026-05-26"), T("2026-06-02")),
        ("Jun 2-9 (to 18:00)", T("2026-06-02"), T("2026-06-09 18:00")),
        ("Jun 9-12 (Fable 5 on)", T("2026-06-09 18:00"), T("2026-06-12 21:00")),
        ("Jun 15-21", T("2026-06-15"), T("2026-06-22")),
        ("Jun 22-28 (reference)", T("2026-06-22"), T("2026-06-29")),
        ("Jun 29-30", T("2026-06-29"), T("2026-07-01")),
        ("Jul 2-8", T("2026-07-02"), T("2026-07-09")),
        ("Jul 9-15", T("2026-07-09"), T("2026-07-16"))]
D["bin"] = None
for nm, a, b in BINS:
    D.loc[(D.created_at >= a) & (D.created_at < b), "bin"] = nm
assert D.bin.notna().all()
cols = []
for nm, a, b in BINS:
    if "reference" in nm:
        continue
    c = f"T_{nm}"; D[c] = ((D.treated == 1) & (D.bin == nm)).astype(float); cols.append(c)
b, se, V = fit(D, "merged_7d", cols, np.ones(len(D)))
print("== event study: treated x period, reference Jun 22-28 (suspension); seven-day merge share, percentage points")
print(f"{'period':26s} {'estimate':>24s} {'treated PRs':>12s} {'developers':>11s} {'Fable share %':>14s} {'merged %':>9s}")
rows = []
for nm, a, bb in BINS:
    g = D[(D.treated == 1) & (D.bin == nm)]
    fs = g.model.eq("Fable 5").mean() * 100 if len(g) else float("nan")
    if "reference" in nm:
        est = "0 (reference)"; lo = hi = 0.0; e = 0.0
    else:
        i = cols.index(f"T_{nm}"); e = b[i] * 100; lo = (b[i] - 1.96 * se[i]) * 100; hi = (b[i] + 1.96 * se[i]) * 100
        est = f"{e:+.1f} [{lo:+.1f}, {hi:+.1f}]"
    print(f"{nm:26s} {est:>24s} {len(g):12d} {g.user.nunique():11d} {fs:14.1f} {g.merged_7d.mean()*100:9.1f}")
    rows.append(dict(period=nm, est=e, lo=lo, hi=hi, prs=len(g), devs=g.user.nunique(), fable_share=fs, merged=g.merged_7d.mean() * 100))
pd.DataFrame(rows).to_csv(f"{BASE}/eventstudy_estimates.csv", index=False)
# control group raw shares by bin, for context
print("\ncontrol merged % by period:", {nm: round(D[(D.treated == 0) & (D.bin == nm)].merged_7d.mean() * 100, 1) for nm, _, _ in BINS})
print("treated merged % by period:", {nm: round(D[(D.treated == 1) & (D.bin == nm)].merged_7d.mean() * 100, 1) for nm, _, _ in BINS})
