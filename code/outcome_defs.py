"""Alternative merge-outcome definitions and time to merge.

usage: python3 -I outcome_defs.py BASE
writes BASE/outcome_defs.csv, BASE/time_to_merge.csv, BASE/outcome_defs.md
"""
import os, sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
BASE = sys.argv[1]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ci, load_suspension, fit, EVENTS, fe_ols, pool, load_reverts, mark, switchers
D = load_suspension(BASE)
D_all = D.copy()

def add_outcomes(df):
    df["merged_at"] = pd.to_datetime(df.merged_at, utc=True)
    hrs = (df.merged_at - df.created_at).dt.total_seconds() / 3600
    df["hrs"] = hrs
    df["merged_1h"] = (df.merged_at.notna() & (hrs <= 1)).astype(float)
    df["merged_1d"] = (df.merged_at.notna() & (hrs <= 24)).astype(float)
    df["merged_7d"] = (df.merged_at.notna() & (hrs <= 168)).astype(float)
    df["merged_28d"] = (df.merged_at.notna() & (hrs <= 672)).astype(float)
    return df

D_all = add_outcomes(D_all)
R = load_reverts(f"{BASE}/revert/reverts_suspension.jsonl")
D_all["reverted_30d"] = 0.0
tr = D_all.treated == 1
D_all.loc[tr, "reverted_30d"] = mark(D_all[tr], R).astype(float).values
D_all["merged_7d_not_rev"] = ((D_all.merged_7d == 1) & (D_all.reverted_30d == 0)).astype(float)

OUT = [("merged_1h", "merged within 1 hour"), ("merged_1d", "merged within 1 day"),
       ("merged_7d", "merged within 7 days (main)"), ("merged_28d", "merged within 28 days"),
       ("merged_7d_not_rev", "merged within 7 days, not reverted within 30")]
rows = []
c = np.array([0, .5, .5])
D = D_all.reset_index(drop=True)
for y, lab in OUT:
    b, se, V = fit(D, y, ["T_P0", "T_P1", "T_P3"], np.ones(len(D)))
    est, s = c @ b, np.sqrt(c @ V @ c)
    tm = D[D.treated == 1]
    rows.append(dict(design="suspension", outcome=lab, estimate=est * 100, ci_low=(est - 1.96 * s) * 100, ci_high=(est + 1.96 * s) * 100,
                     placebo_P0=b[0] * 100, placebo_low=(b[0] - 1.96 * se[0]) * 100, placebo_high=(b[0] + 1.96 * se[0]) * 100,
                     level_ref=tm[tm.phase == "P2"][y].mean() * 100, n_prs=len(D)))
    print(f"suspension {lab}: reversal {ci(est, s)}; placebo {ci(b[0], se[0])}")

# time to merge, suspension: medians by phase, and the main model on log(1+hours) among merged
tt = []
mg = D[D.merged_at.notna()].copy()
for p in ["P0", "P1", "P2", "P3"]:
    x = mg[(mg.treated == 1) & (mg.phase == p)].hrs
    tt.append(dict(design="suspension", group=f"treated {p}", n_merged=len(x), median_hours=x.median(), q1=x.quantile(.25), q3=x.quantile(.75)))
x = mg[mg.treated == 0].hrs
tt.append(dict(design="suspension", group="comparison, all phases", n_merged=len(x), median_hours=x.median(), q1=x.quantile(.25), q3=x.quantile(.75)))
mg["loghrs"] = np.log1p(mg.hrs.clip(lower=0))
D = mg.reset_index(drop=True)
b, se, V = fit(D, "loghrs", ["T_P0", "T_P1", "T_P3"], np.ones(len(D)))
est, s = c @ b, np.sqrt(c @ V @ c)
within_susp = dict(design="suspension", contrast="Fable phases minus suspension, log(1+hours), developer and day FE",
                   estimate=est, ci_low=est - 1.96 * s, ci_high=est + 1.96 * s, pct_change=(np.exp(est) - 1) * 100,
                   pct_low=(np.exp(est - 1.96 * s) - 1) * 100, pct_high=(np.exp(est + 1.96 * s) - 1) * 100)
print("suspension time to merge:", within_susp)

# upgrades
R_events = load_reverts(f"{BASE}/revert/reverts_events.jsonl")
per = {y: [] for y, _ in OUT}; per_t = []
for k, d, old, new, f_old, f_new in EVENTS:
    if k not in [3, 4, 6, 7, 8]:
        continue
    a, b_, sw = switchers(BASE, d, old, new, f_old, f_new)
    a["new"] = 0.0; b_["new"] = 1.0
    E = pd.concat([a[a.user.isin(sw)], b_[b_.user.isin(sw)]], ignore_index=True).drop_duplicates("html_url")
    E = add_outcomes(E)
    E["reverted_30d"] = mark(E, R_events).astype(float)
    E["merged_7d_not_rev"] = ((E.merged_7d == 1) & (E.reverted_30d == 0)).astype(float)
    for y, _ in OUT:
        bb, VV = fe_ols(E, y, ["new"])
        per[y].append((k, bb[0] * 100, np.sqrt(VV[0, 0]) * 100, len(E)))
    m = E[E.merged_at.notna()].copy()
    for nn, lab in [(0.0, "before"), (1.0, "after")]:
        x = m[m.new == nn].hrs
        tt.append(dict(design="upgrades", group=f"event {k} {lab}", n_merged=len(x), median_hours=x.median(), q1=x.quantile(.25), q3=x.quantile(.75)))
    m["loghrs"] = np.log1p(m.hrs.clip(lower=0))
    bb, VV = fe_ols(m, "loghrs", ["new"])
    per_t.append((k, bb[0], np.sqrt(VV[0, 0])))
    allm = m
    tt_pool_before = None

for y, lab in OUT:
    P = pd.DataFrame(per[y], columns=["event", "est", "se", "n"])
    p = pool(P.est, P.se)
    rows.append(dict(design="upgrades (5 events pooled)", outcome=lab, estimate=p["fe"], ci_low=p["fe"] - 1.96 * p["fe_se"],
                     ci_high=p["fe"] + 1.96 * p["fe_se"], placebo_P0=np.nan, placebo_low=np.nan, placebo_high=np.nan,
                     level_ref=np.nan, n_prs=int(P.n.sum()), I2=p["I2"]))
    print(f"upgrades {lab}: {p['fe']:+.1f} [{p['fe']-1.96*p['fe_se']:+.1f}, {p['fe']+1.96*p['fe_se']:+.1f}]  I2={p['I2']:.2f}")
Pt = pd.DataFrame(per_t, columns=["event", "est", "se"]); p = pool(Pt.est, Pt.se)
within_up = dict(design="upgrades", contrast="after minus before, log(1+hours), developer FE, 5 events pooled",
                 estimate=p["fe"], ci_low=p["fe"] - 1.96 * p["fe_se"], ci_high=p["fe"] + 1.96 * p["fe_se"],
                 pct_change=(np.exp(p["fe"]) - 1) * 100, pct_low=(np.exp(p["fe"] - 1.96 * p["fe_se"]) - 1) * 100,
                 pct_high=(np.exp(p["fe"] + 1.96 * p["fe_se"]) - 1) * 100)
print("upgrades time to merge:", within_up)

Rdf = pd.DataFrame(rows); TT = pd.DataFrame(tt); W = pd.DataFrame([within_susp, within_up])
Rdf.to_csv(f"{BASE}/outcome_defs.csv", index=False)
TT.to_csv(f"{BASE}/time_to_merge.csv", index=False)
W.to_csv(f"{BASE}/time_to_merge_within.csv", index=False)

f = lambda v: f"{v:+.1f}"
md = ["# Outcome definitions and time to merge", "",
      "Data collected up to Nov 2, 2026. Percentage points with 95% CI. Suspension: reversal contrast (P1+P3)/2 minus P2, developer and calendar-day fixed effects, CR1 by developer, one weight per pull request. Upgrades: overall change after minus before among switchers, developer fixed effects, events 3, 4, 6, 7, 8 pooled by inverse-variance weighting.", "",
      "## Outcome definitions", "",
      "| Outcome | Suspension: reversal [95% CI] | Suspension: P0 placebo [95% CI] | Treated level in P2 (%) | Upgrades, 5 pooled [95% CI] |",
      "|---|---|---|---|---|"]
for y, lab in OUT:
    s_ = Rdf[(Rdf.design == "suspension") & (Rdf.outcome == lab)].iloc[0]
    u_ = Rdf[(Rdf.design != "suspension") & (Rdf.outcome == lab)].iloc[0]
    md.append(f"| {lab} | {f(s_.estimate)} [{f(s_.ci_low)}, {f(s_.ci_high)}] | {f(s_.placebo_P0)} [{f(s_.placebo_low)}, {f(s_.placebo_high)}] | {s_.level_ref:.1f} | {f(u_.estimate)} [{f(u_.ci_low)}, {f(u_.ci_high)}] |")
md += ["", "Not reverted: in the suspension model reverts were collected for treated repositories only, so comparison rows carry 0 (as in revert.py).", "",
       "## Hours from creation to merge (merged pull requests)", "", "| Group | Merged pull requests | Median hours | IQR |", "|---|---|---|---|"]
for r in TT.itertuples():
    md.append(f"| {r.design}: {r.group} | {r.n_merged:,} | {r.median_hours:.2f} | {r.q1:.2f} to {r.q3:.2f} |")
md += ["", "## Within-developer difference in time to merge", "", "| Design | Contrast | Change in log(1+hours) [95% CI] | As % change [95% CI] |", "|---|---|---|---|"]
for r in W.itertuples():
    md.append(f"| {r.design} | {r.contrast} | {r.estimate:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}] | {r.pct_change:+.1f}% [{r.pct_low:+.1f}%, {r.pct_high:+.1f}%] |")
open(f"{BASE}/outcome_defs.md", "w").write("\n".join(md) + "\n")
print("written")
