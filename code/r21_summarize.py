"""Sample check: tables for part 1 (commit -> pull request) and part 2 (weekly search counts).

usage: python3 -I r21_summarize.py R21_DIR
Reads r21_part2_rows.jsonl and r21_part1_commits.jsonl (whichever exist) and writes
r21_part2.csv, r21_part1_by_phase.csv and r21_summary.md in R21_DIR.
"""
import sys, os, json
import pandas as pd

D = sys.argv[1]
md = ["# Sample check", ""]

p2 = f"{D}/r21_part2_rows.jsonl"
if os.path.exists(p2):
    w = pd.DataFrame([json.loads(l) for l in open(p2)]).drop_duplicates("week", keep="last").sort_values("week")
    cols = ["week", "A_model_named_total", "A_model_named_merged", "A_model_named_share", "A_model_named_lo", "A_model_named_hi",
            "B_default_only_total", "B_default_only_merged", "B_default_only_share", "B_default_only_lo", "B_default_only_hi", "diff_A_minus_B"]
    w[cols].round(1).to_csv(f"{D}/r21_part2.csv", index=False)
    md += ["## Part 2: merged share by week, pull requests naming a model (A) vs Claude Code default line only (B)", "",
           "Bot accounts excluded. Merged share = merged / opened, from GitHub search counts; 95% Wilson intervals.", "",
           "| Week | A opened | A merged % | B opened | B merged % | A − B |", "|---|---:|---:|---:|---:|---:|"]
    for r in w.itertuples():
        md.append(f"| {r.week.replace('..', ' to ')} | {r.A_model_named_total:,} | {r.A_model_named_share:.1f} "
                  f"({r.A_model_named_lo:.1f}–{r.A_model_named_hi:.1f}) | {r.B_default_only_total:,} | "
                  f"{r.B_default_only_share:.1f} ({r.B_default_only_lo:.1f}–{r.B_default_only_hi:.1f}) | {r.diff_A_minus_B:+.1f} |")
    md += ["", f"Range of A − B across {len(w)} weeks: {w.diff_A_minus_B.min():+.1f} to {w.diff_A_minus_B.max():+.1f} points.", ""]

p1 = f"{D}/r21_part1_commits.jsonl"
if os.path.exists(p1):
    c = pd.DataFrame([json.loads(l) for l in open(p1)]).drop_duplicates("sha", keep="last")
    named = c[c.commit_model.notna()]
    withpr = named[named.get("pr", pd.Series(index=named.index, dtype=object)).notna()] if "pr" in named else named.iloc[0:0]
    if len(withpr):
        withpr = withpr.assign(cat=withpr.apply(lambda r: "names model" if r.desc_model else
                                                ("default line only" if r.desc_default else "no Claude line"), axis=1))
        tab = (withpr.groupby("phase").cat.value_counts(normalize=True).unstack(fill_value=0) * 100).round(1)
        tab["pull requests"] = withpr.groupby("phase").size()
        allrow = (withpr.cat.value_counts(normalize=True) * 100).round(1)
        tab.loc["all"] = list(allrow.reindex(tab.columns[:-1]).fillna(0)) + [len(withpr)]
        tab.to_csv(f"{D}/r21_part1_by_phase.csv")
        merged = (withpr.assign(m=withpr.pr_merged.astype(float)).groupby("cat").m.mean() * 100).round(1)
        md += ["## Part 1: what the pull request description says, for commits that name a Claude model", "",
               f"Commits sampled: {len(c):,}; naming a model: {len(named):,}; pull request found: {len(withpr):,}.", "",
               tab.to_markdown(), "", "Merged share by description type (%): " +
               "; ".join(f"{k} {v}" for k, v in merged.items()), ""]
    else:
        md += ["## Part 1", "", f"Commits sampled so far: {len(c):,}; none matched to a pull request yet.", ""]

open(f"{D}/r21_summary.md", "w").write("\n".join(md))
print("\n".join(md))
