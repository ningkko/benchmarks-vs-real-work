"""Merging account and size for the sampled treated pull requests.
Sample: every outside-author pull request of the treated developers plus 1,000 insider pull requests (seed 11),
read through the GitHub pull request endpoint (merger/out_*.jsonl).

usage: python3 -I merger.py BASE
"""
import os, glob, json, sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
BASE = sys.argv[1]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ci, load_suspension, fit
D = load_suspension(BASE)
D_all = D.copy()

recs = []
for f in sorted(glob.glob(f"{BASE}/merger/out_*.jsonl")):
    for line in open(f):
        line = line.strip()
        if line:
            try:
                recs.append(json.loads(line))
            except Exception:
                pass
M = pd.DataFrame(recs).drop_duplicates(["repo", "number"])
M["number"] = M.number.astype(int)
S = pd.read_csv(f"{BASE}/merger/sample.csv")
S["number"] = S.number.astype(int)
X = S.merge(M, on=["repo", "number"], how="left")
ok = X[X.error.isna()] if "error" in X else X
print(f"sampled {len(S)} (outside {int((S.grp=='outside').sum())}, insider {int((S.grp=='insider').sum())}); "
      f"looked up {len(M)}; usable {len(ok)}; errors {int(X.error.notna().sum()) if 'error' in X else 0}")

mg = ok[ok.merged == True].copy()
mg["self_merged"] = mg.merged_by.fillna("").str.lower() == mg.user.str.lower()
print("\n== (a) who merged, among merged pull requests")
for grp in ["insider", "outside"]:
    g = mg[mg.grp == grp]
    print(f"   {grp}: {len(g)} merged; merged by the author {g.self_merged.mean()*100:.1f}%; "
          f"by phase " + ", ".join(f"{p} {g[g.phase==p].self_merged.mean()*100:.0f}% (n={int((g.phase==p).sum())})" for p in ["P0", "P1", "P2", "P3"]))
print("   outside-author pull requests merged by a bot account:",
      int(mg[(mg.grp == "outside")].merged_by.fillna("").str.contains(r"\[bot\]|bot$", case=False).sum()))

# (b) someone-else-merges status at the developer x repository level
pair = mg.groupby(["user", "repo"]).self_merged.agg(["sum", "size"]).reset_index()
pair["other_merges"] = pair["sum"] == 0        # no self-merge seen for this developer in this repository
status = {(r.user, r.repo): r.other_merges for r in pair.itertuples()}
print(f"\n== (b) contrasts with merger-based status (developer x repository pairs with a merged pull request: {len(pair)}, "
      f"of which someone else merges: {int(pair.other_merges.sum())})")
D_all["key"] = list(zip(D_all.user, D_all.repo))
D_all["in_sample"] = D_all.html_url.isin(S.html_url)
D_all["other_merges"] = D_all.key.map(status)
for label, mask in [("someone else merges (by merger)", D_all.other_merges == True), ("author merges own work (by merger)", D_all.other_merges == False)]:
    keep = (D_all.treated == 0) | (mask & D_all.in_sample)
    D = D_all[keep].reset_index(drop=True)
    nt = int(D.treated.sum()); nd = D[D.treated == 1].user.nunique()
    b, se, V = fit(D, "merged_7d", ["T_P0", "T_P1", "T_P3"], np.ones(len(D)))
    c = np.array([0, .5, .5])
    b2, se2, _ = fit(D, "merged_7d", ["fable_pr"], np.ones(len(D)))
    print(f"   [{label}] treated PRs {nt}, developers {nd}: reversal {ci(c @ b, np.sqrt(c @ V @ c))}, as used {ci(b2[0], se2[0])}")
# the same split by author association on the same sampled pull requests, for comparison
for label, mask in [("outside author (association)", D_all.html_url.isin(S[S.grp == "outside"].html_url)),
                    ("insider (association), sampled", D_all.html_url.isin(S[S.grp == "insider"].html_url))]:
    D = D_all[(D_all.treated == 0) | mask].reset_index(drop=True)
    b, se, V = fit(D, "merged_7d", ["T_P0", "T_P1", "T_P3"], np.ones(len(D)))
    c = np.array([0, .5, .5]); b2, se2, _ = fit(D, "merged_7d", ["fable_pr"], np.ones(len(D)))
    print(f"   [{label}] treated PRs {int(D.treated.sum())}: reversal {ci(c @ b, np.sqrt(c @ V @ c))}, as used {ci(b2[0], se2[0])}")

# (c) size within developer: Fable-tagged vs Opus 4.8-tagged
print("\n== (c) pull request size, within developer (sampled treated pull requests with size fields)")
Z = ok[ok.additions.notna()].copy()
Z["loglines"] = np.log1p(Z.additions.fillna(0) + Z.deletions.fillna(0))   # zero-valued fields are omitted by the endpoint and recorded as 0
Z["files"] = Z.changed_files.fillna(0)
Z = Z[Z.model.isin(["Fable 5", "Opus 4.8"])]
print(f"   {len(Z)} pull requests, median lines changed: Fable 5 {int(np.expm1(Z[Z.model=='Fable 5'].loglines.median()))}, "
      f"Opus 4.8 {int(np.expm1(Z[Z.model=='Opus 4.8'].loglines.median()))}")
Z["fable"] = (Z.model == "Fable 5").astype(float)
for y, lab in [("loglines", "log(1 + lines changed)"), ("files", "files changed")]:
    # developer fixed effects by demeaning, clustered by developer
    g = Z.groupby("user")
    yv = Z[y] - g[y].transform("mean"); xv = Z.fable - g.fable.transform("mean")
    keep = xv.abs() > 1e-9
    if keep.sum() < 10:
        print(f"   {lab}: too few developers with both models"); continue
    beta = (xv[keep] * yv[keep]).sum() / (xv[keep] ** 2).sum()
    resid = yv[keep] - beta * xv[keep]
    cl = (xv[keep] * resid).groupby(Z.user[keep]).sum()
    se = np.sqrt((cl ** 2).sum()) / (xv[keep] ** 2).sum()
    print(f"   {lab}: Fable 5 minus Opus 4.8 within developer {beta:+.3f} [{beta-1.96*se:+.3f}, {beta+1.96*se:+.3f}] "
          f"({Z.user[keep].nunique()} developers, {int(keep.sum())} pull requests)")
