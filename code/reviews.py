"""Reviews on the 2,019 sampled pull requests (the merger sample), read from GitHub's
pull request reviews endpoint (reviews/out_*.jsonl). A reviewer is "another account" when its login differs from
the author's, and "bot-like" when the login ends in "bot" or "[bot]".

usage: python3 -I reviews.py BASE
"""
import os, glob, json, re, sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
BASE = sys.argv[1]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ci, load_suspension, fit
D = load_suspension(BASE)
D_all = D.copy()

BOT = re.compile(r"(\[bot\]|bot)$", re.I)
recs = []
for f in sorted(glob.glob(f"{BASE}/reviews/out_*.jsonl")):
    for line in open(f):
        line = line.strip()
        if line:
            try:
                recs.append(json.loads(line))
            except Exception:
                pass
R = pd.DataFrame(recs).drop_duplicates(["repo", "number"])
R["number"] = R.number.astype(int)
S = pd.read_csv(f"{BASE}/merger/sample.csv"); S["number"] = S.number.astype(int)
M = pd.concat([pd.read_json(f, lines=True) for f in sorted(glob.glob(f"{BASE}/merger/out_*.jsonl"))]).drop_duplicates(["repo", "number"])
M["number"] = M.number.astype(int)
X = S.merge(R, on=["repo", "number"], how="left").merge(M[["repo", "number", "merged", "merged_by"]], on=["repo", "number"], how="left")
print(f"sampled {len(S)}; reviews read {int(X.reviews.notna().sum())}; errors {int(X.error.notna().sum()) if 'error' in X else 0}")
X = X[X.reviews.notna() & X.error.isna()].copy()


def feats(row):
    revs = row.reviews or []
    other = [r for r in revs if (r.get("login") or "").lower() != str(row.user).lower()]
    human = [r for r in other if not BOT.search(r.get("login") or "")]
    return pd.Series(dict(n_reviews=len(revs),
                          any_other=bool(other), any_other_human=bool(human),
                          approved_other=any(r.get("state") == "APPROVED" for r in other),
                          approved_other_human=any(r.get("state") == "APPROVED" for r in human),
                          changes_requested=any(r.get("state") == "CHANGES_REQUESTED" for r in other),
                          self_review=any((r.get("login") or "").lower() == str(row.user).lower() for r in revs)))


X = pd.concat([X, X.apply(feats, axis=1)], axis=1)
mg = X[X.merged == True].copy()
mg["self_merged"] = mg.merged_by.fillna("").str.lower() == mg.user.str.lower()
pc = lambda s: f"{s.mean()*100:.1f}%"
print(f"\n== merged pull requests in the sample: {len(mg)}")
print(f"   any review by another account {pc(mg.any_other)}; by another non-bot account {pc(mg.any_other_human)}")
print(f"   APPROVED by another account {pc(mg.approved_other)}; by another non-bot account {pc(mg.approved_other_human)}")
print(f"   CHANGES_REQUESTED by another account {pc(mg.changes_requested)}; a review by the author {pc(mg.self_review)}")
print(f"   merged by the author AND no approval from another non-bot account: {pc(mg.self_merged & ~mg.approved_other_human)}")
print(f"   merged by the author with an approval from another non-bot account: {pc(mg.self_merged & mg.approved_other_human)}")
print(f"   merged by someone else: {pc(~mg.self_merged)}, of which with their or another account's approval {pc((~mg.self_merged) & mg.approved_other_human)}")
for by in ["grp", "phase", "model"]:
    g = mg.groupby(by)
    print(f"   by {by}: " + "; ".join(f"{k} n={len(v)} approved-by-other {pc(v.approved_other_human)}, any-other-review {pc(v.any_other_human)}" for k, v in g))
um = X[X.merged == False]
print(f"\n== unmerged pull requests in the sample: {len(um)}; any review by another non-bot account {pc(um.any_other_human)}; CHANGES_REQUESTED {pc(um.changes_requested)}")

# developer x repository pairs with an outside approval on a merged pull request
pair = mg.groupby(["user", "repo"]).approved_other_human.agg(["sum", "size"]).reset_index()
pair["approved_pair"] = pair["sum"] > 0
status = {(r.user, r.repo): bool(r.approved_pair) for r in pair.itertuples()}
print(f"\n== contrasts by whether the developer x repository pair had an approval from another non-bot account on a merged pull request "
      f"(pairs {len(pair)}, with such an approval {int(pair.approved_pair.sum())})")
D_all["key"] = list(zip(D_all.user, D_all.repo))
D_all["in_sample"] = D_all.html_url.isin(S.html_url)
D_all["approved_pair"] = D_all.key.map(status)
for label, mask in [("pairs with an approval from someone else", D_all.approved_pair == True), ("pairs with none", D_all.approved_pair == False)]:
    keep = (D_all.treated == 0) | (mask & D_all.in_sample)
    D = D_all[keep].reset_index(drop=True)
    nt = int(D.treated.sum()); nd = D[D.treated == 1].user.nunique()
    b, se, V = fit(D, "merged_7d", ["T_P0", "T_P1", "T_P3"], np.ones(len(D)))
    c = np.array([0, .5, .5]); b2, se2, _ = fit(D, "merged_7d", ["fable_pr"], np.ones(len(D)))
    print(f"   [{label}] treated PRs {nt}, developers {nd}: reversal {ci(c @ b, np.sqrt(c @ V @ c))}, as used {ci(b2[0], se2[0])}")
# approval as an outcome, within developer, Fable 5 vs Opus 4.8 (sampled merged pull requests)
Z = mg[mg.model.isin(["Fable 5", "Opus 4.8"])].copy(); Z["fable"] = (Z.model == "Fable 5").astype(float)
for y, lab in [("approved_other_human", "approved by another non-bot account"), ("changes_requested", "changes requested by another account")]:
    yv = Z[y].astype(float); g = Z.groupby("user")
    yd = yv - g[y].transform("mean"); xd = Z.fable - g.fable.transform("mean"); keep = xd.abs() > 1e-9
    if keep.sum() < 10:
        print(f"   {lab}: too few"); continue
    beta = (xd[keep] * yd[keep]).sum() / (xd[keep] ** 2).sum(); resid = yd[keep] - beta * xd[keep]
    cl = (xd[keep] * resid).groupby(Z.user[keep]).sum(); se = np.sqrt((cl ** 2).sum()) / (xd[keep] ** 2).sum()
    print(f"   {lab}, Fable 5 minus Opus 4.8 within developer: {beta*100:+.1f} points [{(beta-1.96*se)*100:+.1f}, {(beta+1.96*se)*100:+.1f}] ({Z.user[keep].nunique()} developers, {int(keep.sum())} merged pull requests)")
