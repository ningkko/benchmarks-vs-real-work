"""Robustness checks for the suspension analysis.

usage: python3 -I robust.py BASE
1. Event study restricted to pull requests outside the author's own repository.
2. Baselines (room to move) during the suspension.
3. Within-developer size proxies, Fable- vs Opus 4.8-tagged pull requests.
4. Pooled population comparison on matched days (what a merge-rate leaderboard would show).
5. Intraclass correlation in the suspension sample.
"""
import json, sys, warnings, os
import numpy as np
warnings.filterwarnings("ignore")
BASE = sys.argv[1]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import T, prep, ci, load_suspension, fit
D = load_suspension(BASE)
rng = np.random.default_rng(11)
D_all = D.copy()

print("== 1. Outside the author's own repository")
D = D_all[~D_all.own_repo].reset_index(drop=True)
print("rows", len(D), "treated PRs", int(D.treated.sum()), "treated devs", D[D.treated == 1].user.nunique())
for y in ["merged_7d", "closed_unmerged"]:
    b, se, V = fit(D, y, ["T_P0", "T_P1", "T_P3"], np.ones(len(D)))
    c = np.array([0, .5, .5])
    b2, se2, _ = fit(D, y, ["fable_pr"], np.ones(len(D)))
    print(f"  {y:16s} reversal {ci(c @ b, np.sqrt(c @ V @ c))} | as used {ci(b2[0], se2[0])}")
D = D_all

print("\n== 2. Room to move during the suspension (treated, P2)")
t2 = D[(D.treated == 1) & (D.phase == "P2")]
print("  merged within 7 days: all %.1f%%, own repo %.1f%%, other repos %.1f%%" % (
    t2.merged_7d.mean() * 100, t2[t2.own_repo].merged_7d.mean() * 100, t2[~t2.own_repo].merged_7d.mean() * 100))
print("  share of treated PRs in own repo by phase:", D[D.treated == 1].groupby("phase").own_repo.mean().mul(100).round(1).to_dict())

print("\n== 3. Size proxies, Fable- minus Opus 4.8-tagged, within treated developers")
tr = D[(D.treated == 1) & D.model.isin(["Fable 5", "Opus 4.8"])].copy()
tr["blen"] = tr.body_len_full.astype(float)
tr["files"] = tr.body_files_mentioned.astype(float)
tr["checks"] = tr.body_checkboxes_full.astype(float)
tr["is_feat"] = tr.title.fillna("").str.match(r"^\s*feat", case=False).astype(float)
tr["is_fix"] = tr.title.fillna("").str.match(r"^\s*fix", case=False).astype(float)
tr["comments"] = tr.comments.astype(float)
for col in ["blen", "files", "checks", "is_feat", "is_fix", "comments"]:
    g = tr.groupby(["user", "model"])[col].mean().unstack("model")
    d = (g["Fable 5"] - g["Opus 4.8"]).dropna().values
    bs = [rng.choice(d, len(d)).mean() for _ in range(2000)]
    base = g["Opus 4.8"].dropna().mean()
    print(f"  {col:9s} n={len(d)} Opus 4.8 mean {base:8.2f}  diff {d.mean():+8.2f} [{np.percentile(bs,2.5):+.2f}, {np.percentile(bs,97.5):+.2f}]")

print("\n== 4. Pooled comparison of every tagged PR on matched days (leaderboard view)")
def load(fs):
    rows = []
    for f in fs:
        p = f"{BASE}/data_fable/{f}"
        if os.path.exists(p):
            rows += [json.loads(l) for l in open(p)]
    return prep(rows) if rows else None
F = load(["fable_jun.jsonl", "fable_jul.jsonl"]); O = load(["opus48_pop.jsonl", "opus48_p0.jsonl", "opus48_jul_full.jsonl"])
for label, a, b in [("Jun 10-12 (P1)", "2026-06-10", "2026-06-13"), ("Jul 7-9 (P3)", "2026-07-07", "2026-07-10")]:
    f = F[(F.model == "Fable 5") & (F.created_at >= T(a)) & (F.created_at < T(b))] if F is not None else None
    o = O[(O.model == "Opus 4.8") & (O.created_at >= T(a)) & (O.created_at < T(b))]
    if f is None or len(f) == 0:
        print("  ", label, "no Fable data yet"); continue
    print(f"  {label}: Fable n={len(f)} devs={f.user.nunique()} merged7 {f.merged_7d.mean()*100:.1f}% closed {f.closed_unmerged.mean()*100:.1f}% own {f.own_repo.mean()*100:.1f}% | "
          f"Opus 4.8 n={len(o)} devs={o.user.nunique()} merged7 {o.merged_7d.mean()*100:.1f}% closed {o.closed_unmerged.mean()*100:.1f}% own {o.own_repo.mean()*100:.1f}%")
    # developer-weighted pooled
    fw = f.groupby("user").merged_7d.mean().mean(); ow = o.groupby("user").merged_7d.mean().mean()
    print(f"      developer-weighted merged7: Fable {fw*100:.1f}% vs Opus 4.8 {ow*100:.1f}%")

print("\n== 5. Intraclass correlation of merging across developers (suspension sample)")
g = D.groupby("user").merged_7d
k = g.size(); E = D[D.user.isin(k[k >= 2].index)].copy(); E["m"] = E.merged_7d.astype(float)
g = E.groupby("user").m; n_i = g.size().values; means = g.mean().values; N = n_i.sum(); A = len(n_i); grand = E.m.mean()
msb = (n_i * (means - grand) ** 2).sum() / (A - 1); msw = ((E.m - E.user.map(g.mean())) ** 2).sum() / (N - A)
n0 = (N - (n_i ** 2).sum() / N) / (A - 1)
print(f"  ICC {(msb - msw) / (msb + (n0 - 1) * msw):.2f} ({A} developers, {N} PRs)")
