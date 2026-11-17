"""All headline numbers for the paper, in one place.

usage: python3 -I final_results.py BASE_DIR
Reads BASE/data/cc_*.jsonl (February upgrades) and BASE/data_fable/users_*.jsonl (suspension).
Every contrast is oriented as (higher-benchmark model) minus (lower-benchmark model), in percentage points,
computed on developer-level means and bootstrapped over developers (2,000 resamples, seed 11).
"""
import os, json, sys, glob, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
BASE = sys.argv[1]
rng = np.random.default_rng(11)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CLAUDE_LINE, T, prep, PHASES

def boot(d):
    d = np.asarray(d, dtype=float)
    bs = [rng.choice(d, len(d)).mean() for _ in range(2000)]
    return d.mean() * 100, np.percentile(bs, 2.5) * 100, np.percentile(bs, 97.5) * 100, len(d)


def paired(a, b, col):
    """developer-level mean of col in frame b minus frame a, developers present in both"""
    ma = a.groupby("user")[col].mean(); mb = b.groupby("user")[col].mean()
    d = (mb - ma).dropna()
    return boot(d.values)


def fmt(r):
    m, lo, hi, n = r
    return f"{m:+.1f} [{lo:+.1f}, {hi:+.1f}] n={n}"


out = {}
# ---------------- February upgrades
feb = []
for f in glob.glob(f"{BASE}/data/cc_*.jsonl"):
    feb += [json.loads(l) for l in open(f)]
F = prep(feb)
F = F[F.model.isin(["Opus 4.5", "Opus 4.6", "Sonnet 4.5", "Sonnet 4.6"]) & (F.created_at >= T("2025-12-01")) & (F.created_at < T("2026-04-11"))]
print("February data PRs by model:", F.model.value_counts().to_dict(), "developers:", F.groupby("model").user.nunique().to_dict())
for old, new, cut, label in [("Opus 4.5", "Opus 4.6", "2026-02-05", "opus"), ("Sonnet 4.5", "Sonnet 4.6", "2026-02-17", "sonnet")]:
    a = F[(F.model == old) & (F.created_at < T(cut))]
    b = F[(F.model == new) & (F.created_at >= T(cut))]
    users = set(a.user) & set(b.user)
    a, b = a[a.user.isin(users)], b[b.user.isin(users)]
    print(f"\n{old} -> {new}: {len(users)} developers, {len(a)} PRs before, {len(b)} after")
    if len(users) >= 10:
        for col in ["merged_7d", "closed_unmerged", "merged_1h", "own_repo"]:
            r = paired(a, b, col); out[(label, col)] = r
            print(f"  {col:16s} {fmt(r)}")
        print("  comments         diff=%+.2f" % (b.groupby('user').comments.mean() - a.groupby('user').comments.mean()).dropna().mean())
        # pooled comparison (all developers) for contrast
        pa = F[(F.model == old)]; pb = F[(F.model == new) & (F.created_at >= T(cut))]
        print("  pooled, all developers: merged_7d %.1f vs %.1f; merged_1h %.1f vs %.1f; outside own repo %.1f vs %.1f" % (
            pa.merged_7d.mean()*100, pb.merged_7d.mean()*100, pa.merged_1h.mean()*100, pb.merged_1h.mean()*100,
            (~pa.own_repo).mean()*100, (~pb.own_repo).mean()*100))

# ICC of merging across developers (one-way ANOVA estimator), February data
g = F.groupby("user").merged_7d
k = g.size(); D = F[F.user.isin(k[k >= 2].index)].copy(); D["m"] = D.merged_7d.astype(float)
g = D.groupby("user").m; n_i = g.size().values; means = g.mean().values; N = n_i.sum(); A = len(n_i); grand = D.m.mean()
msb = (n_i * (means - grand) ** 2).sum() / (A - 1); msw = ((D.m - D.user.map(g.mean())) ** 2).sum() / (N - A)
n0 = (N - (n_i ** 2).sum() / N) / (A - 1)
icc = (msb - msw) / (msb + (n0 - 1) * msw)
print(f"\nICC(1) of merged within 7 days across developers: {icc:.2f} ({A} developers with >=2 PRs, {N} PRs)")
print("own repo vs not, merged_7d: %.1f vs %.1f; merged within 1h: %.1f vs %.1f" % (
    D[D.own_repo].merged_7d.mean()*100, D[~D.own_repo].merged_7d.mean()*100,
    D[D.own_repo].merged_1h.mean()*100, D[~D.own_repo].merged_1h.mean()*100))

# ---------------- Suspension
rows = []
for f in glob.glob(f"{BASE}/data_fable/users_*.jsonl"):
    rows += [json.loads(l) for l in open(f)]
S = prep(rows)
# the phrase search also returns loose matches; keep only pull requests with a real Claude co-author line
S = S[S.body.fillna('').str.contains(CLAUDE_LINE)].copy()
S["phase"] = None
for nm, a, b in PHASES:
    S.loc[(S.created_at >= a) & (S.created_at < b), "phase"] = nm
treated = set(S[(S.phase == "P1") & (S.model == "Fable 5")].user)
ever_fable = set(S[S.model == "Fable 5"].user)
S["group"] = np.where(S.user.isin(treated), "treated", np.where(~S.user.isin(ever_fable), "control", "other"))
S = S[S.phase.notna() & S.group.isin(["treated", "control"])]
print("\nSuspension: treated developers", len(treated), "| control developers with PRs", S[S.group == "control"].user.nunique())
print(S.groupby(["group", "phase"]).agg(prs=("html_url", "size"), devs=("user", "nunique")).unstack("phase"))
for grp in ["treated", "control"]:
    G = S[S.group == grp]
    for a_, b_, sign, label in [("P0", "P1", +1, "intro"), ("P1", "P2", -1, "removal"), ("P2", "P3", +1, "return")]:
        for col in ["merged_7d", "closed_unmerged"]:
            r = paired(G[G.phase == a_], G[G.phase == b_], col)
            r = (sign * r[0], min(sign * r[1], sign * r[2]), max(sign * r[1], sign * r[2]), r[3])
            out[(grp, label, col)] = r
            print(f"  {grp:8s} {label:8s} ({a_}->{b_}, oriented Fable minus Opus) {col:16s} {fmt(r)}")
# Calendar control: Opus 4.8 users on sampled days in each phase who never opened a Fable-tagged PR in our data
crow = []
for f in [f"{BASE}/data_fable/opus48_p0.jsonl", f"{BASE}/data_fable/opus48_pop.jsonl"]:
    try:
        crow += [json.loads(l) for l in open(f)]
    except FileNotFoundError:
        pass
fable_users = set(treated) | ever_fable
try:
    fable_users |= {json.loads(l)["user"] for l in open(f"{BASE}/data_fable/fable_jun.jsonl")}
except FileNotFoundError:
    pass
C = prep(crow)
C = C[(C.model == "Opus 4.8") & ~C.user.isin(fable_users)].copy()
C["phase"] = None
for nm, a, b in PHASES:
    C.loc[(C.created_at >= a) & (C.created_at < b), "phase"] = nm
C = C[C.phase.notna()]
print("\nCalendar control (Opus 4.8, never Fable): PRs and developers by phase")
print(C.groupby("phase").agg(prs=("html_url", "size"), devs=("user", "nunique")).T)
for a_, b_, sign, label in [("P0", "P1", +1, "intro"), ("P1", "P2", -1, "removal"), ("P2", "P3", +1, "return")]:
    for col in ["merged_7d", "closed_unmerged"]:
        dc = (C[C.phase == b_].groupby("user")[col].mean() - C[C.phase == a_].groupby("user")[col].mean()).dropna().values
        G = S[S.group == "treated"]
        dt = (G[G.phase == b_].groupby("user")[col].mean() - G[G.phase == a_].groupby("user")[col].mean()).dropna().values
        if len(dc) >= 10:
            rc = boot(dc); rc = (sign * rc[0], min(sign * rc[1], sign * rc[2]), max(sign * rc[1], sign * rc[2]), rc[3])
            bs = [sign * (rng.choice(dt, len(dt)).mean() - rng.choice(dc, len(dc)).mean()) for _ in range(2000)]
            did = (sign * (dt.mean() - dc.mean()) * 100, np.percentile(bs, 2.5) * 100, np.percentile(bs, 97.5) * 100, len(dt))
            out[("calctrl", label, col)] = rc; out[("did", label, col)] = did
            print(f"  control {label:8s} {col:16s} {fmt(rc)}   | DiD treated minus control {fmt(did)} (control n={len(dc)})")
# per-protocol within treated: Fable-tagged minus Opus-4.8-tagged
TR = S[(S.group == "treated") & S.model.isin(["Fable 5", "Opus 4.8"])]
for col in ["merged_7d", "closed_unmerged"]:
    gg = TR.groupby(["user", "model"])[col].mean().unstack("model")
    r = boot((gg["Fable 5"] - gg["Opus 4.8"]).dropna().values); out[("pp", col)] = r
    print(f"  per-protocol Fable minus Opus 4.8 {col:16s} {fmt(r)}")
gg = TR.groupby(["user", "model"]).comments.mean().unstack("model")
print("  per-protocol comments diff %+.2f" % (gg["Fable 5"] - gg["Opus 4.8"]).dropna().mean())
P3 = S[(S.phase == "P3") & S.model.isin(["Fable 5", "Opus 4.8"])]
print("  pooled P3 (all developers in data): merged_7d", P3.groupby("model").merged_7d.mean().mul(100).round(1).to_dict(),
      "closed", P3.groupby("model").closed_unmerged.mean().mul(100).round(1).to_dict(), P3.groupby("model").size().to_dict())
# share of treated PRs by model in each phase (exposure check)
t = S[S.group == "treated"]
print(pd.crosstab(t.phase, t.model, normalize="index").mul(100).round(1).loc[:, lambda x: x.max() > 1])
json.dump({"|".join(k): v for k, v in out.items()}, open(f"{BASE}/final_numbers.json", "w"), indent=1)
