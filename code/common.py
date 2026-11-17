"""Shared code for the analysis scripts.

Every analysis script imports from here instead of repeating these pieces:
reading raw pull-request records, the Fable 5 suspension sample, the upgrade-event samples,
the regression helpers, and the revert matching.

All functions take the data folder (BASE) explicitly. BASE holds the raw GitHub API responses
in the layout data/, data_fable/, events/, calendar/, subst/, merger/, revert/, review/, volume/.
"""
import json, os, re
import numpy as np, pandas as pd
from scipy import stats

# ---------------------------------------------------------------- reading pull requests

# A Claude Code pull request names its model in a "Co-Authored-By: Claude <model> <version>" line.
TAG = re.compile(r"Co-Authored-By:\s*Claude\s+(Opus|Sonnet|Haiku|Fable|Mythos)\s*(\d(?:\.\d)?)?", re.I)
# Any Claude co-author line, with or without a model name.
CLAUDE_LINE = re.compile(r"Co-Authored-By:\s*Claude", re.I)


def T(s):
    """A UTC timestamp."""
    return pd.Timestamp(s, tz="UTC")


def read_jsonl(path):
    return [json.loads(l) for l in open(path)]


def prep(rows):
    """Turn raw pull-request records into one row per pull request with the model and the outcomes.

    Keeps user accounts only (no bots). model is the single model named in the description,
    "MIXED" when several are named and "untagged" when none is.
    """
    df = pd.DataFrame([r for r in rows if not r.get("empty")]).drop_duplicates("html_url")
    df = df[df.user_type == "User"].copy()
    tags = df.body.fillna("").apply(
        lambda b: sorted({(m.group(1).title() + (" " + m.group(2) if m.group(2) else "")) for m in TAG.finditer(b)}))
    df["model"] = tags.apply(lambda t: t[0] if len(t) == 1 else ("MIXED" if t else "untagged"))
    for c in ["created_at", "closed_at", "merged_at"]:
        df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")
    hrs = (df.merged_at - df.created_at).dt.total_seconds() / 3600
    df["merged_7d"] = df.merged_at.notna() & (hrs <= 168)
    df["closed_unmerged"] = (df.state == "closed") & df.merged_at.isna()
    df["own_repo"] = df.repo.str.split("/").str[0].str.lower() == df.user.str.lower()
    df["merged_1h"] = df.merged_at.notna() & (hrs <= 1)
    return df


def ci(b, s):
    """Estimate and 95% interval in percentage points."""
    return f"{b*100:+.1f} [{(b-1.96*s)*100:+.1f}, {(b+1.96*s)*100:+.1f}]"


# ---------------------------------------------------------------- the Fable 5 suspension

# Phases by the time the pull request was opened (UTC). P0 before the release, P1 Fable 5 available,
# P2 suspended (the reference), P3 restored. The days right after each switch are left out.
PHASES = [("P0", T("2026-05-26"), T("2026-06-09 18:00")), ("P1", T("2026-06-09 18:00"), T("2026-06-12 21:00")),
          ("P2", T("2026-06-15"), T("2026-07-01")), ("P3", T("2026-07-02"), T("2026-07-16"))]


def phase(df):
    """Add the phase column and drop pull requests outside the four phases."""
    df["phase"] = None
    for nm, a, b in PHASES:
        df.loc[(df.created_at >= a) & (df.created_at < b), "phase"] = nm
    return df[df.phase.notna()]


def load_treated_history(base):
    """Claude-tagged pull requests of the developers in the treated search, with phases."""
    S = prep(read_jsonl(f"{base}/data_fable/users_treated.jsonl"))
    # the phrase search also returns loose matches; keep only pull requests with a real Claude co-author line
    S = S[S.body.fillna("").str.contains(CLAUDE_LINE)].copy()
    return phase(S)


def load_suspension(base):
    """The suspension sample: treated developers (named Fable 5 in P1) and comparison developers
    (named Opus 4.8 and never Fable 5). Adds the day, the treated-by-phase indicators and fable_pr."""
    S = load_treated_history(base)
    treated = set(S[(S.phase == "P1") & (S.model == "Fable 5")].user)
    S = S[S.user.isin(treated)].copy(); S["treated"] = 1
    C = prep(read_jsonl(f"{base}/data_fable/opus48_p0.jsonl") + read_jsonl(f"{base}/data_fable/opus48_pop.jsonl"))
    fab = {r["user"] for r in read_jsonl(f"{base}/data_fable/fable_jun.jsonl")}
    if os.path.exists(f"{base}/data_fable/weekly_fable.jsonl"):   # also drop comparison developers seen using Fable 5 later
        wf = prep(read_jsonl(f"{base}/data_fable/weekly_fable.jsonl"))
        fab |= set(wf[wf.model == "Fable 5"].user)
    C = C[(C.model == "Opus 4.8") & ~C.user.isin(treated | fab | set(S.user))]
    C = phase(C).copy(); C["treated"] = 0
    D = pd.concat([S, C], ignore_index=True).drop_duplicates("html_url")
    D["day"] = D.created_at.dt.strftime("%Y-%m-%d")
    for p in ["P0", "P1", "P3"]:
        D[f"T_{p}"] = ((D.treated == 1) & (D.phase == p)).astype(float)
    D["fable_pr"] = ((D.treated == 1) & (D.model == "Fable 5")).astype(float)
    print("rows", len(D), "| treated PRs", int(D.treated.sum()), "devs", D[D.treated == 1].user.nunique(),
          "| control PRs", int((D.treated == 0).sum()), "devs", D[D.treated == 0].user.nunique(), "| days", D.day.nunique())
    return D


def demean(v, w, keys, iters=200, tol=1e-10):
    """Weighted within transform for several sets of fixed effects (alternating projections)."""
    v = v.astype(float).copy()
    for _ in range(iters):
        old = v.copy()
        for k in keys:
            m = (v * w).groupby(k).transform("sum") / w.groupby(k).transform("sum")
            v = v - m
        if np.max(np.abs(v - old)) < tol:
            break
    return v


def fit(D, y, Xcols, weight, fe=("user", "day")):
    """Weighted linear probability model with fixed effects (developer and calendar day by default),
    standard errors clustered by developer (CR1). Returns coefficients, standard errors and covariance."""
    w = pd.Series(weight, index=D.index)
    keys = [D[k] for k in fe]
    yt = demean(D[y].astype(float), w, keys)
    Xt = np.column_stack([demean(D[c], w, keys) for c in Xcols])
    W = w.values
    XtW = Xt.T * W
    bread = np.linalg.pinv(XtW @ Xt)
    beta = bread @ (XtW @ yt.values)
    resid = yt.values - Xt @ beta
    g = D.user.values
    meat = np.zeros((len(Xcols), len(Xcols)))
    for u, idx in pd.Series(np.arange(len(g))).groupby(g):
        s = (Xt[idx.values] * (W[idx.values] * resid[idx.values])[:, None]).sum(0)
        meat += np.outer(s, s)
    G = D.user.nunique(); N = len(D); K = len(Xcols)
    V = bread @ meat @ bread * (G / (G - 1)) * ((N - 1) / (N - K))
    se = np.sqrt(np.diag(V))
    return beta, se, V


# ---------------------------------------------------------------- the February upgrades (exploratory)

def load_february(base):
    """All Opus and Sonnet 4.5 and 4.6 pull requests, December 2025 to April 10, 2026, with the exposure columns:
    new_opus / new_sonnet (the pull request names the 4.6 version of its family) and opus_family."""
    import glob
    rows = []
    for f in glob.glob(f"{base}/data/cc_*.jsonl"):
        rows += read_jsonl(f)
    D = prep(rows)
    D = D[D.model.isin(["Opus 4.5", "Opus 4.6", "Sonnet 4.5", "Sonnet 4.6"]) & (D.created_at >= T("2025-12-01"))
          & (D.created_at < T("2026-04-11"))].reset_index(drop=True)
    D["day"] = D.created_at.dt.strftime("%Y-%m-%d")
    D["new_opus"] = (D.model == "Opus 4.6").astype(float)
    D["new_sonnet"] = (D.model == "Sonnet 4.6").astype(float)
    D["opus_family"] = D.model.str.startswith("Opus").astype(float)
    print("rows", len(D), "developers", D.user.nunique(), "days", D.day.nunique())
    return D


# ---------------------------------------------------------------- the upgrade events

# (event, release day, older model, newer model, file before, file after)
EVENTS = [
    (1, "2026-02-05", "Opus 4.5", "Opus 4.6", "cc_opus45", "cc_opus46"),
    (2, "2026-02-17", "Sonnet 4.5", "Sonnet 4.6", "cc_sonnet45", "cc_sonnet46"),
    (3, "2026-04-16", "Opus 4.6", "Opus 4.7", "e3_old", "e3_new"),
    (4, "2026-05-28", "Opus 4.7", "Opus 4.8", "e4_old", "e4_new"),
    (5, "2026-06-30", "Sonnet 4.6", "Sonnet 5", "e5_old", "e5_new"),
    (6, "2026-07-24", "Opus 4.8", "Opus 5", "e6_old", "e6_new"),
    (7, "2026-09-22", "Opus 5", "Opus 5.5", "e7_old", "e7_new"),
    (8, "2026-09-28", "Sonnet 5", "Sonnet 5.5", "e8_old", "e8_new"),
]
# SWE-Bench Pro (%) as reported by Anthropic in a launch post or system card. Opus 4.6 (53.4) is from the
# Opus 4.7 launch table and Opus 5.5 (89.9) from its system card; no Anthropic-reported score was found for
# Opus 4.5 or Sonnet 4.5, so events 1 and 2 have no prediction.
SWEPRO = {"Opus 4.6": 53.4, "Opus 4.7": 64.3, "Opus 4.8": 69.2, "Sonnet 4.6": 58.1, "Sonnet 5": 63.2,
          "Opus 5": 79.2, "Opus 5.5": 89.9, "Sonnet 5.5": 81.3}
INCLUDED = [1, 3, 4, 6, 7, 8]       # events with >= 10 switchers without write access in both windows
STRICT = [3, 4, 6, 7, 8]            # the main pool: events collected after the analysis plan
WRITE = ["OWNER", "MEMBER", "COLLABORATOR"]
DAY = pd.Timedelta(days=1)


def load_event(base, name):
    for p in [f"{base}/events/{name}.jsonl", f"{base}/data/{name}.jsonl"]:
        if os.path.exists(p):
            return prep(read_jsonl(p))
    return None


def fe_ols(df, y, X, unit="user"):
    """OLS with one-way unit fixed effects (within transform), SE clustered by unit (CR1)."""
    d = df.copy()
    cols = [y] + X
    dm = d[cols].astype(float) - d.groupby(unit)[cols].transform("mean").astype(float)
    Xm = dm[X].values; ym = dm[y].values
    XtX = Xm.T @ Xm
    bread = np.linalg.pinv(XtX)
    b = bread @ (Xm.T @ ym)
    e = ym - Xm @ b
    meat = np.zeros((len(X), len(X)))
    for _, idx in pd.Series(np.arange(len(d))).groupby(d[unit].values):
        s = (Xm[idx.values] * e[idx.values][:, None]).sum(0)
        meat += np.outer(s, s)
    G = d[unit].nunique(); N = len(d); K = len(X) + G
    V = bread @ meat @ bread * (G / (G - 1)) * ((N - 1) / max(N - K, 1))
    return b, V


def pool(est, se):
    """Inverse-variance (fixed-effect) pooling, with DerSimonian-Laird random effects as a sensitivity check."""
    est, se = np.asarray(est, float), np.asarray(se, float)
    w = 1 / se ** 2
    fe = (w * est).sum() / w.sum(); fe_se = np.sqrt(1 / w.sum())
    Q = (w * (est - fe) ** 2).sum(); k = len(est)
    tau2 = max(0.0, (Q - (k - 1)) / (w.sum() - (w ** 2).sum() / w.sum())) if k > 1 else 0.0
    wr = 1 / (se ** 2 + tau2)
    re_ = (wr * est).sum() / wr.sum(); re_se = np.sqrt(1 / wr.sum())
    p_fe = 2 * (1 - stats.norm.cdf(abs(fe / fe_se))); p_re = 2 * (1 - stats.norm.cdf(abs(re_ / re_se)))
    i2 = max(0.0, (Q - (k - 1)) / Q) if Q > 0 else 0.0
    return dict(fe=fe, fe_se=fe_se, p_fe=p_fe, re=re_, re_se=re_se, p_re=p_re, tau2=tau2, Q=Q, I2=i2, k=k)


def switchers(base, d, old, new, f_old, f_new):
    """Pull requests naming the older model in the 7 days before the release (a) and the newer model in
    days 1 to 7 after it (b), and the developers in both (the switchers)."""
    A, B = load_event(base, f_old), load_event(base, f_new)
    d0 = T(d)
    a = A[(A.model == old) & (A.created_at >= d0 - 7 * DAY) & (A.created_at < d0)].copy()
    b = B[(B.model == new) & (B.created_at >= d0 + DAY) & (B.created_at < d0 + 8 * DAY)].copy()
    return a, b, set(a.user) & set(b.user)


# ---------------------------------------------------------------- calendar control for the upgrades

# For each event, the comparison model whose users ("stayers") identify the calendar-day effects.
COMP = {1: ("Sonnet 4.5", ["data/cc_sonnet45.jsonl"]),
        3: ("Sonnet 4.6", ["calendar/c3_sonnet46.jsonl"]),
        4: ("Sonnet 4.6", ["calendar/c4_sonnet46.jsonl"]),
        5: ("Opus 4.8", ["data_fable/opus48_pop.jsonl"]),
        6: ("Sonnet 5", ["calendar/c6_sonnet5.jsonl"]),
        7: ("Sonnet 5", ["calendar/c7_sonnet5.jsonl"]),
        8: ("Opus 5.5", ["calendar/c8_opus55.jsonl"])}


def demean_plain(v, keys, iters=500, tol=1e-10):
    """Unweighted within transform for several sets of fixed effects."""
    v = v.astype(float).copy()
    for _ in range(iters):
        old = v.copy()
        for k in keys:
            v = v - v.groupby(k).transform("mean")
        if np.max(np.abs(v - old)) < tol:
            break
    return v


def twfe(D, y, X):
    """OLS with developer and calendar-day fixed effects, SE clustered by developer (CR1)."""
    keys = [D.user, D.day]
    yt = demean_plain(D[y], keys).values
    Xt = np.column_stack([demean_plain(D[c], keys).values for c in X])
    bread = np.linalg.pinv(Xt.T @ Xt)
    b = bread @ (Xt.T @ yt)
    e = yt - Xt @ b
    meat = np.zeros((len(X), len(X)))
    for _, idx in pd.Series(np.arange(len(D))).groupby(D.user.values):
        s = (Xt[idx.values] * e[idx.values][:, None]).sum(0)
        meat += np.outer(s, s)
    G = D.user.nunique(); N = len(D); K = len(X)
    V = bread @ meat @ bread * (G / (G - 1)) * ((N - 1) / (N - K))
    return b, V


def num(df, col):
    return pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(np.nan, index=df.index)


def build(base, k, d, old, new, f_old, f_new):
    """Switchers' and stayers' pull requests around event k, with the after-release indicators."""
    A, B = load_event(base, f_old), load_event(base, f_new)
    d0 = T(d)
    a = A[(A.model == old) & (A.created_at >= d0 - 7 * DAY) & (A.created_at < d0)]
    b = B[(B.model == new) & (B.created_at >= d0 + DAY) & (B.created_at < d0 + 8 * DAY)]
    sw = set(a.user) & set(b.user)
    S_ = pd.concat([a[a.user.isin(sw)], b[b.user.isin(sw)]], ignore_index=True).drop_duplicates("html_url")
    S_["treat"] = 1.0
    cm, files = COMP[k]
    C = prep(sum([read_jsonl(f"{base}/{f}") for f in files], []))
    C = C[C.model == cm].drop_duplicates("html_url")
    cb = C[(C.created_at >= d0 - 7 * DAY) & (C.created_at < d0)]
    ca = C[(C.created_at >= d0 + DAY) & (C.created_at < d0 + 8 * DAY)]
    stay = (set(cb.user) & set(ca.user)) - (set(a.user) | set(b.user))
    Cn = pd.concat([cb[cb.user.isin(stay)], ca[ca.user.isin(stay)]], ignore_index=True)
    Cn["treat"] = 0.0
    D = pd.concat([S_, Cn], ignore_index=True).drop_duplicates("html_url").reset_index(drop=True)
    D["after"] = (D.created_at >= d0 + DAY).astype(float)
    D["day"] = D.created_at.dt.strftime("%Y-%m-%d")
    D["sw_after"] = D.treat * D.after
    D["merged_7d"] = D.merged_7d.astype(float)
    D["merged_1h"] = D.merged_1h.astype(float)
    D["closed_unmerged"] = D.closed_unmerged.astype(float)
    D["comments"] = num(D, "comments")
    D["blen"] = num(D, "body_len_full")
    D["files"] = num(D, "body_files_mentioned")
    D["is_feat"] = D.title.fillna("").str.match(r"^\s*feat", case=False).astype(float)
    return D


def wdemean(v, w, keys, iters=500, tol=1e-10):
    v = v.astype(float).copy()
    for _ in range(iters):
        old = v.copy()
        for k in keys:
            v = v - (v * w).groupby(k).transform("sum") / w.groupby(k).transform("sum")
        if np.max(np.abs(v - old)) < tol:
            break
    return v


def wtwfe(D, y, X, w):
    """Weighted version of twfe()."""
    w = pd.Series(w, index=D.index)
    keys = [D.user, D.day]
    yt = wdemean(D[y], w, keys).values
    Xt = np.column_stack([wdemean(D[c], w, keys).values for c in X])
    W = w.values
    XtW = Xt.T * W
    bread = np.linalg.pinv(XtW @ Xt)
    b = bread @ (XtW @ yt)
    e = yt - Xt @ b
    meat = np.zeros((len(X), len(X)))
    for _, idx in pd.Series(np.arange(len(D))).groupby(D.user.values):
        s = (Xt[idx.values] * (W[idx.values] * e[idx.values])[:, None]).sum(0)
        meat += np.outer(s, s)
    G = D.user.nunique(); N = len(D); K = len(X)
    V = bread @ meat @ bread * (G / (G - 1)) * ((N - 1) / (N - K))
    return b, V


def ratio_gain(old, new):
    """Share of the older model's SWE-Bench Pro failures that the newer model solves."""
    return 1 - (100 - SWEPRO[new]) / (100 - SWEPRO[old]) if (old in SWEPRO and new in SWEPRO) else None


def fmt_pool(p, scale=1.0, dec=1):
    lo, hi = p["fe"] - 1.96 * p["fe_se"], p["fe"] + 1.96 * p["fe_se"]
    return (f"{p['fe']*scale:+.{dec}f} [{lo*scale:+.{dec}f}, {hi*scale:+.{dec}f}] p={p['p_fe']:.4f} "
            f"(RE {p['re']*scale:+.{dec}f}, I2={p['I2']:.2f}, k={p['k']})")


# ---------------------------------------------------------------- reverts

def load_reverts(path):
    R = pd.DataFrame([r for r in read_jsonl(path) if not r.get("_done")]).drop_duplicates("html_url")
    R["created_at"] = pd.to_datetime(R.created_at, utc=True)
    R["text"] = (R.title.fillna("") + "\n" + R.body.fillna("")).str.lower()
    return R


def mark(P, R):
    """P: pull requests with repo, number, title, merged_at (UTC). True when a revert-titled pull request in the
    same repository quotes the title or names the number within 30 days after the merge."""
    byrepo = {k: g for k, g in R.groupby("repo")}
    out = []
    for r in P.itertuples():
        hit = False
        g = byrepo.get(r.repo)
        if g is not None and pd.notna(r.merged_at):
            w = g[(g.created_at >= r.merged_at) & (g.created_at <= r.merged_at + pd.Timedelta(days=30))]
            t = (r.title or "").strip().lower()
            pat = re.compile(rf"(?<![\d#\w])#{int(r.number)}(?!\d)")
            for row in w.itertuples():
                if (t and len(t) > 8 and t in row.text) or pat.search(row.text):
                    hit = True; break
        out.append(hit)
    return pd.Series(out, index=P.index)
