"""Checks prompted by an outside review (listed before running them).

usage: python3 -I review_checks.py BASE
"""
import json, re, sys, os, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
BASE = sys.argv[1]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import T, EVENTS, INCLUDED, DAY, load_event, fe_ols, pool, twfe, build
FIVE = [3, 4, 6, 7, 8]
INSIDE = ["OWNER", "MEMBER", "COLLABORATOR"]


def ci(p, d=2):
    lo, hi = p["fe"] - 1.96 * p["fe_se"], p["fe"] + 1.96 * p["fe_se"]
    return f"{p['fe']:+.{d}f} [{lo:+.{d}f}, {hi:+.{d}f}] p={p['p_fe']:.4f} | RE {p['re']:+.{d}f} [{p['re']-1.96*p['re_se']:+.{d}f}, {p['re']+1.96*p['re_se']:+.{d}f}] I2={p['I2']:.2f} k={p['k']}"


E = pd.read_csv(f"{BASE}/events/event_estimates.csv")
print("== events collected after the plan (3, 4, 6, 7, 8) vs the pre-specified pool (1, 3, 4, 6, 7, 8)")
for lab, ev in [("five", FIVE), ("six", INCLUDED)]:
    h = E[E.event.isin(ev)]
    print(f"[{lab}] interaction {ci(pool(h.inter_est, h.inter_se))}")
    print(f"[{lab}] change, all pull requests {ci(pool(h.all_est, h.all_se))}")
    print(f"[{lab}] change, insiders {ci(pool(h.self_est, h.self_se))}")
    print(f"[{lab}] change, outside authors {ci(pool(h.ext_est, h.ext_se))}")
print("switchers in the five events:", int(E[E.event.isin(FIVE)].switchers.sum()),
      "| with outside-author PRs in both windows:", int(E[E.event.isin(FIVE)].ext_switchers.sum()))


def switch_data(k, d, old, new, f_old, f_new):
    A, B = load_event(BASE, f_old), load_event(BASE, f_new)
    d0 = T(d)
    a = A[(A.model == old) & (A.created_at >= d0 - 7 * DAY) & (A.created_at < d0)].copy()
    b = B[(B.model == new) & (B.created_at >= d0 + DAY) & (B.created_at < d0 + 8 * DAY)].copy()
    sw = set(a.user) & set(b.user)
    a["new"] = 0.0; b["new"] = 1.0
    D = pd.concat([a[a.user.isin(sw)], b[b.user.isin(sw)]], ignore_index=True).drop_duplicates("html_url")
    D["y"] = D.merged_7d.astype(float)
    D["event"] = k
    return D


SW = {k: switch_data(k, d, old, new, f_old, f_new) for k, d, old, new, f_old, f_new in EVENTS if k in INCLUDED}

print("\n== insiders limited to OWNER (MEMBER and COLLABORATOR dropped) vs outside authors")
r2 = []
for k, D in SW.items():
    D2 = D[~D.assoc.isin(["MEMBER", "COLLABORATOR"])].copy()
    D2["external"] = (~D2.assoc.isin(["OWNER"])).astype(float)
    D2["new_x_ext"] = D2.new * D2.external
    bb, V = fe_ols(D2, "y", ["new", "external", "new_x_ext"])
    share_mc = D.assoc.isin(["MEMBER", "COLLABORATOR"]).mean() * 100
    r2.append(dict(event=k, prs=len(D2), dropped_member_collab_pct=share_mc, inter=bb[2] * 100, se=np.sqrt(V[2, 2]) * 100,
                   owner=bb[0] * 100, owner_se=np.sqrt(V[0, 0]) * 100))
owner_only = pd.DataFrame(r2)
print(owner_only.round(2).to_string(index=False))
for lab, ev in [("six", INCLUDED), ("five", FIVE)]:
    h = owner_only[owner_only.event.isin(ev)]
    print(f"[{lab}] interaction (outside authors minus owners) {ci(pool(h.inter, h.se))}")
    print(f"[{lab}] change for owners {ci(pool(h.owner, h.owner_se))}")

print("\n== overlapping windows")
g = json.load(open(f"{BASE}/volume/groups.json"))
s7, s8 = set(g["7"]["switchers"]), set(g["8"]["switchers"])
print(f"switchers in event 7: {len(s7)}, event 8: {len(s8)}, in both: {len(s7 & s8)}")
allsw = [u for k in g for u in g[k]["switchers"]]
print(f"switcher-events {len(allsw)}, distinct developers {len(set(allsw))}")
for drop in [7, 8]:
    h = E[E.event.isin([e for e in INCLUDED if e != drop])]
    print(f"without event {drop}: interaction {ci(pool(h.inter_est, h.inter_se))}")
    print(f"without event {drop}: change, all pull requests {ci(pool(h.all_est, h.all_se))}")
    h5 = E[E.event.isin([e for e in FIVE if e != drop]) & E.pred.notna()]
    p = pool(h5.all_est, h5.all_se); pr = (h5.pred / h5.all_se ** 2).sum() / (1 / h5.all_se ** 2).sum()
    print(f"without event {drop}: benchmark comparison {list(h5.event)} change {p['fe']:+.2f} vs prediction {pr:+.2f}, "
          f"difference {p['fe']-pr:+.2f} [{p['fe']-pr-1.96*p['fe_se']:+.2f}, {p['fe']-pr+1.96*p['fe_se']:+.2f}]")


def fe_cl(df, y, X, fe, cl):
    cols = [y] + X
    dm = df[cols].astype(float) - df.groupby(fe)[cols].transform("mean").astype(float)
    Xm = dm[X].values; ym = dm[y].values
    bread = np.linalg.pinv(Xm.T @ Xm); b = bread @ (Xm.T @ ym); e = ym - Xm @ b
    meat = np.zeros((len(X), len(X)))
    for _, idx in pd.Series(np.arange(len(df))).groupby(df[cl].values):
        s = (Xm[idx.values] * e[idx.values][:, None]).sum(0); meat += np.outer(s, s)
    G = df[cl].nunique(); N = len(df); K = len(X) + df[fe].nunique()
    return b, bread @ meat @ bread * (G / (G - 1)) * ((N - 1) / max(N - K, 1))


ST = pd.concat([SW[k] for k in INCLUDED], ignore_index=True)
ST["ue"] = ST.user.astype(str) + "|" + ST.event.astype(str)
ST["external"] = (~ST.assoc.isin(INSIDE)).astype(float)
ST["new_x_ext"] = ST.new * ST.external
for lab, sub in [("six", ST), ("five", ST[ST.event.isin(FIVE)])]:
    b, V = fe_cl(sub, "y", ["new"], "ue", "user")
    b2, V2 = fe_cl(sub, "y", ["new", "external", "new_x_ext"], "ue", "user")
    print(f"stacked [{lab}] change {b[0]*100:+.2f} [{(b[0]-1.96*np.sqrt(V[0,0]))*100:+.2f}, {(b[0]+1.96*np.sqrt(V[0,0]))*100:+.2f}]"
          f" | interaction {b2[2]*100:+.2f} [{(b2[2]-1.96*np.sqrt(V2[2,2]))*100:+.2f}, {(b2[2]+1.96*np.sqrt(V2[2,2]))*100:+.2f}]"
          f" | clusters {sub.user.nunique()}")

print("\n== observed change as a share of the proportional-transfer prediction")
h = E[E.event.isin(FIVE) & E.pred.notna()]
p = pool(h.all_est, h.all_se); pr = (h.pred / h.all_se ** 2).sum() / (1 / h.all_se ** 2).sum()
lo, hi = p["fe"] - 1.96 * p["fe_se"], p["fe"] + 1.96 * p["fe_se"]
print(f"pre-specified: observed {p['fe']:+.2f} [{lo:+.2f}, {hi:+.2f}], prediction {pr:+.2f}; share {p['fe']/pr:+.2f} [{lo/pr:+.2f}, {hi/pr:+.2f}]")
C = pd.read_csv(f"{BASE}/calendar/calendar_estimates.csv")
h = C[C.event.isin(FIVE) & C.pred.notna()]
p = pool(h.did, h.did_se); pr = (h.pred / h.did_se ** 2).sum() / (1 / h.did_se ** 2).sum()
lo, hi = p["fe"] - 1.96 * p["fe_se"], p["fe"] + 1.96 * p["fe_se"]
print(f"calendar-controlled: observed {p['fe']:+.2f} [{lo:+.2f}, {hi:+.2f}], prediction {pr:+.2f}; share {p['fe']/pr:+.2f} [{lo/pr:+.2f}, {hi/pr:+.2f}]")

print("\n== Claude Code PR footer in tagged PR bodies, change around upgrades against stayers")
FOOT = re.compile(r"Generated with \[?Claude Code", re.I)
r5 = []
for k, d, old, new, f_old, f_new in EVENTS:
    if k not in INCLUDED:
        continue
    D = build(BASE, k, d, old, new, f_old, f_new)
    D["footer"] = D.body.fillna("").str.contains(FOOT).astype(float)
    bb, V = twfe(D, "footer", ["sw_after"])
    r5.append(dict(event=k, n_sw=int((D.treat == 1).sum()), sw_all=D[D.treat == 1].footer.mean() * 100,
                   sw_before=D[(D.treat == 1) & (D.after == 0)].footer.mean() * 100,
                   sw_after=D[(D.treat == 1) & (D.after == 1)].footer.mean() * 100,
                   st_before=D[(D.treat == 0) & (D.after == 0)].footer.mean() * 100,
                   st_after=D[(D.treat == 0) & (D.after == 1)].footer.mean() * 100,
                   did=bb[0] * 100, se=np.sqrt(V[0, 0]) * 100))
footer = pd.DataFrame(r5)
print(footer.round(2).to_string(index=False))
print(f"pooled calendar-controlled change in footer share (points) {ci(pool(footer.did, footer.se), 1)}")
print(f"share of switchers' tagged PR bodies with the Claude Code PR footer, all six events: "
      f"{(footer.sw_all * footer.n_sw).sum() / footer.n_sw.sum():.1f}% of {int(footer.n_sw.sum())} PRs")

print("\n== commit-level attribution for a random sample of merged, tagged switcher PRs")
path = f"{BASE}/review/attrib.jsonl"
if os.path.exists(path):
    A6 = pd.DataFrame([json.loads(l) for l in open(path)])
    ok = A6[A6.ok]
    found = ok[ok.n_seen > 0]
    withm = found[found.n_with_model > 0]
    agree = withm.apply(lambda r: r.pr_model in r.models, axis=1)
    only_other = withm.apply(lambda r: r.pr_model not in r.models, axis=1)
    print(f"sampled {len(A6)}, searched {len(ok)}, with commits by the author found {len(found)}, "
          f"with a model-named co-author line {len(withm)} ({len(withm)/max(len(found),1)*100:.1f}% of found)")
    print(f"commit lines name the PR's model: {agree.sum()} ({agree.mean()*100:.1f}%); name only other models: {only_other.sum()} ({only_other.mean()*100:.1f}%)")
    for side in ["before", "after"]:
        w = withm[withm.side == side]
        if len(w):
            ag = w.apply(lambda r: r.pr_model in r.models, axis=1)
            print(f"   {side}: {len(w)} PRs with model lines in commits, agreement {ag.mean()*100:.1f}%")
    print("models named in commits when they differ from the PR:",
          withm[only_other][["pr_model", "models"]].astype(str).value_counts().head(10).to_dict())
else:
    print("attrib.jsonl not found")

print("\n== commit check with a tight window (1 day before opening to 1 day after merging)")
path = f"{BASE}/review/attrib_tight.jsonl"
if os.path.exists(path):
    A7 = pd.DataFrame([json.loads(l) for l in open(path)])
    ok = A7[A7.ok]
    found = ok[ok.n_seen > 0]
    withm = found[found.n_with_model > 0].copy()
    print(f"sampled {len(A7)}, with commits by the author found {len(found)}, with a model-named co-author line {len(withm)}")
    def cat(r):
        m = set(r.models)
        return "same only" if m == {r.pr_model} else ("same plus other" if r.pr_model in m else "other only")
    withm["cat"] = withm.apply(cat, axis=1)
    print(pd.crosstab(withm.side, withm.cat, margins=True).to_string())
    for side in ["before", "after"]:
        w = withm[withm.side == side]
        print(f"   {side}: {len(w)} PRs, commits name the PR's model {(w.cat != 'other only').mean()*100:.1f}%, "
              f"name only other models {(w.cat == 'other only').mean()*100:.1f}%")
    print("models named in commits when they differ from the PR:",
          withm[withm.cat == "other only"][["side", "pr_model", "models"]].astype(str).value_counts().head(12).to_dict())
else:
    print("attrib_tight.jsonl not found")
