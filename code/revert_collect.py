"""Pull requests with "revert" in the title, per repository, for the repositories of the
treated developers' pull requests (suspension) and of the switchers' pull requests (events 3, 4, 6, 7, 8).

usage: python3 -I revert_collect.py BASE [suspension|events]
"""
import json, sys, os, datetime as dt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collect_search as cs
import pandas as pd

BASE = sys.argv[1]
MODE = sys.argv[2] if len(sys.argv) > 2 else "suspension"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import EVENTS, INCLUDED, build
OUT = f"{BASE}/revert"
os.makedirs(OUT, exist_ok=True)

if MODE == "suspension":
    recs = [json.loads(l) for l in open(f"{BASE}/data_fable/users_treated.jsonl")]
    repos = sorted({r["repo"] for r in recs if r.get("repo") and r.get("merged_at")})
    A, B = "2026-05-26", "2026-08-31"
    path = f"{OUT}/reverts_suspension.jsonl"
else:
    repos, lo, hi = set(), None, None
    for k, d, old, new, f_old, f_new in EVENTS:
        if k not in INCLUDED:
            continue
        D = build(BASE, k, d, old, new, f_old, f_new)
        sw = D[(D.treat == 1) & (D.merged_7d == 1)]
        repos |= set(sw.repo)
        d0 = pd.Timestamp(d)
        lo = d0 - pd.Timedelta(days=7) if lo is None else min(lo, d0 - pd.Timedelta(days=7))
        hi = d0 + pd.Timedelta(days=45) if hi is None else max(hi, d0 + pd.Timedelta(days=45))
    repos = sorted(repos)
    A, B = lo.strftime("%Y-%m-%d"), hi.strftime("%Y-%m-%d")
    path = f"{OUT}/reverts_events.jsonl"

done = set()
if os.path.exists(path):
    for line in open(path):
        r = json.loads(line)
        if r.get("_done"):
            done.add(r["_repo"])
todo = [r for r in repos if r not in done]
print(f"{MODE}: {len(repos)} repositories, {len(todo)} to collect, window {A}..{B}", flush=True)
with open(path, "a") as out:
    for i, repo in enumerate(todo):
        n = cs.window(out, f"repo:{repo} is:pr revert in:title", A, B)
        out.write(json.dumps({"_done": True, "_repo": repo, "n": n}) + "\n"); out.flush()
        if i % 50 == 0:
            print(f"  {i}/{len(todo)} {repo} {n} {dt.datetime.now(dt.UTC):%H:%M:%S}", flush=True)
print("REVERTDONE", flush=True)
