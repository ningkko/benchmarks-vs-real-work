"""Collect every Claude-tagged pull request (any model) that switchers and stayers opened around each included
upgrade (post-hoc check, planned before collection). Several authors share one search query; repeated
author: qualifiers are OR-ed by GitHub search. A batch that returns nothing is retried one author at a time.

usage: python3 -I volume_collect.py BASE
"""
import json, sys, os, datetime as dt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collect_search as cs

BASE = sys.argv[1]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import EVENTS, INCLUDED, build

OUT = f"{BASE}/volume"
groups = {}
if os.path.exists(f"{OUT}/groups.json"):
    groups = json.load(open(f"{OUT}/groups.json"))
for k, d, old, new, f_old, f_new in EVENTS:
    if k not in INCLUDED or str(k) in groups:
        continue
    D = build(BASE, k, d, old, new, f_old, f_new)
    groups[str(k)] = {"date": d, "switchers": sorted(set(D[D.treat == 1].user)),
                      "stayers": sorted(set(D[D.treat == 0].user))}
    json.dump(groups, open(f"{OUT}/groups.json", "w"))
print({k: (len(v["switchers"]), len(v["stayers"])) for k, v in groups.items()}, flush=True)

PREFIX = 'is:pr "Co-Authored-By: Claude" in:body'


def batches(logins, limit=150):
    cur, size = [], 0
    for u in logins:
        piece = len(u) + 8
        if cur and size + piece > limit:
            yield cur
            cur, size = [], 0
        cur.append(u); size += piece
    if cur:
        yield cur


for k, g in groups.items():
    d0 = dt.date.fromisoformat(g["date"])
    a, b = (d0 - dt.timedelta(days=7)).isoformat(), (d0 + dt.timedelta(days=7)).isoformat()
    path = f"{OUT}/v{k}.jsonl"
    done = set()
    if os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            if r.get("_done"):
                done.update(r["_logins"])
    logins = [u for u in sorted(set(g["switchers"]) | set(g["stayers"])) if u not in done]
    print(f"event {k}: {len(logins)} developers to collect ({len(done)} done), window {a}..{b}", flush=True)
    with open(path, "a") as out:
        for i, bt in enumerate(batches(logins)):
            base = PREFIX + " " + " ".join(f"author:{u}" for u in bt)
            n = cs.window(out, base, a, b)
            if n == 0:  # every developer here has pull requests in the window, so 0 means the query failed
                for u in bt:
                    m = cs.window(out, f"{PREFIX} author:{u}", a, b)
                    out.write(json.dumps({"_done": True, "_logins": [u], "n": m, "single": True}) + "\n")
            else:
                out.write(json.dumps({"_done": True, "_logins": bt, "n": n}) + "\n")
            out.flush()
            if i % 20 == 0:
                print(f"  event {k} batch {i}: {len(bt)} developers, {n} PRs", flush=True)
    print(f"event {k} done {dt.datetime.utcnow():%H:%M:%S}", flush=True)
print("VOLDONE", flush=True)
