"""Every pull request, with or without a Claude co-author line, opened from
May 26 to July 16, 2026 by the treated developers and by the sampled control developers of the suspension design.
Several authors share one search query (author: qualifiers are OR-ed). Stores the same fields as collect_search.

usage: python3 -I subst_collect.py BASE
"""
import json, sys, os, datetime as dt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collect_search as cs

BASE = sys.argv[1]
OUT = f"{BASE}/subst"
os.makedirs(OUT, exist_ok=True)
A, B = "2026-05-26", "2026-07-16"


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


for grp in ["treated", "control"]:
    logins = sorted({l.strip() for l in open(f"{BASE}/data_fable/{grp}_logins.txt") if l.strip()})
    path = f"{OUT}/{grp}_all.jsonl"
    done = set()
    if os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            if r.get("_done"):
                done.update(r["_logins"])
    todo = [u for u in logins if u not in done]
    print(f"{grp}: {len(todo)} developers to collect ({len(done)} done)", flush=True)
    with open(path, "a") as out:
        for i, bt in enumerate(batches(todo)):
            base = "is:pr " + " ".join(f"author:{u}" for u in bt)
            n = cs.window(out, base, A, B)
            if n == 0:
                for u in bt:
                    m = cs.window(out, f"is:pr author:{u}", A, B)
                    out.write(json.dumps({"_done": True, "_logins": [u], "n": m, "single": True}) + "\n")
            else:
                out.write(json.dumps({"_done": True, "_logins": bt, "n": n}) + "\n")
            out.flush()
            if i % 10 == 0:
                print(f"  {grp} batch {i}: {len(bt)} developers, {n} PRs {dt.datetime.utcnow():%H:%M:%S}", flush=True)
print("SUBSTDONE", flush=True)
