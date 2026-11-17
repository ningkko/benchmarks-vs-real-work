"""Attribution check: for a random sample of merged, model-tagged switcher PRs,
search the author's commits in the same repository around the PR and record which Claude models their co-author
lines name. Stores only extracted model names and counts, not commit messages.

usage: python3 -I attrib_collect.py BASE [wide|tight]
wide: 7 days before opening to 1 day after merging; tight: 1 day before opening to 1 day after merging.
"""
import json, re, sys, os, subprocess, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collect_search as cs

BASE = sys.argv[1]
MODE = sys.argv[2] if len(sys.argv) > 2 else "wide"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import EVENTS, INCLUDED, build
import pandas as pd

OUT = f"{BASE}/review/attrib.jsonl" if MODE == "wide" else f"{BASE}/review/attrib_tight.jsonl"
MODEL = re.compile(r"Co-Authored-By:\s*Claude\s+(Opus|Sonnet|Haiku|Fable|Mythos)\s*(\d(?:\.\d)?)?", re.I)
GENERIC = re.compile(r"Co-Authored-By:\s*Claude\s*<", re.I)

frames = []
for k, d, old, new, f_old, f_new in EVENTS:
    if k not in INCLUDED:
        continue
    D = build(BASE, k, d, old, new, f_old, f_new)
    sw = D[(D.treat == 1) & (D.merged_7d == 1)].copy()
    sw["event"] = k
    sw["side"] = sw.after.map({0.0: "before", 1.0: "after"})
    frames.append(sw[["html_url", "repo", "user", "model", "created_at", "merged_at", "event", "side"]])
P = pd.concat(frames, ignore_index=True)
sample = pd.concat([P[P.side == "before"].sample(100, random_state=11), P[P.side == "after"].sample(100, random_state=11)])
done = set()
if os.path.exists(OUT):
    done = {json.loads(l)["html_url"] for l in open(OUT)}
print(f"sample {len(sample)}, done {len(done)}", flush=True)


def commits(q):
    cmd = ["gh", "api", "-X", "GET", "search/commits", "-f", f"q={q}", "-f", "per_page=100"]
    for attempt in range(6):
        cs._pace()
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0:
            return json.loads(r.stdout)
        msg = (r.stdout + r.stderr)[:300]
        if "Validation Failed" in msg or "422" in msg:
            return None
        time.sleep(15 * (attempt + 1))
    return None


with open(OUT, "a") as out:
    for i, r in enumerate(sample.itertuples()):
        if r.html_url in done:
            continue
        a = (pd.Timestamp(r.created_at) - pd.Timedelta(days=7 if MODE == "wide" else 1)).strftime("%Y-%m-%d")
        b = (pd.Timestamp(r.merged_at) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        res = commits(f"repo:{r.repo} author:{r.user} committer-date:{a}..{b}")
        rec = dict(html_url=r.html_url, event=int(r.event), side=r.side, pr_model=r.model, ok=res is not None)
        if res is not None:
            msgs = [it["commit"]["message"] for it in res.get("items", [])]
            found = [f"{m.group(1).title()} {m.group(2)}" if m.group(2) else m.group(1).title()
                     for msg in msgs for m in MODEL.finditer(msg)]
            rec.update(n_commits=res.get("total_count", 0), n_seen=len(msgs),
                       n_with_model=sum(1 for msg in msgs if MODEL.search(msg)),
                       n_generic=sum(1 for msg in msgs if GENERIC.search(msg) and not MODEL.search(msg)),
                       models=sorted(set(found)))
        out.write(json.dumps(rec) + "\n"); out.flush()
        if i % 25 == 0:
            print(i, rec.get("n_commits"), rec.get("models"), flush=True)
print("ATTRDONE", flush=True)
