"""Sample check, part 1: commits with a Claude model co-author line -> their pull requests.

Same sample and coding as r21_part1.py, but it finds each commit's pull request with the REST
endpoint repos/{repo}/commits/{sha}/pulls instead of the search API, so it runs fast with a
personal token. No gh CLI needed (standard library only).

usage: python3 -I r21_part1_core.py OUT_DIR [--max-seconds N]

Token: GITHUB_TOKEN in the environment, or the first line of the file .github_token in the
project folder (git-ignored). A classic or fine-grained token with public read access is enough.
Without a token it still runs, but GitHub allows only 60 core calls per hour.

Resumable: rows already in OUT_DIR/r21_part1_commits.jsonl are skipped. With --max-seconds the
run stops cleanly after N seconds, so it can be called in chunks.
"""
import sys, os, json, time, re, random, urllib.request, urllib.parse, urllib.error

OUT = sys.argv[1]
MAXS = float(sys.argv[sys.argv.index("--max-seconds") + 1]) if "--max-seconds" in sys.argv else 1e12
T0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
DAYS = {"2026-06-03": "P0", "2026-06-10": "P1", "2026-06-22": "P2", "2026-07-08": "P3",
        "2026-04-20": "upgrade", "2026-07-28": "upgrade", "2026-09-24": "upgrade", "2026-10-01": "upgrade"}
MODEL = re.compile(r"Co-Authored-By:\s*Claude\s+(Opus|Sonnet|Haiku|Fable|Mythos)\s*(\d(?:\.\d)?)?", re.I)
DEFAULT = re.compile(r"Generated with \[?Claude Code", re.I)
rng = random.Random(11)          # same seed and draw order as r21_part1.py -> same hours

TOKEN = os.environ.get("GITHUB_TOKEN")
for cand in (os.path.join(HERE, "..", ".github_token"), os.path.join(OUT, "..", ".github_token")):
    if not TOKEN and os.path.exists(cand):
        TOKEN = open(cand).read().strip().splitlines()[0]


def api(path):
    url = "https://api.github.com/" + path
    for attempt in range(6):
        req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json", "User-Agent": "r21-part1",
                                                   **({"Authorization": f"Bearer {TOKEN}"} if TOKEN else {})})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                rem = int(r.headers.get("X-RateLimit-Remaining", "1"))
                if rem == 0:
                    time.sleep(max(1, int(r.headers.get("X-RateLimit-Reset", time.time() + 60)) - time.time()) + 1)
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code in (404, 409, 422):
                return None
            wait = int(e.headers.get("Retry-After", "0") or 0)
            if e.code == 403 and e.headers.get("X-RateLimit-Remaining") == "0":
                wait = int(e.headers.get("X-RateLimit-Reset", time.time() + 60)) - time.time() + 1
            time.sleep(max(wait, 20 * (attempt + 1)))
        except Exception:
            time.sleep(10 * (attempt + 1))
    return None


os.makedirs(OUT, exist_ok=True)
path = f"{OUT}/r21_part1_commits.jsonl"
done = {json.loads(l)["sha"] for l in open(path)} if os.path.exists(path) else set()
hours_done = set()
hp = f"{OUT}/r21_part1_hours_done.txt"
if os.path.exists(hp):
    hours_done = set(open(hp).read().split())
out = open(path, "a")
search_gap = 2.1 if TOKEN else 6.5    # search API: 30/min with a token, 10/min without
for day, ph in DAYS.items():
    hours = rng.sample(range(24), 4)
    for h in hours:
        key = f"{day}T{h:02d}"
        if key in hours_done:
            continue
        if time.time() - T0 > MAXS:
            print("STOPPED_EARLY", flush=True); sys.exit(0)
        q = f'"Co-Authored-By: Claude" committer-date:{day}T{h:02d}:00:00Z..{day}T{h:02d}:59:59Z'
        res = api(f"search/commits?q={urllib.parse.quote(q)}&per_page=50")
        time.sleep(search_gap)
        for it in (res or {}).get("items", []):
            sha = it["sha"]
            if sha in done:
                continue
            if time.time() - T0 > MAXS:
                print("STOPPED_EARLY", flush=True); sys.exit(0)
            m = MODEL.search(it["commit"]["message"])
            rec = {"sha": sha, "day": day, "phase": ph, "hour": h, "repo": it["repository"]["full_name"],
                   "commit_model": (m.group(1).title() + (" " + m.group(2) if m.group(2) else "")) if m else None,
                   "method": "core"}
            if m:
                prs = api(f"repos/{rec['repo']}/commits/{sha}/pulls") or []
                prs = [p for p in prs if p.get("base", {}).get("repo", {}).get("full_name") == rec["repo"]] or prs
                if prs:
                    p = prs[0]; body = p.get("body") or ""
                    rec.update({"pr": p["html_url"], "pr_user_type": (p.get("user") or {}).get("type"),
                                "pr_merged": bool(p.get("merged_at")),
                                "desc_model": bool(MODEL.search(body)), "desc_default": bool(DEFAULT.search(body))})
            out.write(json.dumps(rec) + "\n"); out.flush(); done.add(sha)
        open(hp, "a").write(key + "\n"); hours_done.add(key)
        print(day, ph, h, "done", flush=True)
out.close()
print("PART1_DONE", flush=True)
