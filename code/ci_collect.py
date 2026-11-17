"""Collect CI results (check runs and commit statuses) at each pull request's head commit.

usage: python3 -I ci_collect.py TARGETS.csv OUT.jsonl [--max-seconds N] [--batch 20]

TARGETS.csv comes from ci_targets.py (columns sample, html_url, new). One GraphQL request reads up to
`--batch` pull requests: for each, the head commit (last commit of the pull request) and its
statusCheckRollup contexts, i.e. every check run (name, app, status, conclusion) and every commit-status
context (context, state). This is the same information as the REST calls pulls/{n},
commits/{sha}/check-runs and commits/{sha}/status, in far fewer requests.

Token: GITHUB_TOKEN, or the first line of .github_token in the project folder. Resumable: pull requests
already in OUT.jsonl are skipped. Standard library only.
"""
import sys, os, json, time, csv, re, urllib.request, urllib.error

TARGETS, OUT = sys.argv[1], sys.argv[2]
MAXS = float(sys.argv[sys.argv.index("--max-seconds") + 1]) if "--max-seconds" in sys.argv else 1e12
BATCH = int(sys.argv[sys.argv.index("--batch") + 1]) if "--batch" in sys.argv else 20
T0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
TOKEN = os.environ.get("GITHUB_TOKEN")
p = os.path.join(HERE, "..", ".github_token")
if not TOKEN and os.path.exists(p):
    TOKEN = open(p).read().strip().splitlines()[0]
if not TOKEN:
    sys.exit("GraphQL needs a token: set GITHUB_TOKEN or write .github_token")

FIELDS = """
  pullRequest(number: %d) {
    state merged headRefOid
    commits(last: 1) { nodes { commit { oid
      statusCheckRollup { state contexts(first: 100) { totalCount nodes {
        __typename
        ... on CheckRun { name status conclusion checkSuite { app { slug } } }
        ... on StatusContext { context state }
      } } }
    } } }
  }"""


def gql(query, tries=3):
    body = json.dumps({"query": query}).encode()
    for attempt in range(tries):
        req = urllib.request.Request("https://api.github.com/graphql", data=body, method="POST",
                                     headers={"Authorization": f"Bearer {TOKEN}", "User-Agent": "r22-ci"})
        try:
            with urllib.request.urlopen(req, timeout=45) as r:
                rem = int(r.headers.get("X-RateLimit-Remaining", "100"))
                if rem < 20:
                    time.sleep(max(1, int(r.headers.get("X-RateLimit-Reset", time.time() + 60)) - time.time()) + 2)
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            wait = int(e.headers.get("Retry-After", "0") or 0)
            if e.headers.get("X-RateLimit-Remaining") == "0":
                wait = int(e.headers.get("X-RateLimit-Reset", time.time() + 60)) - time.time() + 2
            if e.code in (502, 504) and not wait:
                return None                       # GitHub gave up on a heavy query: the caller splits the batch
            time.sleep(max(wait, 15 * (attempt + 1)))
        except Exception:
            time.sleep(5 * (attempt + 1))
    return None


URL = re.compile(r"https://github.com/([^/]+)/([^/]+)/pull/(\d+)$")
targets = []
seen = set()
for r in csv.DictReader(open(TARGETS)):
    if r["html_url"] not in seen:
        seen.add(r["html_url"]); targets.append(r["html_url"])
done = set()
if os.path.exists(OUT):
    for l in open(OUT):
        try: done.add(json.loads(l)["html_url"])
        except Exception: pass
todo = [u for u in targets if u not in done]
print(f"targets {len(targets)}, done {len(done)}, to do {len(todo)}", flush=True)
out = open(OUT, "a")


def fetch(chunk):
    """One GraphQL request for a list of URLs; on failure split the list in half, down to single pull requests."""
    parts = []
    for j, u in enumerate(chunk):
        o, n, num = URL.match(u).groups()
        parts.append(f'r{j}: repository(owner: {json.dumps(o)}, name: {json.dumps(n)}) {{{FIELDS % int(num)}}}')
    res = gql("query {\n" + "\n".join(parts) + "\n}", tries=2 if len(chunk) > 1 else 3)
    if (res is None or (res.get("data") is None and res.get("errors"))) and len(chunk) > 1:
        h = len(chunk) // 2
        return fetch(chunk[:h]) + fetch(chunk[h:])
    data = (res or {}).get("data") or {}
    errs = {}
    for e in (res or {}).get("errors", []) or []:
        if e.get("path"):
            errs[e["path"][0]] = e.get("type") or e.get("message", "")[:80]
    recs = []
    for j, u in enumerate(chunk):
        node = (data.get(f"r{j}") or {}).get("pullRequest") if data.get(f"r{j}") else None
        rec = {"html_url": u}
        if node is None:
            rec["missing"] = errs.get(f"r{j}", "no data" if res else "request failed")
        else:
            commits = (node.get("commits") or {}).get("nodes") or []
            c = commits[0]["commit"] if commits else None
            roll = (c or {}).get("statusCheckRollup")
            rec.update({"state": node.get("state"), "merged": node.get("merged"), "head": node.get("headRefOid"),
                        "last_commit": (c or {}).get("oid"), "rollup": (roll or {}).get("state"),
                        "n_contexts": ((roll or {}).get("contexts") or {}).get("totalCount", 0),
                        "checks": [{"t": x.get("__typename"), "name": x.get("name") or x.get("context"),
                                    "app": ((x.get("checkSuite") or {}).get("app") or {}).get("slug"),
                                    "status": x.get("status"), "conclusion": x.get("conclusion") or x.get("state")}
                                   for x in (((roll or {}).get("contexts") or {}).get("nodes") or [])]})
        recs.append(rec)
    return recs


for i in range(0, len(todo), BATCH):
    if time.time() - T0 > MAXS:
        print("STOPPED_EARLY", flush=True); sys.exit(0)
    for rec in fetch(todo[i:i + BATCH]):
        out.write(json.dumps(rec) + "\n")
    out.flush()
    if (i // BATCH) % 25 == 0:
        print(f"{i + BATCH} / {len(todo)}", flush=True)
out.close()
print("CI_DONE", flush=True)
