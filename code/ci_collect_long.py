"""Re-read pull requests with more than 100 CI contexts, following all pages (ci_collect.py reads the first 100).

usage: python3 -I ci_collect_long.py CI_RAW.jsonl
Appends a complete record for each such pull request; readers keep the last record per pull request.
"""
import sys, os, json, re, time, urllib.request
RAW = sys.argv[1]
TOKEN = os.environ.get("GITHUB_TOKEN") or open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".github_token")).read().split()[0]
URL = re.compile(r"https://github.com/([^/]+)/([^/]+)/pull/(\d+)$")
Q = """query($o:String!,$n:String!,$num:Int!,$after:String){repository(owner:$o,name:$n){pullRequest(number:$num){
 commits(last:1){nodes{commit{statusCheckRollup{contexts(first:100,after:$after){totalCount pageInfo{hasNextPage endCursor}
 nodes{__typename ... on CheckRun{name status conclusion checkSuite{app{slug}}} ... on StatusContext{context state}}}}}}}}}}"""


def gql(v):
    for a in range(4):
        try:
            req = urllib.request.Request("https://api.github.com/graphql", method="POST",
                                         data=json.dumps({"query": Q, "variables": v}).encode(),
                                         headers={"Authorization": f"Bearer {TOKEN}", "User-Agent": "r22-ci"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())
        except Exception:
            time.sleep(10 * (a + 1))
    return None


last = {}
for l in open(RAW):
    r = json.loads(l); last[r["html_url"]] = r
long_ = [r for r in last.values() if r.get("n_contexts", 0) > 100 and len(r.get("checks", [])) < r["n_contexts"]]
print("to complete", len(long_))
out = open(RAW, "a")
for r in long_:
    o, n, num = URL.match(r["html_url"]).groups()
    checks, after = [], None
    while True:
        res = gql({"o": o, "n": n, "num": int(num), "after": after})
        try:
            ctx = res["data"]["repository"]["pullRequest"]["commits"]["nodes"][0]["commit"]["statusCheckRollup"]["contexts"]
        except Exception:
            checks = None; break
        checks += [{"t": x.get("__typename"), "name": x.get("name") or x.get("context"),
                    "app": ((x.get("checkSuite") or {}).get("app") or {}).get("slug"),
                    "status": x.get("status"), "conclusion": x.get("conclusion") or x.get("state")} for x in ctx["nodes"]]
        if not ctx["pageInfo"]["hasNextPage"]:
            break
        after = ctx["pageInfo"]["endCursor"]
    if checks is not None:
        r = dict(r, checks=checks, complete=True)
        out.write(json.dumps(r) + "\n"); out.flush()
print("done")
