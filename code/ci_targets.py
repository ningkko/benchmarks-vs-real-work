"""List the pull requests whose CI results are collected.

usage: python3 -I ci_targets.py BASE OUT.csv
Upgrade events 3, 4, 6, 7, 8: switcher pull requests built exactly as in events.py.
Suspension supplement: the 2,019 pull requests of merger/sample.csv.
The output holds repository names and URLs, so it stays in the private folder.
"""
import os, sys
import pandas as pd
BASE, OUT = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import EVENTS, switchers
rows = []
for k, d, old, new, f_old, f_new in EVENTS:
    if k not in (3, 4, 6, 7, 8):
        continue
    a, b, sw = switchers(BASE, d, old, new, f_old, f_new)
    a["new"] = 0; b["new"] = 1
    D = pd.concat([a[a.user.isin(sw)], b[b.user.isin(sw)]], ignore_index=True).drop_duplicates("html_url")
    for r in D.itertuples():
        rows.append({"sample": f"event{k}", "html_url": r.html_url, "new": r.new})
M = pd.read_csv(f"{BASE}/merger/sample.csv")
url = "html_url" if "html_url" in M.columns else None
if url is None:
    M["html_url"] = "https://github.com/" + M.repo + "/pull/" + M.number.astype(int).astype(str)
for u in M.html_url:
    rows.append({"sample": "suspension", "html_url": u, "new": ""})
X = pd.DataFrame(rows)
X = X[X.html_url.str.match(r"https://github.com/[^/]+/[^/]+/pull/\d+$")]
X.to_csv(OUT, index=False)
print(X.groupby("sample").size().to_string(), "\ntotal", len(X), "distinct", X.html_url.nunique())
