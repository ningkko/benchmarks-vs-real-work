"""Descriptive numbers quoted in the paper (Sections 3-4, Limitations, Table A3).

usage: python3 -I descriptives.py BASE
"""
import os, json, sys, warnings
import pandas as pd
warnings.filterwarnings("ignore")
BASE = sys.argv[1]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import prep, load_suspension
D = load_suspension(BASE)
t = D[D.treated == 1]
c = D[D.treated == 0]
print("treated developers", t.user.nunique(), "| treated PRs P0-P3", len(t))
print("treated developers active by phase:", t.groupby("phase").user.nunique().to_dict())
both23 = set(t[t.phase == "P2"].user) & set(t[t.phase == "P3"].user)
print("treated developers with PRs in both P2 and P3:", len(both23))
print("control developers", c.user.nunique(), "| control PRs", len(c), "| control days", c.created_at.dt.strftime('%Y-%m-%d').nunique())
fam = t.model.str.split().str[0]
mix = pd.crosstab(t.phase, t.model, normalize="index").mul(100).round(1)
print("model share by phase (treated, %):")
print(mix.loc[:, mix.max() > 1].to_string())
u3 = t[(t.phase == "P3") & (t.model == "untagged")]
if len(u3):
    print("P3 untagged PRs", len(u3), "top developer share %.0f%%" % (u3.user.value_counts().iloc[0] / len(u3) * 100))
print("own-repo share by phase (treated, %):", t.groupby("phase").own_repo.mean().mul(100).round(1).to_dict())
wa = t.assoc.isin(["OWNER", "MEMBER", "COLLABORATOR"])
print("write-access share (treated, %%): %.1f; PRs with write access %d, without %d; developers without write access %d"
      % (wa.mean() * 100, wa.sum(), (~wa).sum(), t[~wa].user.nunique()))
print("outside own repo: treated PRs", int((~t.own_repo).sum()))
p2 = t[t.phase == "P2"]
print("P2 merged within 7 days (treated): %.1f%%" % (p2.merged_7d.mean() * 100))
# Fable-tagged PRs per 6 hours around the directive (all developers)
F = prep([json.loads(l) for l in open(f"{BASE}/data_fable/fable_jun.jsonl")])
F = F[F.model == "Fable 5"]
b = F.groupby(F.created_at.dt.floor("6h")).size()
print("Fable-tagged PRs per 6h, Jun 12-13:", b.loc["2026-06-12":"2026-06-13"].to_dict())
