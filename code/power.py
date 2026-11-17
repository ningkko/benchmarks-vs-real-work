"""Developers needed to detect a 3-point change in the seven-day merge share (80% power, two-sided 0.05),
from the spread of developer-level paired differences.

usage: python3 -I power.py BASE
"""
import os, json, sys, glob, warnings
import numpy as np
warnings.filterwarnings("ignore")
BASE = sys.argv[1]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CLAUDE_LINE, T, prep
z = 1.959964 + 0.841621
def need(d, delta=3.0):
    sd = np.std(np.asarray(d, float) * 100, ddof=1)
    return sd, int(np.ceil((z * sd / delta) ** 2))
rows = [json.loads(l) for l in open(f"{BASE}/data_fable/users_treated.jsonl")]
S = prep(rows)
# the phrase search also returns loose matches; keep only pull requests with a real Claude co-author line
S = S[S.body.fillna('').str.contains(CLAUDE_LINE)].copy()
PHASES = [("P0", T("2026-05-26"), T("2026-06-09 18:00")), ("P1", T("2026-06-09 18:00"), T("2026-06-12 21:00")),
        ("P2", T("2026-06-15"), T("2026-07-01")), ("P3", T("2026-07-02"), T("2026-07-16"))]
S["phase"] = None
for nm, a, b in PHASES:
    S.loc[(S.created_at >= a) & (S.created_at < b), "phase"] = nm
tr = set(S[(S.phase == "P1") & (S.model == "Fable 5")].user)
S = S[S.user.isin(tr) & S.phase.notna()]
out = {}
d = (S[S.phase == "P2"].groupby("user").merged_7d.mean() - S[S.phase == "P1"].groupby("user").merged_7d.mean()).dropna()
out["removal"] = (len(d),) + need(d)
g = S[S.model.isin(["Fable 5", "Opus 4.8"])].groupby(["user", "model"]).merged_7d.mean().unstack("model")
d = (g["Fable 5"] - g["Opus 4.8"]).dropna()
out["as_used"] = (len(d),) + need(d)
feb = []
for f in glob.glob(f"{BASE}/data/cc_*.jsonl"):
    feb += [json.loads(l) for l in open(f)]
Fb = prep(feb)
a = Fb[(Fb.model == "Opus 4.5") & (Fb.created_at < T("2026-02-05")) & (Fb.created_at >= T("2025-12-01"))]
b = Fb[(Fb.model == "Opus 4.6") & (Fb.created_at >= T("2026-02-05")) & (Fb.created_at < T("2026-04-11"))]
d = (b.groupby("user").merged_7d.mean() - a.groupby("user").merged_7d.mean()).dropna()
out["feb_opus"] = (len(d),) + need(d)
for k, (n, sd, req) in out.items():
    print(f"{k:9s} developers {n:4d}  SD of developer differences {sd:5.1f} points  -> developers needed for 3 points: {req}")
