"""Sample check, part 2: merged share of tagged pull requests (A) vs Claude Code pull requests with only the default line (B).

usage: python3 -I r21_part2.py OUT_DIR
"""
import sys, json, time, subprocess, urllib.parse, math
OUT = sys.argv[1]
WEEKS = [("2026-04-09", "2026-04-15"), ("2026-04-16", "2026-04-22"), ("2026-05-19", "2026-05-25"),
         ("2026-05-26", "2026-06-01"), ("2026-06-02", "2026-06-08"), ("2026-06-09", "2026-06-15"),
         ("2026-06-16", "2026-06-22"), ("2026-06-23", "2026-06-29"), ("2026-06-30", "2026-07-06"),
         ("2026-07-07", "2026-07-13"), ("2026-07-14", "2026-07-20"), ("2026-07-21", "2026-07-27"),
         ("2026-07-28", "2026-08-03"), ("2026-09-15", "2026-09-21"), ("2026-09-22", "2026-09-28"),
         ("2026-09-29", "2026-10-05")]
NOBOT = "-author:app/dependabot -author:app/renovate -author:app/github-actions -author:app/claude -author:app/shipyard-local"
KIND = {"A_model_named": '("Co-Authored-By: Claude Opus" OR "Co-Authored-By: Claude Sonnet" OR "Co-Authored-By: Claude Fable") in:body',
        "B_default_only": '"Generated with Claude Code" -"Co-Authored-By" in:body'}


def search(q, per_page=1, sort=None):
    url = f"search/issues?q={urllib.parse.quote(q)}&per_page={per_page}"
    if sort:
        url += f"&sort={sort}&order=desc"
    for attempt in range(6):
        r = subprocess.run(["gh", "api", url], capture_output=True, text=True)
        time.sleep(2.2)
        if r.returncode == 0:
            return json.loads(r.stdout)
        time.sleep(30 * (attempt + 1))
    raise RuntimeError(r.stderr)


def wilson(k, n):
    if n == 0:
        return (float("nan"),) * 2
    z = 1.96; p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


import os
JL = f"{OUT}/r21_part2_rows.jsonl"
rows = [json.loads(l) for l in open(JL)] if os.path.exists(JL) else []
done = {r["week"] for r in rows}
for a, b in WEEKS:
    if f"{a}..{b}" in done:
        continue
    r = {"week": f"{a}..{b}"}
    for k, q in KIND.items():
        base = f"is:pr {q} created:{a}..{b} " + NOBOT
        tot = search(base)["total_count"]; mer = search(base + " is:merged")["total_count"]
        page = search(base, per_page=100, sort="created")["items"]
        users = [i for i in page if (i.get("user") or {}).get("type") == "User"]
        r.update({f"{k}_total": tot, f"{k}_merged": mer, f"{k}_share": mer / tot * 100 if tot else float("nan"),
                  f"{k}_page_n": len(page), f"{k}_page_bot_pct": (1 - len(users) / len(page)) * 100 if page else float("nan"),
                  f"{k}_page_user_merged_pct": sum(1 for i in users if (i.get("pull_request") or {}).get("merged_at")) / len(users) * 100 if users else float("nan")})
        lo, hi = wilson(mer, tot); r[f"{k}_lo"], r[f"{k}_hi"] = lo * 100, hi * 100
    r["diff_A_minus_B"] = r["A_model_named_share"] - r["B_default_only_share"]
    rows.append(r); open(JL, "a").write(json.dumps(r) + "\n"); print(r["week"], "done", flush=True)
json.dump(rows, open(f"{OUT}/r21_part2.json", "w"), indent=1)
