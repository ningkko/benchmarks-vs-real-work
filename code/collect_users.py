"""For each developer login, collect their Claude Code PRs in a date window via GitHub search.

usage: python3 -I collect_users.py LOGINS.txt OUT.jsonl START END COHORT
"""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collect_search as cs


def main():
    src, dst, start, end, cohort = sys.argv[1:6]
    done = set()
    if os.path.exists(dst):
        for line in open(dst):
            try:
                done.add(json.loads(line)["_login"])
            except Exception:
                pass
    logins = [l.strip() for l in open(src) if l.strip() and l.strip() not in done]
    print(f"{len(logins)} logins to collect ({len(done)} done)", flush=True)
    with open(dst, "a") as out:
        for i, login in enumerate(logins):
            q = f'is:pr author:{login} "Co-Authored-By: Claude" in:body created:{start}..{end}'
            page, n = 1, 0
            while True:
                res = cs.search(q, page)
                if res is None:
                    break
                items = res.get("items", [])
                for it in items:
                    rec = cs.keep(it); rec["_login"] = login; rec["cohort"] = cohort
                    out.write(json.dumps(rec) + "\n"); n += 1
                total = res.get("total_count", 0)
                if len(items) < 100 or page * 100 >= min(total, 1000):
                    break
                page += 1
            if n == 0:  # remember logins with no hits so we do not re-query
                out.write(json.dumps({"_login": login, "cohort": cohort, "empty": True}) + "\n")
            out.flush()
            if i % 25 == 0:
                print(f"{i}/{len(logins)} {login}: {n}", flush=True)


if __name__ == "__main__":
    main()
