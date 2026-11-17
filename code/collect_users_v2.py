"""Collect each developer's Claude-tagged PRs in a window, splitting windows so nothing is truncated.

usage: python3 -I collect_users_v2.py LOGINS.txt OUT.jsonl START END
Writes one marker line per finished login: {"_login": ..., "empty": true, "done": true}.
"""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collect_search as cs


def main():
    src, dst, start, end = sys.argv[1:5]
    done = set()
    if os.path.exists(dst):
        for line in open(dst):
            d = json.loads(line)
            if d.get("done"):
                done.add(d["_login"])
    logins = [l.strip() for l in open(src) if l.strip() and l.strip() not in done]
    print(f"{len(logins)} logins to collect ({len(done)} done)", flush=True)
    with open(dst, "a") as out:
        for i, login in enumerate(logins):
            base = f'is:pr author:{login} "Co-Authored-By: Claude" in:body'
            n = cs.window(out, base, start, end)
            out.write(json.dumps({"_login": login, "empty": True, "done": True, "n": n}) + "\n")
            out.flush()
            if i % 25 == 0:
                print(f"{i}/{len(logins)} {login}: {n}", flush=True)
    print("finished", flush=True)


if __name__ == "__main__":
    main()
