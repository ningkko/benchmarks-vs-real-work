"""Collect PR-level records from the GitHub search API via `gh api`.

usage: python3 -I collect_search.py OUT.jsonl "QUERY WITHOUT DATE" START END [step_days]
Splits [START, END] into windows of step_days (default 1); if a window has
more than 1000 hits it is split further by halving (down to hourly is not
needed for our volumes; we warn instead).
"""
import json, subprocess, sys, time, datetime as dt

SLEEP = 2.05  # search API: 30 requests / minute -> one call every 2 s, latency included
_last = [0.0]


def _pace():
    wait = SLEEP - (time.time() - _last[0])
    if wait > 0:
        time.sleep(wait)
    _last[0] = time.time()


def search(q, page):
    cmd = ["gh", "api", "-X", "GET", "search/issues", "-f", f"q={q}",
           "-f", "per_page=100", "-f", f"page={page}"]
    for attempt in range(6):
        _pace()
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0:
            return json.loads(r.stdout)
        msg = (r.stdout + r.stderr)[:300]
        if "Validation Failed" in msg or "422" in msg:
            print("SKIP422", q[:80], file=sys.stderr)
            return None
        if "rate limit" in msg.lower() or "403" in msg or "secondary" in msg.lower():
            time.sleep(15 * (attempt + 1))
            continue
        print("ERR", q, page, msg, file=sys.stderr)
        time.sleep(5)
    return None


def keep(it):
    pr = it.get("pull_request") or {}
    return {
        "html_url": it.get("html_url"),
        "number": it.get("number"),
        "repo": (it.get("repository_url") or "").replace("https://api.github.com/repos/", ""),
        "user": (it.get("user") or {}).get("login"),
        "user_type": (it.get("user") or {}).get("type"),
        "assoc": it.get("author_association"),
        "state": it.get("state"),
        "draft": it.get("draft"),
        "created_at": it.get("created_at"),
        "closed_at": it.get("closed_at"),
        "merged_at": pr.get("merged_at"),
        "comments": it.get("comments"),
        "title": it.get("title"),
        "body": (it.get("body") or "")[-2500:],  # trailer lives at the end
        "body_len_full": len(it.get("body") or ""),
        "body_bullets_full": (it.get("body") or "").count("\n- ") + (it.get("body") or "").count("\n* "),
        "body_checkboxes_full": (it.get("body") or "").count("- [ ]") + (it.get("body") or "").count("- [x]"),
        "body_files_mentioned": len(set(__import__("re").findall(r"[\w./-]+\.(?:py|ts|tsx|js|jsx|go|rs|java|rb|md|yml|yaml|json|sql|sh|css|html|swift|kt|c|cpp|h)\b", it.get("body") or ""))),
        "labels": [l.get("name") for l in it.get("labels") or []],
    }


def window(out, base, a, b):
    q = f"{base} created:{a}..{b}"
    first = search(q, 1)
    if first is None:
        return 0
    total = first.get("total_count", 0)
    if total > 1000:
        f = lambda t: t.strftime("%Y-%m-%dT%H:%M:%SZ")
        if "T" in a:  # sub-day window: halve by time down to 30 minutes
            ta = dt.datetime.fromisoformat(a.replace("Z", "")); tb = dt.datetime.fromisoformat(b.replace("Z", ""))
            if (tb - ta) > dt.timedelta(minutes=30):
                mid = ta + (tb - ta) / 2
                return window(out, base, a, f(mid)) + window(out, base, f(mid + dt.timedelta(seconds=1)), b)
        elif a == b:  # one day: split into halves
            return (window(out, base, f"{a}T00:00:00Z", f"{a}T11:59:59Z") +
                    window(out, base, f"{a}T12:00:00Z", f"{a}T23:59:59Z"))
        else:
            da = dt.date.fromisoformat(a); db = dt.date.fromisoformat(b)
            mid = da + (db - da) // 2
            return window(out, base, a, mid.isoformat()) + window(out, base, (mid + dt.timedelta(days=1)).isoformat(), b)
    if total > 1000:
        print(f"WARN window {a} has {total} > 1000 hits; truncated", file=sys.stderr)
    n = 0
    items = first.get("items", [])
    page = 1
    while True:
        for it in items:
            rec = keep(it); rec["query"] = base
            out.write(json.dumps(rec) + "\n"); n += 1
        if len(items) < 100 or page * 100 >= min(total, 1000):
            break
        page += 1
        nxt = search(q, page)
        if nxt is None:
            break
        items = nxt.get("items", [])
    out.flush()
    return n


def main():
    path, base, start, end = sys.argv[1:5]
    step = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    d = dt.date.fromisoformat(start); e = dt.date.fromisoformat(end)
    tot = 0
    with open(path, "a") as out:
        while d <= e:
            b = min(d + dt.timedelta(days=step - 1), e)
            n = window(out, base, d.isoformat(), b.isoformat())
            tot += n
            print(f"{base[:60]} {d}..{b}: {n} (cum {tot})", flush=True)
            d = b + dt.timedelta(days=1)


if __name__ == "__main__":
    main()
