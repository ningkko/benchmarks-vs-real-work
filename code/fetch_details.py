"""Fetch PR details (size, merged_by, commits) from the GitHub REST pulls API.

usage: python3 -I fetch_details.py IN_URLS.txt OUT.jsonl [threads]
IN_URLS.txt: one PR html_url per line. Already-fetched URLs in OUT are skipped.
"""
import json, sys, time, threading, queue, os, subprocess

lock = threading.Lock()


def parse(url):
    p = url.split("github.com/")[1].split("/")
    return f"{p[0]}/{p[1]}", p[3]


def gh_get(path):
    """Call the REST API through the preconfigured gh CLI; return (status, json)."""
    r = subprocess.run(["gh", "api", path, "-i"], capture_output=True, text=True)
    out = r.stdout
    head, _, body = out.partition("\r\n\r\n")
    if not body:
        head, _, body = out.partition("\n\n")
    try:
        status = int(head.split()[1])
    except Exception:
        status = 0
    try:
        data = json.loads(body) if body.strip() else {}
    except Exception:
        data = {}
    return status, data


def get(url):
    repo, num = parse(url)
    for attempt in range(4):
        status, d = gh_get(f"repos/{repo}/pulls/{num}")
        if status == 200:
            return {
                "html_url": url,
                "merged": d.get("merged"),
                "merged_by": (d.get("merged_by") or {}).get("login"),
                "additions": d.get("additions"), "deletions": d.get("deletions"),
                "changed_files": d.get("changed_files"), "commits": d.get("commits"),
                "review_comments": d.get("review_comments"), "comments": d.get("comments"),
                "draft": d.get("draft"), "base_ref": (d.get("base") or {}).get("ref"),
                "head_ref": (d.get("head") or {}).get("ref"),
                "head_repo": ((d.get("head") or {}).get("repo") or {}).get("full_name"),
                "base_repo": ((d.get("base") or {}).get("repo") or {}).get("full_name"),
                "repo_stars": ((d.get("base") or {}).get("repo") or {}).get("stargazers_count"),
                "repo_fork": ((d.get("base") or {}).get("repo") or {}).get("fork"),
                "repo_private": ((d.get("base") or {}).get("repo") or {}).get("private"),
                "auto_merge": d.get("auto_merge") is not None,
                "status": 200,
            }
        if status in (404, 410, 451):
            return {"html_url": url, "status": status}
        if status in (403, 429):
            time.sleep(60)
            continue
        time.sleep(3)
    return {"html_url": url, "status": "fail"}


def main():
    src, dst = sys.argv[1], sys.argv[2]
    threads = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    done = set()
    if os.path.exists(dst):
        with open(dst) as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["html_url"])
                except Exception:
                    pass
    todo = [u.strip() for u in open(src) if u.strip() and u.strip() not in done]
    print(f"{len(todo)} to fetch ({len(done)} done)", flush=True)
    q = queue.Queue()
    for u in todo:
        q.put(u)
    out = open(dst, "a")
    count = [0]

    def worker():
        while True:
            try:
                u = q.get_nowait()
            except queue.Empty:
                return
            rec = get(u)
            with lock:
                out.write(json.dumps(rec) + "\n"); count[0] += 1
                if count[0] % 250 == 0:
                    out.flush(); print(count[0], flush=True)
            time.sleep(0.25)

    ts = [threading.Thread(target=worker) for _ in range(threads)]
    [t.start() for t in ts]; [t.join() for t in ts]
    out.close(); print("finished", count[0], flush=True)


if __name__ == "__main__":
    main()
