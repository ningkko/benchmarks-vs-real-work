"""Build pseudonymized per-pull-request tables from the frozen raw snapshot.

usage: python3 -I make_public_data.py CLEAN_DIR OUT_DIR SALT_FILE
Developer logins and repository names are replaced by salted SHA-256 prefixes; titles, bodies and URLs are dropped.
The salt is written to SALT_FILE (keep it private) so the tables can be re-linked to the raw data if ever needed.
"""
import sys, os, json, re, hashlib, secrets, glob, csv
import pandas as pd

CLEAN, OUT, SALT_FILE = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(OUT, exist_ok=True)
if os.path.exists(SALT_FILE):
    salt = open(SALT_FILE).read().strip()
else:
    salt = secrets.token_hex(16); open(SALT_FILE, "w").write(salt)

TAG = re.compile(r"Co-Authored-By:\s*Claude\s+(Opus|Sonnet|Haiku|Fable|Mythos)\s*(\d(?:\.\d)?)?", re.I)
CL = re.compile(r"Co-Authored-By:\s*Claude", re.I)
DEFAULT_LINE = re.compile(r"Generated with \[Claude Code\]", re.I)


def h(s):
    return hashlib.sha256((salt + str(s).lower()).encode()).hexdigest()[:12]


def model_of(body):
    tags = sorted({m.group(1).title() + (" " + m.group(2) if m.group(2) else "") for m in TAG.finditer(body or "")})
    return tags[0] if len(tags) == 1 else ("MIXED" if tags else "untagged")


def rows_of(path):
    for l in open(path):
        try:
            r = json.loads(l)
        except Exception:
            continue
        if r.get("empty"):
            continue
        body = r.get("body") or ""
        c = pd.to_datetime(r.get("created_at"), utc=True, errors="coerce")
        m = pd.to_datetime(r.get("merged_at"), utc=True, errors="coerce")
        hrs = (m - c).total_seconds() / 3600 if (pd.notna(m) and pd.notna(c)) else None
        repo = r.get("repo") or ""
        user = r.get("user") or ""
        title = r.get("title") or ""
        yield {
            "pr_id": h(r.get("html_url")),
            "dev_id": h(user),
            "repo_id": h(repo),
            "user_type": r.get("user_type"),
            "assoc": r.get("assoc"),
            "own_repo": repo.split("/")[0].lower() == user.lower() if repo and user else None,
            "draft": r.get("draft"),
            "state": r.get("state"),
            "created_at": r.get("created_at"),
            "merged": pd.notna(m),
            "merged_7d": bool(pd.notna(m) and hrs is not None and hrs <= 168),
            "merged_1h": bool(pd.notna(m) and hrs is not None and hrs <= 1),
            "closed_unmerged": r.get("state") == "closed" and pd.isna(m),
            "hours_to_merge": round(hrs, 3) if hrs is not None else None,
            "comments": r.get("comments"),
            "model": model_of(body),
            "claude_line": bool(CL.search(body)),
            "default_pr_line": bool(DEFAULT_LINE.search(body)),
            "body_len": len(body),
            "body_len_full": r.get("body_len_full"),
            "body_files_mentioned": r.get("body_files_mentioned"),
            "title_feat": title.lower().startswith("feat"),
            "n_labels": len(r.get("labels") or []),
        }


manifest = []
groups = {
    "suspension": "data_fable/*.jsonl",
    "february": "data/*.jsonl",
    "upgrade_events": "events/*.jsonl",
    "calendar_control": "calendar/*.jsonl",
    "volume": "volume/*.jsonl",
    "all_pull_requests": "subst/*.jsonl",
}
for g, pat in groups.items():
    for f in sorted(glob.glob(f"{CLEAN}/{pat}")):
        name = os.path.basename(f).replace(".jsonl", "")
        out = f"{OUT}/{g}__{name}.csv"
        n = 0
        with open(out, "w", newline="") as fh:
            w = None
            for row in rows_of(f):
                if w is None:
                    w = csv.DictWriter(fh, fieldnames=list(row.keys())); w.writeheader()
                w.writerow(row); n += 1
        if n == 0:
            os.remove(out)
        manifest.append((g, name, n))
        print(g, name, n)

# merger + reviews sample (join on html_url, then drop it)
S = pd.read_csv(f"{CLEAN}/merger/sample.csv")
mer = []
for f in sorted(glob.glob(f"{CLEAN}/merger/out_*.jsonl")):
    for l in open(f):
        try: mer.append(json.loads(l))
        except Exception: pass
M = pd.DataFrame(mer)
rev = []
for f in sorted(glob.glob(f"{CLEAN}/reviews/out_*.jsonl")):
    for l in open(f):
        try: rev.append(json.loads(l))
        except Exception: pass
R = pd.DataFrame(rev)
if len(M):
    X = S.merge(M, on=["repo", "number"], how="left")
    X["pr_id"] = X.html_url.map(h); X["dev_id"] = X.user.map(h); X["repo_id"] = X.repo.map(h)
    X["merged_by_author"] = X.apply(lambda r: (str(r.get("merged_by")).lower() == str(r.get("user")).lower()) if pd.notna(r.get("merged_by")) else None, axis=1)
    keep = ["pr_id", "dev_id", "repo_id", "assoc", "phase", "model", "merged_7d", "grp", "merged", "merged_by_author", "additions", "deletions", "changed_files", "commits"]
    X[[c for c in keep if c in X.columns]].to_csv(f"{OUT}/merger_sample.csv", index=False)
    print("merger_sample", len(X))
if len(R):
    print("reviews columns:", list(R.columns)[:12])
    Rk = R.copy()
    for c in ["repo", "number"]:
        if c not in Rk.columns:
            break
    else:
        Rk = Rk.merge(S[["repo", "number", "user", "html_url"]], on=["repo", "number"], how="left")
        Rk["pr_id"] = Rk.html_url.map(h)
        BOT = re.compile(r"(\[bot\]|bot)$", re.I)
        if "reviewer" in Rk.columns:
            Rk["reviewer_is_author"] = Rk.apply(lambda r: str(r.reviewer).lower() == str(r.user).lower(), axis=1)
            Rk["reviewer_is_bot"] = Rk.reviewer.astype(str).str.contains(BOT)
            Rk["reviewer_id"] = Rk.reviewer.map(h)
        drop = [c for c in ["repo", "number", "user", "html_url", "reviewer"] if c in Rk.columns]
        Rk.drop(columns=drop).to_csv(f"{OUT}/reviews_sample.csv", index=False)
        print("reviews_sample", len(Rk))

# reverts: pseudonymized match table
for f in ["reverts_suspension", "reverts_events"]:
    p = f"{CLEAN}/revert/{f}.jsonl"
    if os.path.exists(p):
        rr = []
        for l in open(p):
            try: rr.append(json.loads(l))
            except Exception: pass
        D = pd.DataFrame(rr)
        if len(D):
            for c in list(D.columns):
                if c in ("repo",):
                    D["repo_id"] = D[c].map(h)
                if c in ("html_url", "revert_url", "url"):
                    D[c + "_id"] = D[c].map(h)
            D = D.drop(columns=[c for c in D.columns if c in ("repo", "html_url", "revert_url", "url", "title", "body", "user")])
            D.to_csv(f"{OUT}/{f}.csv", index=False)
            print(f, len(D), list(D.columns)[:10])

with open(f"{OUT}/MANIFEST.txt", "w") as fh:
    for g, name, n in manifest:
        fh.write(f"{g}\t{name}\t{n} rows\n")


# Last step: an exact creation second plus a pull request number can be searched on GitHub and would lead back
# to the author. Floor every timestamp to the hour (the phase cut-offs are on the hour, and hours_to_merge and the
# merged-within flags keep the timing the analyses use) and drop pull request numbers.
HOUR = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}):\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:?\d{2})?$")


def coarsen(path):
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        return
    keep = [i for i, c in enumerate(rows[0]) if c != "number"]
    out = [[HOUR.sub(lambda m: m.group(1) + ":00:00Z", r[i] if i < len(r) else "") for i in keep] for r in rows]
    with open(path, "w", newline="") as fh:
        csv.writer(fh).writerows(out)


for path in glob.glob(f"{OUT}/*.csv"):
    coarsen(path)
lines = open(f"{OUT}/MANIFEST.txt").read().splitlines()
with open(f"{OUT}/MANIFEST.txt", "w") as fh:
    for line in lines:
        fh.write(line.replace("\t0 rows", "\t0 rows (empty in the raw data, no file)") + "\n")
