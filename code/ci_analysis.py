"""CI failure around the upgrade events and in the suspension supplement.

usage: python3 -I ci_analysis.py BASE CI_RAW.jsonl
BASE is the run folder of run_all.sh (raw data plus scripts/). Writes BASE/ci_events.csv and prints every estimate.
"""
import os, sys, json, re
import numpy as np, pandas as pd
BASE, RAW = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import EVENTS, fe_ols, pool, switchers

TB = re.compile(r"test|build|compile|unit|integration|e2e|spec|pytest|jest|vitest|mocha|cargo|gradle|maven|tox|ctest|make", re.I)
EXCL = re.compile(r"lint|format|prettier|eslint|ruff|label|title|semantic|size|codecov|coverage|cla|license|dependabot|vercel|"
                  r"netlify|preview|deploy|review|claude|copilot|coderabbit|sonar|snyk|stale", re.I)


def ci_fields(rec):
    checks = rec.get("checks") or []
    runs = [c for c in checks if c["t"] == "CheckRun"]
    stat = [c for c in checks if c["t"] == "StatusContext"]
    done = [c for c in runs if (c.get("status") or "").upper() == "COMPLETED"]
    has = len(done) + len(stat) > 0
    bad = lambda c: (c["t"] == "CheckRun" and (c.get("conclusion") or "").upper() in ("FAILURE", "TIMED_OUT")) or \
                    (c["t"] == "StatusContext" and (c.get("conclusion") or "").upper() in ("FAILURE", "ERROR"))
    rel = done + stat
    tb = [c for c in rel if TB.search(c.get("name") or "") and not EXCL.search(c.get("name") or "")]
    return dict(missing=int("missing" in rec), has_ci=int(has), ci_fail=int(any(bad(c) for c in rel)) if has else np.nan,
                has_tb=int(len(tb) > 0), tb_fail=int(any(bad(c) for c in tb)) if tb else np.nan,
                n_checks=len(rel), n_tb=len(tb), n_neither=sum(1 for c in rel if not TB.search(c.get("name") or "") and
                                                             not EXCL.search(c.get("name") or "")),
                n_excl=sum(1 for c in rel if EXCL.search(c.get("name") or "")))


CI = {}
for l in open(RAW):
    r = json.loads(l)
    CI[r["html_url"]] = ci_fields(r)
C = pd.DataFrame.from_dict(CI, orient="index")
print(f"pull requests read {len(C)}; missing {int(C.missing.sum())} ({C.missing.mean() * 100:.1f}%)")
ok = C[C.missing == 0]
print(f"with CI {int(ok.has_ci.sum())} ({ok.has_ci.mean() * 100:.1f}%); CI failure among them {ok.ci_fail.mean() * 100:.1f}%; "
      f"with a test/build check {int(ok.has_tb.sum())} ({ok.has_tb.mean() * 100:.1f}%), test/build failure {ok.tb_fail.mean() * 100:.1f}%")
nck = ok.n_checks.sum()
print(f"checks: {int(nck)}; test/build {ok.n_tb.sum() / nck * 100:.1f}%, excluded kinds {ok.n_excl.sum() / nck * 100:.1f}%, "
      f"neither (left out of test/build) {ok.n_neither.sum() / nck * 100:.1f}%")


def est(D, y, X):
    d = D.dropna(subset=[y])
    d = d[d.groupby("user")["new"].transform("nunique") > 0]
    b, V = fe_ols(d, y, X)
    return b, V, len(d), d.user.nunique()


rows = []
print("\n== upgrade events: change in each outcome, new minus old model, developer fixed effects (percentage points)")
for k, d, old, new, f_old, f_new in EVENTS:
    if k not in (3, 4, 6, 7, 8):
        continue
    a, b, sw = switchers(BASE, d, old, new, f_old, f_new)
    a["new"] = 0.0; b["new"] = 1.0
    D = pd.concat([a[a.user.isin(sw)], b[b.user.isin(sw)]], ignore_index=True).drop_duplicates("html_url")
    D["external"] = (~D.assoc.isin(["OWNER", "MEMBER", "COLLABORATOR"])).astype(float)
    D["new_x_ext"] = D.new * D.external
    D = D.join(C, on="html_url")
    D["merged7"] = D.merged_7d.astype(float)
    miss = D.missing.fillna(1)
    r = dict(event=k, old=old, new=new, prs=len(D), missing_pct=miss.mean() * 100,
             missing_old=miss[D.new == 0].mean() * 100, missing_new=miss[D.new == 1].mean() * 100)
    Dok = D[miss == 0].copy()
    r["has_ci_old"] = Dok[Dok.new == 0].has_ci.mean() * 100; r["has_ci_new"] = Dok[Dok.new == 1].has_ci.mean() * 100
    Dci = Dok[Dok.has_ci == 1].copy()
    r["ci_prs"] = len(Dci); r["ci_devs"] = Dci.user.nunique()
    r["fail_old"] = Dci[Dci.new == 0].ci_fail.mean() * 100; r["fail_new"] = Dci[Dci.new == 1].ci_fail.mean() * 100
    for lab, frame, y in [("ci_fail", Dci, "ci_fail"), ("tb_fail", Dok, "tb_fail"), ("has_ci", Dok, "has_ci"),
                          ("merged7_ci", Dci, "merged7")]:
        bb, V, n, g = est(frame, y, ["new"])
        r[f"{lab}_est"], r[f"{lab}_se"], r[f"{lab}_n"] = bb[0] * 100, np.sqrt(V[0, 0]) * 100, n
    bb, V, n, g = est(Dci, "ci_fail", ["new", "external", "new_x_ext"])
    r["inter_est"], r["inter_se"] = bb[2] * 100, np.sqrt(V[2, 2]) * 100
    rows.append(r)
R = pd.DataFrame(rows)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
show = R[["event", "old", "new", "prs", "missing_pct", "ci_prs", "ci_devs", "has_ci_old", "has_ci_new", "fail_old", "fail_new",
          "ci_fail_est", "ci_fail_se", "tb_fail_est", "tb_fail_se", "merged7_ci_est", "merged7_ci_se", "inter_est", "inter_se"]]
print(show.round(1).to_string(index=False))
R.to_csv(f"{BASE}/ci_events.csv", index=False)
print("\n== pooled over the five events (inverse-variance; DerSimonian-Laird as sensitivity)")
for lab in ["ci_fail", "tb_fail", "has_ci", "merged7_ci", "inter"]:
    P = pool(R[f"{lab}_est"], R[f"{lab}_se"])
    print(f"{lab:12s} FE {P['fe']:+.2f} [{P['fe'] - 1.96 * P['fe_se']:+.2f}, {P['fe'] + 1.96 * P['fe_se']:+.2f}] p={P['p_fe']:.3f} | "
          f"RE {P['re']:+.2f} [{P['re'] - 1.96 * P['re_se']:+.2f}, {P['re'] + 1.96 * P['re_se']:+.2f}] I2={P['I2']:.2f}")
print("missing by side (%): old", round(R.missing_old.mean(), 1), "new", round(R.missing_new.mean(), 1))

# ---------------- suspension supplement
print("\n== suspension supplement (2,019 sampled pull requests; treated developers only; no comparison group)")
M = pd.read_csv(f"{BASE}/merger/sample.csv").join(C, on="html_url")
M = M[M.missing == 0].copy()
print(f"read {len(M)}; with CI {int(M.has_ci.sum())} ({M.has_ci.mean() * 100:.1f}%)")
print("CI failure % by phase (PRs with CI):", (M[M.has_ci == 1].groupby("phase").ci_fail.mean() * 100).round(1).to_dict())
Mc = M[M.has_ci == 1].copy()
for p_ in ["P0", "P1", "P3"]:
    Mc[p_] = (Mc.phase == p_).astype(float)
for y in ["ci_fail", "tb_fail"]:
    d = Mc.dropna(subset=[y])
    b, V = fe_ols(d, y, ["P0", "P1", "P3"])
    c = np.array([0, 0.5, 0.5]); e = c @ b * 100; se = np.sqrt(c @ V @ c) * 100
    print(f"{y}: reversal (P1+P3)/2 - P2 {e:+.1f} [{e - 1.96 * se:+.1f}, {e + 1.96 * se:+.1f}] (n={len(d)}, developers {d.user.nunique()}); "
          f"P0 vs P2 {b[0] * 100:+.1f} [{(b[0] - 1.96 * np.sqrt(V[0, 0])) * 100:+.1f}, {(b[0] + 1.96 * np.sqrt(V[0, 0])) * 100:+.1f}]")
    d = d.assign(fable=(d.model == "Fable 5").astype(float), unit=d.user + "|" + d.repo)
    b, V = fe_ols(d, y, ["fable"], unit="unit")
    print(f"{y}: as used, Fable 5 vs other, developer x repository FE {b[0] * 100:+.1f} "
          f"[{(b[0] - 1.96 * np.sqrt(V[0, 0])) * 100:+.1f}, {(b[0] + 1.96 * np.sqrt(V[0, 0])) * 100:+.1f}]")
