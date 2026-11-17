#!/bin/bash
# Rebuild every number and figure in the paper from the raw GitHub API responses, in one run.
#
# usage:  bash code/run_all.sh RAW_DIR
#
# RAW_DIR holds the raw responses (data/, data_fable/, events/, calendar/, subst/, merger/, revert/, review/,
# reviews/, volume/). They contain GitHub logins and are not released. The script copies RAW_DIR and code/ into
# build/, runs every analysis in order, writes results/results_all.txt with the CSV and markdown tables, and
# redraws figures/.
set -u
P="$(cd "$(dirname "$0")/.." && pwd)"
RAW="${1:-$P/private_do_not_share/raw}"
B="$P/build"
rm -rf "$B"
mkdir -p "$B/scripts" "$P/results" "$P/figures"
cp -R "$RAW"/. "$B/"
cp "$P"/code/*.py "$B/scripts/"
T=""; command -v timeout >/dev/null 2>&1 && T="timeout 1800"
LOG="$B/results_all.txt"
: > "$LOG"
for s in descriptives twfe robust decider repo_fe final_results leaderboard pooled_vs_within twfe_feb power events \
         calendar_check posthoc volume review_checks trend_check subst merger revert eventstudy reviews allprs \
         outcome_defs table1 robust2; do
  echo "################ $s" >> "$LOG"
  echo "running $s"
  $T python3 -I "$B/scripts/$s.py" "$B" >> "$LOG" 2>&1
done
for w in twfe twfe_feb; do
  echo "################ sens_cap $w" >> "$LOG"
  $T python3 -I "$B/scripts/sens_cap.py" "$B" "$w" >> "$LOG" 2>&1
done

# record the size and checksum of every input file, so a rerun can be matched to the same snapshot
FILES="data/*.jsonl data_fable/*.jsonl events/*.jsonl calendar/*.jsonl volume/*.jsonl review/attrib.jsonl
       review/attrib_tight.jsonl subst/*.jsonl merger/out_*.jsonl revert/*.jsonl reviews/out_*.jsonl"
SHA="sha256sum"; command -v sha256sum >/dev/null 2>&1 || SHA="shasum -a 256"
echo "################ snapshot" >> "$LOG"
( cd "$B" && wc -l $FILES && $SHA $FILES volume/groups.json ) >> "$LOG"
echo ALLDONE >> "$LOG"

cp "$LOG" "$P/results/results_all.txt"
cp "$B"/*.csv "$B"/*.md "$P/results/" 2>/dev/null
cp "$B/events/event_estimates.csv" "$B/calendar/calendar_estimates.csv" "$B/calendar/posthoc_estimates.csv" \
   "$B/volume/volume_estimates.csv" "$P/results/" 2>/dev/null
python3 -I "$P/code/make_paper_figures.py" "$P/results" "$P/fonts" "$P/figures"
echo "done: results/results_all.txt and figures/"
