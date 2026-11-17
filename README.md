# An 11-Point Benchmark Lead Did Not Show Up in Real Work

Data and code for the paper by Yining Hua (Harvard University).

The study uses the 19-day suspension of Claude Fable 5 in June 2026 as a natural experiment, and five model upgrades in Claude Code, to test whether higher scores on SWE-Bench Pro lead to more merged pull requests on public GitHub, and to fewer failed CI checks.

## Contents

- `data/` holds one table per sample, one row per pull request, compressed with xz (`xz -dk data/*.xz`). GitHub logins, repository names and pull request links are replaced by salted hash codes, timestamps are rounded down to the hour, and pull request numbers are removed, so no developer or repository can be identified. `MANIFEST.txt` gives the row counts.
- `code/` holds the collection and analysis scripts. `common.py` has the shared pieces (reading pull requests, the suspension and upgrade samples, the regression models); each analysis script imports from it. `run_all.sh` runs every analysis and writes the numbers and figures in the paper. `make_paper_figures.py` draws the figures from `results/`.
- `results/` holds the output behind every number in the paper: `results_all.txt` (all estimates), the CSV and markdown tables, the CI analysis (`ci_results.txt`, `ci_events.csv`) and the sample check (`r21_summary.md`).
- `figures/` holds the paper's figures (PNG and PDF).
- `fonts/` holds Open Sans (SIL Open Font License), used in the figures.

## Reproducing the results

Python 3.10 or later:

```
pip install -r requirements.txt
python3 code/make_paper_figures.py results fonts figures
```

`bash code/run_all.sh RAW_DIR` reruns every analysis from the raw GitHub API responses. Those responses contain GitHub logins and are not released; the tables in `data/` carry the same per-pull-request variables. The collection scripts query the GitHub API and need a token with public read access in `GITHUB_TOKEN`.

## Citation

If you use the data or code, please cite:

Hua Y. An 11-Point Benchmark Lead Did Not Show Up in Real Work. arXiv preprint, 2026.
