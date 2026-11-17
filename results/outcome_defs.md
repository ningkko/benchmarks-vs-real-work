# Outcome definitions and time to merge

Data collected up to Nov 2, 2026. Percentage points with 95% CI. Suspension: reversal contrast (P1+P3)/2 minus P2, developer and calendar-day fixed effects, CR1 by developer, one weight per pull request. Upgrades: overall change after minus before among switchers, developer fixed effects, events 3, 4, 6, 7, 8 pooled by inverse-variance weighting.

## Outcome definitions

| Outcome | Suspension: reversal [95% CI] | Suspension: P0 placebo [95% CI] | Treated level in P2 (%) | Upgrades, 5 pooled [95% CI] |
|---|---|---|---|---|
| merged within 1 hour | -0.0 [-5.8, +5.8] | +7.8 [-2.4, +18.0] | 66.3 | -0.0 [-1.9, +1.8] |
| merged within 1 day | +0.5 [-4.1, +5.1] | -2.9 [-7.2, +1.5] | 89.2 | -0.8 [-2.1, +0.6] |
| merged within 7 days (main) | +1.4 [-1.4, +4.3] | -2.6 [-6.2, +1.1] | 91.3 | -0.9 [-1.9, +0.2] |
| merged within 28 days | +0.6 [-2.1, +3.3] | -3.2 [-6.7, +0.3] | 92.0 | -0.3 [-1.3, +0.8] |
| merged within 7 days, not reverted within 30 | +1.5 [-1.6, +4.5] | -2.5 [-6.1, +1.2] | 91.0 | -0.6 [-1.7, +0.4] |

Not reverted: in the suspension model reverts were collected for treated repositories only, so comparison rows carry 0 (as in revert.py).

## Hours from creation to merge (merged pull requests)

| Group | Merged pull requests | Median hours | IQR |
|---|---|---|---|
| suspension: treated P0 | 1,288 | 0.17 | 0.03 to 1.28 |
| suspension: treated P1 | 1,440 | 0.04 | 0.00 to 0.27 |
| suspension: treated P2 | 1,500 | 0.19 | 0.03 to 1.23 |
| suspension: treated P3 | 1,650 | 0.39 | 0.04 to 2.36 |
| suspension: comparison, all phases | 31,100 | 0.04 | 0.00 to 0.30 |
| upgrades: event 3 before | 675 | 0.24 | 0.06 to 1.69 |
| upgrades: event 3 after | 683 | 0.16 | 0.06 to 1.16 |
| upgrades: event 4 before | 1,369 | 0.06 | 0.00 to 0.54 |
| upgrades: event 4 after | 1,918 | 0.07 | 0.04 to 0.30 |
| upgrades: event 6 before | 3,060 | 0.04 | 0.00 to 0.37 |
| upgrades: event 6 after | 2,238 | 0.06 | 0.00 to 0.42 |
| upgrades: event 7 before | 1,897 | 0.10 | 0.01 to 0.74 |
| upgrades: event 7 after | 1,929 | 0.07 | 0.00 to 0.58 |
| upgrades: event 8 before | 470 | 0.04 | 0.00 to 0.38 |
| upgrades: event 8 after | 461 | 0.09 | 0.01 to 0.66 |

## Within-developer difference in time to merge

| Design | Contrast | Change in log(1+hours) [95% CI] | As % change [95% CI] |
|---|---|---|---|
| suspension | Fable phases minus suspension, log(1+hours), developer and day FE | +0.081 [-0.094, +0.256] | +8.4% [-9.0%, +29.2%] |
| upgrades | after minus before, log(1+hours), developer FE, 5 events pooled | +0.024 [-0.022, +0.069] | +2.4% [-2.1%, +7.2%] |
