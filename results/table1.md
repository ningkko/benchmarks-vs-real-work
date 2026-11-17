# Table 1. Baseline balance and treated developers by phase

All pull requests the developers opened (tagged or not), May 26 to July 15, 2026; the 297 treated developers of the main design and the sampled comparison developers. Panel A uses P0 (before the Fable 5 release) only. SMD = standardized difference (difference in means over the pooled SD; pull requests per developer, comments and description length on the log(1+x) scale). Panel B SMDs compare each phase with P2, the suspension.

## Panel A. Treated vs comparison, P0

| Variable | Treated | Comparison | Comparison, weighted | SMD | SMD, weighted |
|---|---|---|---|---|---|
| Developers with at least one pull request | 244 | 315 | 312.6 (effective) |  |  |
| Pull requests | 21,377 | 30,630 | 30,630 |  |  |
| Pull requests per developer, median [IQR] | 34 [10-88] | 35 [10-89] | 32 [10-84] | -0.06 | -0.02 |
| Merged within 7 days (%) | 92.7 | 91.5 | 92.7 | +0.05 | -0.00 |
| Merged within 1 hour (%) | 73.9 | 71.5 | 72.8 | +0.05 | +0.03 |
| Closed without merge (%) | 6.0 | 6.8 | 5.6 | -0.03 | +0.02 |
| Comments per pull request, mean | 0.98 | 0.82 | 0.79 | +0.12 | +0.13 |
| Opened in the author's own repository (%) | 65.9 | 67.3 | 65.9 | -0.03 | -0.00 |
| Author association: OWNER (%) | 65.9 | 67.3 | 65.9 | -0.03 | -0.00 |
| Author association: MEMBER (%) | 5.1 | 5.3 | 5.9 | -0.01 | -0.03 |
| Author association: COLLABORATOR (%) | 10.2 | 10.6 | 11.1 | -0.02 | -0.03 |
| Author association: CONTRIBUTOR (%) | 18.6 | 16.0 | 16.5 | +0.07 | +0.05 |
| Author association: NONE (%) | 0.3 | 0.7 | 0.6 | -0.07 | -0.05 |
| Title starts with "feat" (%) | 23.5 | 31.5 | 31.0 | -0.18 | -0.17 |
| Description length, median characters | 1393 | 1259 | 1277 | +0.15 | +0.13 |

Entropy balancing: 315 comparison developers with P0 pull requests; effective number 312.6; largest weight 1.12 (weights scaled to mean 1). Balanced exactly on the P0 share merged within 7 days, the P0 share in the author's own repository and P0 pull requests per developer.

## Panel B. Treated developers by phase

| Variable | P0 | P1 | P2 | P3 | SMD P0 vs P2 | SMD P1 vs P2 | SMD P3 vs P2 |
|---|---|---|---|---|---|---|---|
| Developers with at least one pull request | 244 | 297 | 253 | 234 |  |  |  |
| Pull requests | 21,377 | 9,202 | 24,399 | 21,929 |  |  |  |
| Pull requests per developer, median [IQR] | 34 [10-88] | 16 [6-38] | 30 [8-94] | 24 [6-81] | +0.03 | -0.42 | -0.08 |
| Merged within 7 days (%) | 92.7 | 94.6 | 91.8 | 90.5 | +0.04 | +0.11 | -0.04 |
| Merged within 1 hour (%) | 73.9 | 76.9 | 64.7 | 60.9 | +0.20 | +0.27 | -0.08 |
| Closed without merge (%) | 6.0 | 4.1 | 6.9 | 7.6 | -0.04 | -0.12 | +0.03 |
| Comments per pull request, mean | 0.98 | 0.75 | 1.48 | 1.54 | -0.22 | -0.37 | +0.10 |
| Opened in the author's own repository (%) | 65.9 | 66.0 | 59.6 | 62.5 | +0.13 | +0.13 | +0.06 |
| Author association: OWNER (%) | 65.9 | 66.0 | 59.6 | 62.5 | +0.13 | +0.13 | +0.06 |
| Author association: MEMBER (%) | 5.1 | 5.8 | 7.6 | 6.0 | -0.10 | -0.07 | -0.06 |
| Author association: COLLABORATOR (%) | 10.2 | 9.4 | 10.9 | 10.7 | -0.02 | -0.05 | -0.01 |
| Author association: CONTRIBUTOR (%) | 18.6 | 18.4 | 21.3 | 20.0 | -0.07 | -0.07 | -0.03 |
| Author association: NONE (%) | 0.3 | 0.3 | 0.6 | 0.7 | -0.05 | -0.03 | +0.02 |
| Title starts with "feat" (%) | 23.5 | 26.9 | 22.5 | 20.6 | +0.02 | +0.10 | -0.05 |
| Description length, median characters | 1393 | 1273 | 1506 | 1683 | -0.02 | -0.10 | +0.19 |

## Suspension model with entropy-balanced weights

Seven-day merging on all pull requests; developer and calendar-day fixed effects; CR1 by developer; percentage points.

| Model | Reversal contrast [95% CI] | P0 placebo [95% CI] | Treated PRs | Comparison PRs (developers) |
|---|---|---|---|---|
| unweighted (R17 model, comparison developers with P0 pull requests) | +0.5 [-2.3, +3.3] | -1.6 [-4.2, +1.1] | 76,907 | 72,405 (315) |
| entropy-balanced comparison | +0.4 [-2.4, +3.3] | -1.9 [-4.6, +0.9] | 76,907 | 72,405 (315) |
