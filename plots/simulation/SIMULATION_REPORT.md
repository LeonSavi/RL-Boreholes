# Simulator validation report

Compared **50 synthetic maps** against `data/clean/samples.parquet` across four axes: petrophysical marginals, formation composition, vertical sequence statistics, and spatial structure.

## 1. DistributionBank — what the simulator draws from

![Distribution bank](01_distribution_bank_overview.png)

**Figure 1** — empirical KDE of `rhob` for 8 representative (rock × depth-bin) cells.  The simulator samples directly from these KDEs; their shapes therefore *are* the simulator's petrophysical priors.

## 2. Per-rock marginals — sim vs real

![rhob](02_marginal_sim_vs_real_rhob.png)

![gr](03_marginal_sim_vs_real_gr.png)

**Worst-fit (rock × variable)** by KS statistic:

| rock | variable | KS | Wasserstein | sim p50 | real p50 |
|---|---|---|---|---|---|
| clay | dt_us_ft | 0.3169 | 23.149 | 127.4083 | 158.298 |
| halite_pure | rhob | 0.2829 | 0.1234 | 2.1743 | 2.0761 |
| other | rhob | 0.279 | 0.2098 | 2.5907 | 2.395 |
| anhydrite | rhob | 0.2749 | 0.1545 | 2.3251 | 2.1137 |
| claystone_hot | dt_us_ft | 0.268 | 6.521 | 68.9669 | 73.4411 |

(KS < 0.10 = excellent fit, < 0.30 = good, > 0.40 = systematic mismatch worth investigating.)

## 3. Formation composition — sim vs real

![composition](04_formation_composition.png)

Total-variation distance per formation (lower = closer match to real corpus composition):

| formation | TVD (%) |
|---|---|
| CK | 0.0% |
| NU | 0.0% |
| NM | 0.0% |
| NL | 0.0% |
| SL | 1.3% |
| RO | 1.9% |
| RB | 4.5% |
| ZE | 4.8% |
| DC | 7.2% |
| AT | 7.6% |
| RN | 10.7% |
| SG | 20.8% |
| KN | 23.4% |
| SK | 50.0% |
| other | 50.0% |

## 4. Vertical sequences — Markov transition matrices

![transitions](05_transition_matrices.png)

![run lengths](06_run_length_distributions.png)

Frobenius distance between sim and real transition matrices per formation (aligned on the union of observed rocks):

| formation | Frobenius | n rocks |
|---|---|---|
| CK | 0.0 | 1 |
| SL | 0.0791 | 2 |
| DC | 0.1421 | 2 |
| AT | 0.1534 | 3 |
| KN | 0.1585 | 3 |
| RN | 0.1604 | 3 |
| ZE | 0.2336 | 6 |
| RB | 0.9667 | 4 |
| RO | 0.9906 | 3 |

## 5. Spatial structure

![sample column](07_sample_map_cross_section.png)

![lateral coherence](08_lateral_coherence.png)

Figure 7 shows one (x, y) column with its rock sequence and three variable curves.  Figure 8 shows lateral variability across the x-axis at fixed y — formation boundaries should wiggle smoothly (thanks to the anisotropic Gaussian random field used as boundary perturbation), not jump.

## 6. Summary

![summary](09_metric_summary.png)

CSV tables alongside this report contain the full per-cell numbers:

- `per_rock_marginal_distances.csv` — KS + Wasserstein per (rock × variable)
- `per_formation_composition.csv` — sim/real rock-fractions + TVD
- `per_formation_transition_distance.csv` — Frobenius per formation
