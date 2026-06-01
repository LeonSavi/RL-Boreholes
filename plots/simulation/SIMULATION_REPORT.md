# Simulator validation report

Compared **500 synthetic maps** against `data/clean/samples.parquet` across four axes: petrophysical marginals, formation composition, vertical sequence statistics, and spatial structure.

## 1. DistributionBank — what the simulator draws from

![Distribution bank](01_distribution_bank_overview.png)

**Figure 1** — empirical KDE of `rhob` for 8 representative (rock × depth-bin) cells.  The simulator samples directly from these KDEs; their shapes therefore *are* the simulator's petrophysical priors.

## 2. Per-rock marginals — sim vs real

![rhob](02_marginal_sim_vs_real_rhob.png)

![gr](03_marginal_sim_vs_real_gr.png)

**Worst-fit (rock × variable)** by KS statistic:

| rock | variable | KS | Wasserstein | sim p50 | real p50 |
|---|---|---|---|---|---|
| other | dt_us_ft | 0.9033 | 66.0582 | 88.8388 | 171.0438 |
| other | rhob | 0.8449 | 0.4774 | 2.3678 | 1.943 |
| clay | dt_us_ft | 0.3105 | 21.887 | 127.9042 | 158.298 |
| halite_pure | rhob | 0.3072 | 0.1248 | 2.1923 | 2.0767 |
| anhydrite | rhob | 0.2759 | 0.1402 | 2.35 | 2.1212 |

(KS < 0.10 = excellent fit, < 0.30 = good, > 0.40 = systematic mismatch worth investigating.)

## 3. Formation composition — sim vs real

![composition](04_formation_composition.png)

Total-variation distance per formation (lower = closer match to real corpus composition):

| formation | TVD (%) |
|---|---|
| CK | 0.0% |
| DC | 0.0% |
| NM | 0.0% |
| NL | 0.0% |
| NU | 0.0% |
| SL | 0.0% |
| SG | 0.0% |
| ZE | 0.9% |
| RB | 6.8% |
| AT | 8.1% |
| RO | 8.2% |
| KN | 9.0% |
| RN | 10.3% |

## 4. Vertical sequences — Markov transition matrices

![transitions](05_transition_matrices.png)

![run lengths](06_run_length_distributions.png)

Frobenius distance between sim and real transition matrices per formation (aligned on the union of observed rocks):

| formation | Frobenius | n rocks |
|---|---|---|
| CK | 0.0 | 1 |
| DC | 0.0 | 1 |
| SL | 0.0 | 1 |
| RB | 0.0915 | 3 |
| AT | 0.0933 | 2 |
| RN | 0.1006 | 2 |
| ZE | 0.1012 | 4 |
| RO | 0.1064 | 2 |
| KN | 0.1441 | 2 |

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
