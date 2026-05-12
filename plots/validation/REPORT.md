# Simulator validation report
_Generated from 10 simulator maps (32×32×440 cells each) vs the full NLOG corpus._
## Pass / fail summary
- Run-length Wasserstein > 30 cells (formation, rock) pairs: **1**
- KS > 0.30 (rock, depth_bin, variable) cells: **33**
- Worst transition-matrix Frobenius: **0.76** (RB)

## 1. Run-length distributions — worst 10 by Wasserstein
| fm | rock | n_real_runs | n_sim_runs | real_mean_cells | sim_mean_cells | wasserstein_cells |
| --- | --- | --- | --- | --- | --- | --- |
| CK | chalk | 1026 | 10240 | 68.1 | 123.4 | 55.66 |
| NU | clay | 699 | 5120 | 46.1 | 64.6 | 27.41 |
| SG | claystone_hot | 30 | 3025 | 27.1 | 6.3 | 20.79 |
| AT | claystone_hot | 256 | 2048 | 25.3 | 7.2 | 18.08 |
| SL | claystone | 157 | 1024 | 21.7 | 6.0 | 16.54 |
| NL | clay | 705 | 5120 | 37.5 | 48.6 | 15.31 |
| SG | claystone | 102 | 3001 | 18.0 | 3.0 | 14.97 |
| ZE | anhydrite | 841 | 18418 | 14.0 | 13.0 | 13.32 |
| AT | claystone_cool | 110 | 980 | 12.3 | 1.0 | 11.34 |
| RB | claystone_hot | 728 | 6144 | 20.4 | 9.3 | 11.13 |

## 2. Facies fractions — worst 10 by |mean diff|
| fm | rock | real_mean | real_p5 | real_p95 | sim_mean | sim_p5 | sim_p95 | abs_mean_diff |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SG | claystone_hot | 0.219 | 0.000 | 1.000 | 0.728 | 0.636 | 0.857 | 0.508 |
| SG | claystone | 0.781 | 0.000 | 1.000 | 0.272 | 0.143 | 0.364 | 0.508 |
| KN | claystone_cool | 0.437 | 0.000 | 1.000 | 0.042 | 0.000 | 0.255 | 0.395 |
| SL | claystone_cool | 0.516 | 0.000 | 1.000 | 0.886 | 0.882 | 0.891 | 0.370 |
| SL | claystone | 0.484 | 0.000 | 1.000 | 0.114 | 0.109 | 0.118 | 0.370 |
| KN | claystone | 0.468 | 0.000 | 0.963 | 0.782 | 0.373 | 1.000 | 0.314 |
| RN | claystone | 0.520 | 0.070 | 1.000 | 0.822 | 0.667 | 0.947 | 0.302 |
| DC | claystone_hot | 0.391 | 0.000 | 1.000 | 0.688 | 0.452 | 0.942 | 0.297 |
| DC | claystone | 0.609 | 0.000 | 1.000 | 0.312 | 0.058 | 0.548 | 0.297 |
| RB | sandstone_shaly | 0.279 | 0.000 | 0.964 | 0.569 | 0.000 | 1.000 | 0.289 |

## 3. Transition matrices — Frobenius + KL per formation
| fm | n_real_trans | n_sim_trans | frobenius | max_row_kl | mean_row_kl |
| --- | --- | --- | --- | --- | --- |
| RB | 22148 | 125104 | 0.758 | 0.883 | 0.309 |
| RO | 18036 | 455361 | 0.738 | 0.858 | 0.286 |
| AT | 9767 | 24488 | 0.737 | 0.795 | 0.286 |
| RN | 15293 | 25294 | 0.715 | 0.519 | 0.292 |
| SG | 2627 | 26176 | 0.424 | 0.225 | 0.149 |
| KN | 36347 | 390344 | 0.196 | 0.077 | 0.035 |
| SL | 8416 | 52627 | 0.171 | 0.069 | 0.035 |
| ZE | 42297 | 806718 | 0.109 | 0.112 | 0.037 |
| DC | 10176 | 676423 | 0.049 | 0.005 | 0.004 |
| NM | 4475 | 15358 | 0.049 | 0.000 | 0.000 |
| CK | 69309 | 1253268 | 0.007 | 0.000 | 0.000 |
| NU | 31445 | 325712 | 0.006 | 0.000 | 0.000 |
| NL | 26099 | 243735 | 0.006 | 0.000 | 0.000 |

## 4. Petrophysical marginals — worst 10 by KS
| rock | bin | var | n_real | n_sim | ks |
| --- | --- | --- | --- | --- | --- |
| sandstone_clean | 6 | res_deep_log | 53 | 38936 | 0.721 |
| chalk | 0 | rhob | 190 | 199680 | 0.709 |
| chalk | 4 | nphi | 104 | 221526 | 0.612 |
| chalk | 4 | res_deep_log | 424 | 221526 | 0.595 |
| claystone_cool | 5 | nphi | 87 | 39323 | 0.590 |
| sandstone_clean | 9 | res_deep_log | 174 | 47050 | 0.565 |
| clay | 0 | nphi | 55 | 199680 | 0.554 |
| clay | 0 | res_deep_log | 47 | 199680 | 0.519 |
| sandstone_shaly | 5 | res_deep_log | 117 | 98455 | 0.507 |
| sandstone_clean | 10 | res_deep_log | 54 | 18143 | 0.498 |

## 5. Variable bounds — empirical vs hard-set (Task 5a)
| var | hard_lo | hard_hi | empirical_lo | empirical_hi | tighter_lo_by | tighter_hi_by |
| --- | --- | --- | --- | --- | --- | --- |
| dt_us_ft | 40.000 | 240.000 | 48.506 | 196.265 | +8.506 | -43.735 |
| gr_api | 0.000 | 300.000 | 4.705 | 170.076 | +4.705 | -129.924 |
| nphi | -0.050 | 0.600 | -0.031 | 0.555 | +0.019 | -0.045 |
| pef | 1.000 | 10.000 | 1.471 | 9.369 | +0.471 | -0.631 |
| res_deep_log | -1.000 | 5.000 | -0.605 | 3.306 | +0.395 | -1.694 |
| rhob | 1.200 | 3.200 | 1.568 | 2.978 | +0.368 | -0.222 |

## 6. Basin coverage of the combination pool (Task 5b)
_k-means k=5 on (x_rd, y_rd) of the 624 NLOG wells with coords. Basins below 5 wells are dropped from stratified sampling._
| basin | centroid_x_rd_km | centroid_y_rd_km | wells_in_pool | pool_pct | uniform_pct | unique_combinations |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 58 | 603 | 38 | 48.1% | 33.3% | 17 |
| 3 | 26 | 680 | 22 | 27.8% | 33.3% | 11 |
| 4 | 126 | 655 | 19 | 24.1% | 33.3% | 13 |

## Caveats
- **Effective sample size for thin formations.** The simulator draws ONE base column per map, then produces the (32×32) grid by spatially wiggling that column's layer boundaries.  So a formation appearing in N maps' chosen combinations contributes N independent Markov-chain realisations, replicated 1024 times each.  For thin formations (AT, RN, SG, SL) that only appear in a few combinations, facies-fraction and transition-matrix metrics with `--n-maps 5–10` are dominated by which 1–3 base realisations got drawn.  Increase `--n-maps` to 30+ for stable per-formation marginals; the run-length distributions converge faster because they aggregate over all (x,y) copies of each realisation.

## Plots
- `runlength_<FM>.png`         — run-length histograms
- `facies_fractions_<FM>.png`  — facies fraction box plots
- `transition_compare_<FM>.png` — real / sim / diff heatmaps
- `transition_matrices/<FM>_transition.png` — Task-2 fitted matrices
- `grf_perturbation_example.png` — Task-1 layer-boundary GRF demo
- `variogram_check.png`         — Task-4 variogram fit (gaussian_filter vs gstools)
- `basin_distribution.png`      — Task-5b k-means basins + pool coverage
