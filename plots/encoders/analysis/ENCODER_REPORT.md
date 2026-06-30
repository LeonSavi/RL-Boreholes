# Encoder evaluation report

Validation set: **3,000 independent borehole columns** (each column has its own freshly-sampled stratigraphic stack).

Encoders evaluated: AE, JEPA, AE_formation, JEPA_formation.

Naming: `*_formation` variants were trained on the formation-resolution dataset (`data/dataset_formation/`); the bare names are the rock-resolution baselines (`data/dataset/`).

## 1. UMAP latent space, coloured by label scheme

![AE UMAP](01_ae_umap.png)

![JEPA UMAP](02_jepa_umap.png)

![AE_formation UMAP](05_ae_formation_umap.png)

![JEPA_formation UMAP](06_jepa_formation_umap.png)

Each 2×2 panel reuses the same UMAP projection of the encoder's latents, coloured by four different label schemes:

- **rock (middle 50%)** — dominant rock in the central depth window
- **formation (middle 50%)** — dominant formation in the same window
- **formation (deepest)** — formation reached at total depth
- **rock (bottom 25%)** — dominant rock in the deep window

Silhouette score reads: > +0.3 strong, +0.1-0.3 modest, ~0 random.

## 2. Silhouette comparison

![silhouette](03_silhouette_comparison.png)

Per-label-set silhouette:

| label set | AE | JEPA | AE_formation | JEPA_formation |
|---|---|---|---|---|
| formation (deepest) | +0.020 | -0.030 | +0.063 | -0.080 |
| formation (middle 50%) | +0.497 | +0.544 | +0.476 | +0.375 |
| rock (bottom 25%) | +0.027 | +0.012 | +0.091 | -0.068 |
| rock (middle 50%) | +0.627 | +0.703 | +0.574 | +0.579 |

## 3. Autoencoder reconstruction quality

### AE

![AE reconstruction](04_ae_reconstruction.png)

| variable | R² | SmoothL1 | n cells |
|---|---|---|---|
| rhob | 0.461 | 0.2389 | 1,320,000 |
| gr_api | 0.385 | 0.3223 | 1,320,000 |
| dt_us_ft | 0.546 | 0.1558 | 1,320,000 |
| nphi | 0.535 | 0.2091 | 1,320,000 |
| res_deep_log | 0.465 | 0.2329 | 1,320,000 |

### AE_formation

![AE_formation reconstruction](08_ae_formation_reconstruction.png)

| variable | R² | SmoothL1 | n cells |
|---|---|---|---|
| rhob | 0.391 | 0.3238 | 1,320,000 |
| gr_api | -0.005 | 0.5014 | 1,320,000 |
| dt_us_ft | 0.465 | 0.1917 | 1,320,000 |
| nphi | 0.304 | 0.3362 | 1,320,000 |
| res_deep_log | 0.102 | 0.3999 | 1,320,000 |

R² ≈ 1 and SmoothL1 ≪ 1 indicate the autoencoder can reconstruct each channel from the latent. Drops on a specific variable point to information loss in the bottleneck.

## 4. Files

- `silhouette_summary.csv` — silhouette per (encoder × label set)
- `ae_reconstruction.csv` — per-(encoder, variable) R² and SmoothL1
- `0N_<encoder>_umap.png` — per-encoder UMAP grids
- `03_silhouette_comparison.png` — grouped bar chart
- `0N_<encoder>_reconstruction.png` — per-variable truth-vs-recon scatter (AE family only)
