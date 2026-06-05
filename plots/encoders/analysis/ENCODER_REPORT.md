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
| formation (deepest) | +0.051 | +0.030 | +0.078 | -0.085 |
| formation (middle 50%) | +0.477 | +0.495 | +0.471 | +0.393 |
| rock (bottom 25%) | +0.070 | +0.018 | +0.094 | -0.144 |
| rock (middle 50%) | +0.616 | +0.629 | +0.570 | +0.454 |

## 3. Autoencoder reconstruction quality

### AE

![AE reconstruction](04_ae_reconstruction.png)

| variable | R² | SmoothL1 | n cells |
|---|---|---|---|
| rhob | 0.402 | 0.2570 | 1,320,000 |
| gr_api | 0.190 | 0.4518 | 1,320,000 |
| dt_us_ft | 0.488 | 0.1425 | 1,320,000 |
| nphi | 0.453 | 0.2624 | 1,320,000 |
| res_deep_log | 0.247 | 0.3246 | 1,320,000 |

### AE_formation

![AE_formation reconstruction](08_ae_formation_reconstruction.png)

| variable | R² | SmoothL1 | n cells |
|---|---|---|---|
| rhob | 0.385 | 0.3319 | 1,320,000 |
| gr_api | 0.009 | 0.4949 | 1,320,000 |
| dt_us_ft | 0.438 | 0.1987 | 1,320,000 |
| nphi | 0.320 | 0.3352 | 1,320,000 |
| res_deep_log | 0.135 | 0.3979 | 1,320,000 |

R² ≈ 1 and SmoothL1 ≪ 1 indicate the autoencoder can reconstruct each channel from the latent. Drops on a specific variable point to information loss in the bottleneck.

## 4. Files

- `silhouette_summary.csv` — silhouette per (encoder × label set)
- `ae_reconstruction.csv` — per-(encoder, variable) R² and SmoothL1
- `0N_<encoder>_umap.png` — per-encoder UMAP grids
- `03_silhouette_comparison.png` — grouped bar chart
- `0N_<encoder>_reconstruction.png` — per-variable truth-vs-recon scatter (AE family only)
