# RL-Boreholes

Synthetic Dutch subsurface simulator + frozen encoders for borehole
observations. Built on NLOG (Dutch onshore + offshore wells) and LILY
(global IODP ocean drilling). Intended use: an RL agent that picks
where to drill next, given partial well-log observations.

## Quick start

Artifacts are already fitted in `data/clean/` and `checkpoints/`. To
generate maps and encode boreholes:

```python
from simulator.distributions import DistributionBank, DiscoveryPrior
from simulator.formation_geometry import FormationGeometry
from simulator.map_generator import MapGenerator, SimConfig

bank  = DistributionBank.load("data/clean/distributions.pkl")
prior = DiscoveryPrior.load("data/clean/discovery_prior.pkl")
geom  = FormationGeometry.load("data/clean/formation_geometry.pkl")

gen = MapGenerator(bank, geom, SimConfig(), seed=2, prior=prior)
m   = next(gen)   # one map
```

Each map is a dict with these keys:

| key | type | shape | what |
|---|---|---|---|
| `rock_types`   | object ndarray | `(32, 32, 440)` | `rock_type_fine` label per cell |
| `formations`   | object ndarray | `(32, 32, 440)` | stratigraphic formation per cell |
| `variables`    | dict of 5 ndarrays | each `(32, 32, 440)` | `rhob, gr_api, dt_us_ft, nphi, res_deep_log` |
| `yield_field`  | float32 ndarray | `(32, 32, 440)` | ground-truth ore yield |
| `bodies`       | list[`OreBody`] | — | 0–3 stratabound deposits (empty when no ore is placed) |
| `depth_axis`   | float64 ndarray | `(440,)` | depths in metres |
| `config`       | dict | — | the `SimConfig` used |

A cell is 100 m × 100 m × 10 m, so the map is 3.2 × 3.2 × 4.4 km.

**Note:** `pef` (photoelectric factor) used to be in this list but was
dropped — it was logged in only ~6% of NLOG wells, so the encoder was
spending capacity on a near-constant zero channel.

## Pipeline

The repo follows a numbered-script convention so the order to run is
obvious:

```
1_scraper.py                       scrape NLOG well metadata + LAS files
2_pull_data.py                     clean + merge → data/clean/samples.parquet
scripts/reclassify_coarse_rocks.py reclassify residual coarse rows (see below)
3_save_distributions.py            fit DistributionBank / FormationGeometry / DiscoveryPrior
4_pull_maps.py                     pre-generate the training dataset (data/dataset/)
5a_train_jepa.py                   train the JEPA encoder
5b_train_encoder.py                train the autoencoder
```

### Coarse-row reclassification

`2_pull_data.py` maps NLOG `strat_unit` codes to fine
rock classes via longest-prefix matching, but some codes (e.g. a
generic `KNNC`, or any Zechstein row labelled only `ZE`) are too
generic to assign. Those rows land in the coarse buckets
`claystone`, `sandstone`, `other`, and `clay`. `scripts/reclassify_coarse_rocks.py` then runs a
single per-formation data-driven match: it builds wireline
templates from rows that already have a fine label, opens an
adaptive depth window around the coarse row (start at the row's
own 10 m bin, expand until ≥ 50 rows are collected), and assigns
the row to the closest fine class if the mean per-channel z-score
is ≤ 2.0 and the row has ≥ 2 wireline channels.

There is **no off-formation fallback** and **no geology default** —
a row that cannot be matched is dropped from the bank.
`clay` is the one exception: it is protected (Quaternary
unconsolidated; distinct petrophysics from lithified claystone)
and never reassigned. On the ~1.02 M coarse rows, 198 k are
reassigned, 609 k clay rows pass through, and ~213 k are dropped.
The bank ends up with 9 fine rock classes (the coarse classes
`claystone`, `sandstone`, `other` are eliminated).

Plus three analysis scripts that produce thesis-grade plots and reports:

```
analysis_data.py             EDA on samples.parquet (data census, coverage, design justification)
analysis_simulation.py       sim-vs-real comparison (uses saved maps if labels present)
analysis_encoders.py         AE-vs-JEPA evaluation (silhouettes + reconstruction quality)
```

Outputs land in `plots/<area>/` as a markdown report plus PNGs + CSVs:

```
plots/analysis/EDA_REPORT.md            <- analysis_data.py
plots/simulation/SIMULATION_REPORT.md   <- analysis_simulation.py
plots/encoders/analysis/ENCODER_REPORT.md  <- analysis_encoders.py
```

## Pre-generated dataset

`4_pull_maps.py` writes a dataset to `data/dataset/` (default 10 000
maps, 16 workers, ~2–3 h):

```
data/dataset/
  stats.pkl                    <- standardisation stats (mean, std) per variable
  config.pkl                   <- {variables, n_maps, seed}
  labels_vocab.pkl             <- {"rocks": {str→int}, "formations": {str→int}}
  boreholes_00000.npy          <- (1024, 5, 440) float16, standardised values
  boreholes_00001.npy
  ...
  labels_00000.npz             <- {"rocks": (1024, 440) int8, "formations": same}
  labels_00001.npz
  ...
```

`5a_train_jepa.py` and `5b_train_encoder.py` consume `boreholes_*.npy`
(labels aren't needed for training). `analysis_simulation.py` consumes
the labels to compare per-rock distributions sim-vs-real.

The script is resume-aware: stopping and restarting picks up where it
left off.

## Using the trained encoders

```python
import torch
from encoder.jepa_encoder import load_jepa_checkpoint
from encoder.autoencoder    import load_checkpoint, standardise

# JEPA
jepa, stats, variables = load_jepa_checkpoint("checkpoints/jepa.pt")
jepa.eval()

# Autoencoder
ae, stats, variables = load_checkpoint("checkpoints/ae.pt")
ae.eval()
```

Both encoders expect inputs **standardised by `stats`** before
forwarding:

```python
from encoder.autoencoder import standardise
import numpy as np

# borehole: (B, V=5, D=440) raw values in physical units
x = standardise(borehole, stats, variables)
x = np.nan_to_num(x, nan=0.0).astype(np.float32)
x = torch.from_numpy(x)

with torch.no_grad():
    # JEPA: target encoder, mean-pooled tokens
    z_jepa = jepa.target_encoder(x).mean(dim=1)    # (B, 128)
    # AE: encoder output
    z_ae   = ae.encoder(x)                          # (B, 128)
```

`stats` is a `{variable: (mean, std)}` dict saved inside each
checkpoint, so you don't need a separate standardisation file.

## Using it for RL

The natural framing is one map = one episode. Per step:

1. Agent picks a cell `(x, y)` to drill.
2. Env returns the borehole at that cell:
   `np.stack([m["variables"][v][x, y, :] for v in vars])` → shape `(5, 440)`.
3. Reward = `m["yield_field"][x, y, :].sum()` (or whatever you define).
4. Done after N drills.

The borehole is the agent's observation. Feeding the raw `(5, 440)` to
a policy is wasteful — encode it to a 128-dim latent first.

## Ground truth: ore bodies

Each `OreBody` in `m["bodies"]` is stratabound — its vertical extent
follows the host rock layer in each column, with lateral Gaussian
falloff and internal grade heterogeneity. Fields:

```
center_x, center_y, center_z   # body centre (cells, cells, metres)
host_rock                      # rock type the body lives in
z_top, z_bot                   # host-rock interval at (cx, cy)
radius_x, radius_y             # lateral extent in cells
peak_yield, orientation
```

Use these for reward shaping or evaluation.

## What you can tune

`SimConfig` has the usual knobs (see `simulator/map_generator.py`):

```python
SimConfig(
    n_x=32, n_y=32, n_depth=440, max_depth=4400.0,
    lateral_correlation_cells=2.0,
    vertical_correlation_cells=3.0,
    spatial_correlation_strength=0.7,
    facies_persistence=0.95,
    # layer-boundary perturbation (anisotropic GRF, gstools)
    layer_perturbation_std=10.0,
    layer_perturbation_range_cells=8.0,
    grf_method="gaussian_filter",
    # ore-body placement
    ore_depth_window=(1600.0, 4400.0),
    ore_yield_peak_range=(0.5, 5.0),
)
```

Don't change `n_depth` or `max_depth` — the encoder is locked to
`(5, 440)` at 4400 m max.

## Design decisions (and where they're justified)

- **5 variables, not 6** — `pef` dropped because only 6% of NLOG wells
  logged it. See `plots/analysis/10_variable_availability.png` and the
  EDA report.
- **Fine rock classes** (`claystone_hot/cool`, `sandstone_clean/shaly`)
  reduce within-formation IQR by 30–48 %. See
  `plots/analysis/09_fine_vs_coarse_rhob.png`.
- **k-means basin stratification** on `(x_rd, y_rd)` corrects
  geographic over-representation. The map at
  `plots/analysis/03_nlog_well_map.png` shows it.
- **Markov transition matrices** per formation (Laplace-smoothed,
  run-length-calibrated to `max_p_self=0.998`) generate realistic
  vertical sequences. Sim-vs-real Frobenius distances are in
  `plots/simulation/per_formation_transition_distance.csv`.
- **Anisotropic Gaussian random field** for layer-boundary perturbation
  replaces sinusoidal wiggle. Variogram fit in
  `plots/simulation/08_lateral_coherence.png`.
- **Empirical [p005, p995] clipping bounds** per `(rock × variable)`
  cell — see `plots/analysis/11_support_bounds.png`.

## Re-fitting from scratch

If the underlying data changes:

```bash
pip install -r requirements.txt

python 1_scraper.py            # NLOG: scrape per-well JSON + LAS
# Then download LILY .csv files into data/lily/ from
#   https://zenodo.org/records/10425539
# (KAPPA, MAD, NGR, PWC, TCON _DataLITH)

python 2_pull_data.py          # → data/clean/samples.parquet
python 3_save_distributions.py # → the three .pkl files (~5 min)
python analysis_data.py        # EDA report (optional but useful)

python 4_pull_maps.py          # 10 000-map dataset (~2–3 h, 16 workers)

python 5a_train_jepa.py        # JEPA encoder (~1–3 h with early stop)
python 5b_train_encoder.py     # autoencoder (~1–3 h with early stop)

python analysis_simulation.py  # sim-vs-real comparison
python analysis_encoders.py    # AE-vs-JEPA evaluation
```

All training entry points default to `--dataset-dir data/dataset`,
`--steps 200000`, `--patience 40`, `--min-maps-warmup 3500`, so plain
`python 5a_train_jepa.py` does the right thing.

## Performance

- Map generation: ~1.2 maps/s with 16 workers
- Dataset on disk: ~5 MB per map (float16) + ~450 KB labels = ~55 GB
  for 10 000 maps
- Encoder inference: <10 ms per batch of 128 on GPU

## References

- NLOG: Dutch subsurface borehole repository
- LILY: Childress et al. 2024, Zenodo 10425539
- Architecture inspired by Mern & Caers (2023), *GMD*
