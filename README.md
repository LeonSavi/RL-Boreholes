# RL-Boreholes

Synthetic Dutch subsurface simulator + a frozen encoder for boreholes.
Built on NLOG and LILY data. Intended use: an RL agent that picks
where to drill next, given partial well-log observations.

## Quick start

Artifacts are already fitted in `data/clean/` and `checkpoints/`. To
generate maps and encode boreholes:

```python
from simulator import (
    DistributionBank, DiscoveryPrior, FormationGeometry,
    MapGenerator, SimConfig,
)

bank  = DistributionBank.load("data/clean/distributions.pkl")
prior = DiscoveryPrior.load("data/clean/discovery_prior.pkl")
geom  = FormationGeometry.load("data/clean/formation_geometry.pkl")

gen = MapGenerator(bank, geom, SimConfig(), seed=42, prior=prior)
m = next(gen)   # one map
```

Each map is a dict with these keys:

- `rock_types`, `formations`: `(32, 32, 440)` object arrays
- `variables`: dict of 6 logs, each `(32, 32, 440)` float32
  (rhob, gr_api, dt_us_ft, nphi, pef, res_deep_log)
- `yield_field`: `(32, 32, 440)` float32, ground-truth ore
- `bodies`: list of `OreBody` (0-3 stratabound deposits)
- `depth_axis`: `(440,)` depths in metres

A cell is 100m × 100m × 10m, so the map is 3.2 × 3.2 × 4.4 km.

## Using it for RL

The natural framing is one map = one episode. Per step:

1. Agent picks a cell `(x, y)` to drill.
2. Env returns the borehole at that cell:
   `np.stack([m["variables"][v][x, y, :] for v in vars])` of shape `(6, 440)`.
3. Reward = `m["yield_field"][x, y, :].sum()` (or whatever you define).
4. Done after N drills.

The borehole is the agent's observation. Feeding the raw `(6, 440)`
to a policy is wasteful — use the frozen encoder to compress it to a
128-dim latent first:

```python
import torch
from encoder.jepa import JEPAModel, JEPAConfig

cfg = JEPAConfig(n_variables=6, n_depth=440, latent_dim=128)
model = JEPAModel(cfg)
model.load_state_dict(torch.load("checkpoints/jepa.pt"))
model.eval()

with torch.no_grad():
    z = model.encode(borehole_standardised)   # (B, 128)
```

The encoder expects standardised inputs. Stats are at
`checkpoints/standardisation_stats.json` — apply them before encoding.

JEPA outperforms the autoencoder (`checkpoints/ae.pt`) on rock-type
clustering (silhouette +0.357 vs +0.260), so prefer JEPA unless you
have a reason not to.

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

Use these for reward shaping or evaluation if you want.

## What you can tune

`SimConfig` has the usual knobs:

```python
SimConfig(
    n_x=32, n_y=32, n_depth=440, max_depth=4400.0,
    facies_persistence=0.95,
    spatial_correlation_strength=0.7,
    ore_depth_window=(1600.0, 4400.0),
    ore_yield_peak_range=(0.5, 5.0),
    # ... see simulator/map_generator.py for all fields
)
```

Don't change `n_depth` or `max_depth` — the encoder is locked to
`(6, 440)` at 4400m max.

## Performance

Map generation is CPU-bound at roughly 2s/map. If you need a lot of
maps for training, generate in a background process or pool — they're
independent. Encoder inference is fast on either CPU or GPU.

## Re-fitting from scratch

If the underlying data changes:

```bash
pip install -r requirements.txt

python scraper.py        # NLOG
# Then download LILY .csv files into data/lily/ from
# https://zenodo.org/records/10425539
# (KAPPA, MAD, NGR, PWC, TCON _DataLITH)

python pull_data.py            # produces data/clean/samples.parquet
python save_distributions.py   # fits all three .pkl files (~5 min)
python train_encoder.py        # autoencoder, overnight
python train_jepa.py           # JEPA, overnight
```

## References

- NLOG: Dutch subsurface borehole repository
- LILY: Childress et al. 2024, Zenodo 10425539
- Architecture inspired by Mern & Caers (2023), GMD