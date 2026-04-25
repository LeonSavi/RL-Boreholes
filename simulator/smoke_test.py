"""
End-to-end smoke test.  Verifies the pipeline wires up correctly without
requiring the real samples.parquet.  Uses a fake DistributionBank with a
handful of rock types so the map generator and autoencoder can be
instantiated and a training step run.

Run:
    python -m simulator.smoke_test
"""
from __future__ import annotations

import numpy as np
from scipy.stats import gaussian_kde
import torch

from .distributions import DistributionBank, CellDistribution, _nearest_psd
from .map_generator import generate_map, SimConfig, MapGenerator
from ..encoder.autoencoder import BoreholeAutoencoder, AEConfig
from .train import boreholes_from_map, compute_standardisation_stats, standardise


def build_fake_bank() -> DistributionBank:
    """Create a DistributionBank with synthetic distributions for the rock
    types the stratigraphy module can produce."""
    variables = [
        "rhob", "gr_api", "dt_us_ft", "nphi", "pef", "res_deep_log",
    ]
    depth_bins = [0, 400, 800, 1200, 1600, 2000, 2400, 2800, 3200, 3600, 4000, 4400, 4800, 5200, 5600, 6000]
    bank = DistributionBank(variables, depth_bins)

    rock_profiles = {
        "sandstone_clean": {"rhob": 2.40, "gr_api": 35,  "dt_us_ft": 80,
                            "nphi": 0.15, "pef": 2.0},
        "sandstone_shaly": {"rhob": 2.50, "gr_api": 80,  "dt_us_ft": 85,
                            "nphi": 0.22, "pef": 2.5},
        "claystone_cool":  {"rhob": 2.50, "gr_api": 65,  "dt_us_ft": 90,
                            "nphi": 0.28, "pef": 3.2},
        "claystone_hot":   {"rhob": 2.55, "gr_api": 120, "dt_us_ft": 95,
                            "nphi": 0.35, "pef": 3.6},
        "claystone":       {"rhob": 2.52, "gr_api": 90,  "dt_us_ft": 92,
                            "nphi": 0.30, "pef": 3.4},
        "clay":            {"rhob": 2.1,  "gr_api": 85,  "dt_us_ft": 140,
                            "nphi": 0.40, "pef": 3.0},
        "chalk":           {"rhob": 2.35, "gr_api": 20,  "dt_us_ft": 100,
                            "nphi": 0.30, "pef": 4.9},
        "halite_pure":     {"rhob": 2.10, "gr_api": 5,   "dt_us_ft": 67,
                            "nphi": 0.0,  "pef": 4.6},
        "halite":          {"rhob": 2.15, "gr_api": 15,  "dt_us_ft": 70,
                            "nphi": 0.05, "pef": 4.7},
        "anhydrite":       {"rhob": 2.95, "gr_api": 10,  "dt_us_ft": 50,
                            "nphi": 0.01, "pef": 5.1},
        "dolomite":        {"rhob": 2.82, "gr_api": 25,  "dt_us_ft": 62,
                            "nphi": 0.10, "pef": 3.1},
        "carbonate":       {"rhob": 2.70, "gr_api": 25,  "dt_us_ft": 65,
                            "nphi": 0.15, "pef": 5.0},
    }
    defaults = {"res_deep_log": 1.5}

    rng = np.random.default_rng(0)
    for rock, prof in rock_profiles.items():
        for bin_idx in range(len(depth_bins) - 1):
            # compaction: density and slowness shift with depth
            depth_centre = 0.5 * (depth_bins[bin_idx] + depth_bins[bin_idx + 1])
            depth_factor = depth_centre / 1500.0
            cell = CellDistribution(
                rock_type=rock,
                depth_lo=float(depth_bins[bin_idx]),
                depth_hi=float(depth_bins[bin_idx + 1]),
                variables=list(bank.variables),
                n_samples=500,
            )
            for v in bank.variables:
                if v in prof:
                    mean = prof[v]
                    if v == "rhob":
                        mean = mean + 0.05 * depth_factor
                    elif v == "dt_us_ft":
                        mean = mean - 5 * depth_factor
                else:
                    mean = defaults.get(v, 0.0)
                std = 0.05 * abs(mean) + 0.02 if mean != 0 else 0.1
                samples = rng.normal(mean, std, size=500)
                cell.kdes[v] = gaussian_kde(samples)
                cell.supports[v] = (float(np.quantile(samples, 0.05)),
                                    float(np.quantile(samples, 0.95)))
                cell.means[v] = float(samples.mean())
                cell.stds[v] = float(samples.std())
            # identity correlation matrix for simplicity
            cell.corr_variables = [v for v in bank.variables if v in cell.kdes]
            n = len(cell.corr_variables)
            cell.corr_matrix = np.eye(n)
            bank.cells[(rock, bin_idx)] = cell
    bank.rock_types = list(rock_profiles.keys())
    return bank


def main():
    print("=" * 60)
    print("smoke test: building fake distribution bank")
    bank = build_fake_bank()
    print(f"  rock_types: {bank.rock_types}")
    print(f"  cells: {len(bank.cells)}")

    print("=" * 60)
    print("generating one map...")
    cfg = SimConfig(n_x=16, n_y=16, n_depth=400, max_depth=4000.0)
    rng = np.random.default_rng(42)
    m = generate_map(bank, cfg, rng)
    print(f"  rock_types shape: {m['rock_types'].shape}")
    print(f"  variables:")
    for v, arr in m["variables"].items():
        finite_frac = np.isfinite(arr).mean()
        print(f"    {v:14s}  shape={arr.shape}  finite={finite_frac:.1%}  "
              f"mean={np.nanmean(arr):.3f}")
    print(f"  yield_field: mean={m['yield_field'].mean():.3f}  "
          f"max={m['yield_field'].max():.3f}")
    print(f"  thickness_field: mean={m['thickness_field'].mean():.3f}  "
          f"max={m['thickness_field'].max():.3f}")
    print(f"  orebodies: {len(m['bodies'])}")

    print("=" * 60)
    print("testing autoencoder forward + backward pass...")
    variables = list(cfg.variables)
    gen = MapGenerator(bank, cfg, seed=1)
    stats = compute_standardisation_stats(gen, variables, n_maps=3)
    boreholes = boreholes_from_map(next(gen), variables)
    boreholes = standardise(boreholes, stats, variables)
    boreholes = np.nan_to_num(boreholes, nan=0.0)
    x = torch.from_numpy(boreholes[:32].astype(np.float32))

    ae_cfg = AEConfig(
        n_variables=len(variables),
        n_depth=cfg.n_depth,
        latent_dim=64,
        channels=(16, 32, 64),
        mask_prob=0.3,
    )
    model = BoreholeAutoencoder(ae_cfg)
    model.train()
    recon, z = model(x)
    loss = torch.nn.functional.mse_loss(recon, x)
    loss.backward()
    print(f"  input shape: {x.shape}")
    print(f"  latent shape: {z.shape}")
    print(f"  recon shape: {recon.shape}")
    print(f"  loss: {loss.item():.4f}")
    print(f"  params: {sum(p.numel() for p in model.parameters()):,}")

    print("=" * 60)
    print("all stages work.  ready for real data.")


if __name__ == "__main__":
    main()