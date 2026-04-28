"""
End-to-end smoke test. Verifies the pipeline wires up correctly without
requiring the real samples.parquet. Uses a fake DistributionBank,
DiscoveryPrior, and FormationGeometry so the map generator and
autoencoder can be instantiated and a training step run.

Run:
    python -m simulator.smoke_test
"""
from __future__ import annotations

import numpy as np
from scipy.stats import gaussian_kde
import torch

from simulator.distributions import (
    DistributionBank, CellDistribution, DiscoveryPrior, _nearest_psd,
)
from simulator.formation_geometry import (
    FormationGeometry, FormationStats, FORMATION_ORDER,
)
from simulator.map_generator import generate_map, SimConfig, MapGenerator
from encoder.autoencoder import BoreholeAutoencoder, AEConfig
from train_encoder import boreholes_from_map, compute_standardisation_stats, standardise


def build_fake_bank() -> DistributionBank:
    """DistributionBank with synthetic distributions for the rock types
    the fake formation geometry can produce."""
    variables = [
        "rhob", "gr_api", "dt_us_ft", "nphi", "pef", "res_deep_log",
    ]
    depth_bins = [0, 400, 800, 1200, 1600, 2000, 2400, 2800, 3200, 3600,
                  4000, 4400, 4800, 5200, 5600, 6000]
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
        "anhydrite":       {"rhob": 2.95, "gr_api": 10,  "dt_us_ft": 50,
                            "nphi": 0.01, "pef": 5.1},
        "dolomite":        {"rhob": 2.82, "gr_api": 25,  "dt_us_ft": 62,
                            "nphi": 0.10, "pef": 3.1},
        "other":           {"rhob": 2.50, "gr_api": 50,  "dt_us_ft": 90,
                            "nphi": 0.20, "pef": 3.0},
    }
    defaults = {"res_deep_log": 1.5}

    rng = np.random.default_rng(0)
    for rock, prof in rock_profiles.items():
        for bin_idx in range(len(depth_bins) - 1):
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
            cell.corr_variables = [v for v in bank.variables if v in cell.kdes]
            n = len(cell.corr_variables)
            cell.corr_matrix = np.eye(n)
            bank.cells[(rock, bin_idx)] = cell
    bank.rock_types = list(rock_profiles.keys())
    return bank


def build_fake_geometry() -> FormationGeometry:
    """FormationGeometry whose empirical distributions match the actual
    NLOG numbers from formation_depth_stats.py output (rough, for
    smoke-test purposes only). Each formation gets ~50 synthetic per-well
    top depths and thicknesses so the KDEs have something to fit on."""
    geom = FormationGeometry(list(FORMATION_ORDER))
    geom.n_wells_total = 1500

    # (formation, prevalence, top_median, top_std, thick_median, thick_std,
    #  facies_dict)
    fake_stats = [
        ("NU", 0.46, 144,  150, 414, 200, {"clay": 1.0}),
        ("NM", 0.39, 671,  300,  60,  40, {"clay": 1.0}),
        ("NL", 0.47, 721,  300, 393, 200, {"clay": 1.0}),
        ("CK", 0.68, 1138, 350, 662, 300, {"chalk": 1.0}),
        ("KN", 0.74, 1785, 600, 227, 200, {"claystone": 0.46,
                                            "claystone_cool": 0.41,
                                            "sandstone_shaly": 0.13}),
        ("SL", 0.21, 1950, 500, 180, 150, {"claystone_cool": 0.61,
                                            "claystone": 0.39}),
        ("SG", 0.08, 2476, 400, 149, 100, {"claystone": 0.69,
                                            "claystone_hot": 0.31}),
        ("AT", 0.21, 1945, 500, 199, 200, {"claystone_hot": 0.66,
                                            "claystone": 0.20,
                                            "claystone_cool": 0.14}),
        ("RN", 0.31, 2200, 500, 262, 200, {"claystone": 0.54,
                                            "dolomite": 0.40,
                                            "anhydrite": 0.06}),
        ("RB", 0.52, 2293, 600, 267, 200, {"claystone_hot": 0.67,
                                            "sandstone_shaly": 0.33}),
        ("ZE", 0.60, 2475, 500, 326, 250, {"halite_pure": 0.43,
                                            "anhydrite": 0.28,
                                            "other": 0.22,
                                            "dolomite": 0.07}),
        ("RO", 0.51, 3274, 500, 224, 200, {"sandstone_clean": 0.56,
                                            "claystone_hot": 0.44}),
        ("DC", 0.41, 3466, 600,  63, 100, {"claystone_hot": 0.59,
                                            "claystone": 0.41}),
    ]

    rng = np.random.default_rng(0)
    for (fm, prev, top_med, top_std, thick_med, thick_std, facies) in fake_stats:
        n_synthetic = 50
        tops = rng.normal(top_med, top_std, size=n_synthetic).clip(min=10.0)
        thicks = rng.normal(thick_med, thick_std,
                              size=n_synthetic).clip(min=20.0)
        geom.formations[fm] = FormationStats(
            name=fm,
            n_wells=int(prev * 1500),
            prevalence=prev,
            top_depths=tops,
            thicknesses=thicks,
            facies=facies,
        )
    return geom


def build_fake_prior(bank: DistributionBank) -> DiscoveryPrior:
    """Concentrate prior probability on reservoir / seal / source rocks
    at Rotliegend depths, so we can verify rock-conditioned ore placement."""
    rocks = list(bank.rock_types)
    n_bins = len(bank.depth_bins) - 1
    prob = np.full((len(rocks), n_bins), 1.0 / len(rocks))
    boost_bins = [6, 7, 8, 9, 10]  # 2400-4400m
    boost_rocks = ["sandstone_clean", "claystone_hot", "halite_pure"]
    for bi in boost_bins:
        for r in boost_rocks:
            if r in rocks:
                prob[rocks.index(r), bi] = 0.3
        prob[:, bi] /= prob[:, bi].sum()
    return DiscoveryPrior(
        rock_types=rocks,
        depth_bins=list(bank.depth_bins),
        prob=prob,
        n_positive_wells=300,
        n_positive_rows=600_000,
    )


def main():
    print("=" * 60)
    print("smoke test: building fake bank + prior + geometry")
    bank = build_fake_bank()
    prior = build_fake_prior(bank)
    geom = build_fake_geometry()
    print(f"  rock_types : {bank.rock_types}")
    print(f"  cells      : {len(bank.cells)}")
    print(f"  formations : {list(geom.formations.keys())}")

    print("=" * 60)
    print("generating one map (with prior + geometry)...")
    cfg = SimConfig(n_x=16, n_y=16, n_depth=440, max_depth=4400.0)
    rng = np.random.default_rng(42)
    m = generate_map(bank, geom, cfg, rng, prior=prior)
    print(f"  rock_types shape: {m['rock_types'].shape}")

    # quick column inspection — verify formation depth ordering
    print("\n  column at (8, 8):")
    cols = m["formations"][8, 8, :]
    rocks = m["rock_types"][8, 8, :]
    da = m["depth_axis"]
    last = None
    for d, f, r in zip(da, cols, rocks):
        if f != last:
            print(f"    {d:>5.0f}m  {f:<6s} {r}")
            last = f

    print("\n  variables:")
    for v, arr in m["variables"].items():
        finite_frac = np.isfinite(arr).mean()
        print(f"    {v:14s}  finite={finite_frac:.1%}  "
              f"mean={np.nanmean(arr):.3f}")
    print(f"\n  yield_field   : shape={m['yield_field'].shape}  "
          f"max={m['yield_field'].max():.3f}  "
          f"nnz={(m['yield_field'] > 0).sum()}")
    print(f"  orebodies     : {len(m['bodies'])}")
    for i, b in enumerate(m["bodies"]):
        print(f"    body {i}: centre=({b.center_x:.1f}, {b.center_y:.1f}, "
              f"{b.center_z:.0f}m)  yield_peak={b.peak_yield:.2f}  "
              f"radii=({b.radius_x:.1f}, {b.radius_y:.1f}, {b.radius_z:.0f}m)")

    print("=" * 60)
    print("testing autoencoder forward + backward pass...")
    variables = list(cfg.variables)
    gen = MapGenerator(bank, geom, cfg, seed=1, prior=prior)
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
    print(f"  input shape : {x.shape}")
    print(f"  latent shape: {z.shape}")
    print(f"  recon shape : {recon.shape}")
    print(f"  loss        : {loss.item():.4f}")
    print(f"  params      : {sum(p.numel() for p in model.parameters()):,}")

    print("=" * 60)
    print("all stages work. ready for real data.")


if __name__ == "__main__":
    main()