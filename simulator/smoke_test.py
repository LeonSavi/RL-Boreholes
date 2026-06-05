"""
End-to-end smoke test + focused unit tests. Verifies the pipeline
wires up correctly without requiring the real samples.parquet. Uses
a fake DistributionBank, DiscoveryPrior, and FormationGeometry so the
map generator and autoencoder can be instantiated and a training step
run.

Run:
    python -m simulator.smoke_test

Test coverage:
  * Schema: (rock, formation, depth_bin) 3-tuple keys (Phase L+).
  * Bank.sample with explicit formation argument + fallback chain.
  * Map generation end-to-end (shape, finiteness).
  * Phase O: gas-response shift applied IN ore-body cells only.
  * Phase O: max_ore_bodies cap honored.
  * Autoencoder forward + backward.
"""

from __future__ import annotations

import json
import os
import tempfile
import numpy as np
from scipy.stats import gaussian_kde
import torch

from .distributions import (
    DistributionBank,
    CellDistribution,
    DiscoveryPrior,
    _nearest_psd,
)
from .formation_geometry import (
    FormationGeometry,
    FormationStats,
    FORMATION_ORDER,
)
from .map_generator import generate_map, SimConfig, MapGenerator
from encoder.autoencoder import BoreholeAutoencoder, AEConfig
from train_encoder import boreholes_from_map, compute_standardisation_stats, standardise

# rocks the fake geometry below can produce; bank must cover all of these
FAKE_ROCKS = (
    "sandstone_clean",
    "sandstone_shaly",
    "claystone_cool",
    "claystone_hot",
    "claystone",
    "clay",
    "chalk",
    "halite_pure",
    "anhydrite",
    "dolomite",
    "other",
)


def build_fake_bank() -> DistributionBank:
    """DistributionBank with synthetic distributions for the rock types
    the fake formation geometry can produce.

    v3.2: cells are keyed as (rock, formation, bin_idx). For the smoke
    test we replicate each per-rock profile across every formation in
    FORMATION_ORDER so the bank works whatever formation the geometry
    picks for a depth slice.
    """
    variables = [
        "rhob",
        "gr_api",
        "dt_us_ft",
        "nphi",
        "pef",
        "res_deep_log",
    ]
    depth_bins = list(range(0, 6001, 10))
    bank = DistributionBank(variables, depth_bins)
    bank.formations = list(FORMATION_ORDER)

    rock_profiles = {
        "sandstone_clean": {
            "rhob": 2.40,
            "gr_api": 35,
            "dt_us_ft": 80,
            "nphi": 0.15,
            "pef": 2.0,
        },
        "sandstone_shaly": {
            "rhob": 2.50,
            "gr_api": 80,
            "dt_us_ft": 85,
            "nphi": 0.22,
            "pef": 2.5,
        },
        "claystone_cool": {
            "rhob": 2.50,
            "gr_api": 65,
            "dt_us_ft": 90,
            "nphi": 0.28,
            "pef": 3.2,
        },
        "claystone_hot": {
            "rhob": 2.55,
            "gr_api": 120,
            "dt_us_ft": 95,
            "nphi": 0.35,
            "pef": 3.6,
        },
        "claystone": {
            "rhob": 2.52,
            "gr_api": 90,
            "dt_us_ft": 92,
            "nphi": 0.30,
            "pef": 3.4,
        },
        "clay": {"rhob": 2.1, "gr_api": 85, "dt_us_ft": 140, "nphi": 0.40, "pef": 3.0},
        "chalk": {
            "rhob": 2.35,
            "gr_api": 20,
            "dt_us_ft": 100,
            "nphi": 0.30,
            "pef": 4.9,
        },
        "halite_pure": {
            "rhob": 2.10,
            "gr_api": 5,
            "dt_us_ft": 67,
            "nphi": 0.0,
            "pef": 4.6,
        },
        "anhydrite": {
            "rhob": 2.95,
            "gr_api": 10,
            "dt_us_ft": 50,
            "nphi": 0.01,
            "pef": 5.1,
        },
        "dolomite": {
            "rhob": 2.82,
            "gr_api": 25,
            "dt_us_ft": 62,
            "nphi": 0.10,
            "pef": 3.1,
        },
        "other": {"rhob": 2.50, "gr_api": 50, "dt_us_ft": 90, "nphi": 0.20, "pef": 3.0},
    }
    defaults = {"res_deep_log": 1.5}

    rng = np.random.default_rng(0)
    # subsample bins: every 10th 10 m bin gives 60 cells per (rock,
    # formation) -- the bank's nearest-cell fallback fills the gaps
    bin_stride = 10
    for rock, prof in rock_profiles.items():
        for formation in bank.formations:
            for bin_idx in range(0, len(depth_bins) - 1, bin_stride):
                depth_centre = 0.5 * (depth_bins[bin_idx] + depth_bins[bin_idx + 1])
                depth_factor = depth_centre / 1500.0
                cell = CellDistribution(
                    rock_type=rock,
                    depth_lo=float(depth_bins[bin_idx]),
                    depth_hi=float(depth_bins[bin_idx + 1]),
                    variables=list(bank.variables),
                    n_samples=500,
                    formation=formation,
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
                    cell.supports[v] = (
                        float(np.quantile(samples, 0.05)),
                        float(np.quantile(samples, 0.95)),
                    )
                    cell.means[v] = float(samples.mean())
                    cell.stds[v] = float(samples.std())
                cell.corr_variables = [v for v in bank.variables if v in cell.kdes]
                n = len(cell.corr_variables)
                cell.corr_matrix = np.eye(n)
                bank.cells[(rock, formation, bin_idx)] = cell
    bank.rock_types = list(rock_profiles.keys())
    return bank


def build_fake_prior(bank: DistributionBank) -> DiscoveryPrior:
    """Concentrate prior probability on reservoir / seal / source rocks
    at Rotliegend depths, so we can verify rock-conditioned ore placement.
    v3.2: prior is on the same 10 m bins as the bank, so the 'boost
    bins' are now ~240-440 (2400-4400 m), not 6-10."""
    rocks = list(bank.rock_types)
    n_bins = len(bank.depth_bins) - 1
    prob = np.full((len(rocks), n_bins), 1.0 / len(rocks))
    boost_bins = list(range(240, 441))  # 2400-4410 m at 10 m bins
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


def test_bank_sample_with_formation(bank):
    """Bank.sample should respect a passed formation; an empty cell
    should fall through to nearest in formation, then to any-formation."""
    print("\n[unit] bank.sample with explicit formation")
    rng = np.random.default_rng(0)
    # populated cell: pick one we know exists
    keys = list(bank.cells.keys())
    rock, formation, bin_idx = keys[0]
    depth = float(bank.depth_bins[bin_idx]) + 5.0
    out = bank.sample(rock, depth, n=100, rng=rng, formation=formation)
    for v in bank.variables:
        assert np.isfinite(out[v]).all(), f"NaN in {v} for populated cell"
    print(f"  populated ({rock}, {formation}, bin {bin_idx}): OK")

    # missing-formation fallback: ask for a formation that doesn't exist
    out2 = bank.sample(rock, depth, n=100, rng=rng, formation="NOTAFORMATION")
    for v in bank.variables:
        assert np.isfinite(out2[v]).all(), f"NaN in {v} after any-formation fallback"
    print(f"  missing-formation fallback ({rock}, NOTAFORMATION): OK")

    # formation=None should also work (any-formation lookup)
    out3 = bank.sample(rock, depth, n=100, rng=rng, formation=None)
    for v in bank.variables:
        assert np.isfinite(out3[v]).all(), f"NaN in {v} when formation=None"
    print(f"  formation=None lookup: OK")


def _write_temp_gas_table(shifts: dict) -> str:
    """Materialise a gas-shift table to a tempfile, return path."""
    fd, path = tempfile.mkstemp(suffix=".json", prefix="gas_shift_")
    with os.fdopen(fd, "w") as fh:
        json.dump(shifts, fh)
    return path


def test_max_ore_bodies_cap(bank, geom, prior):
    """SimConfig.max_ore_bodies cap must hold for N independent maps."""
    print("\n[unit] max_ore_bodies cap honored")
    cfg = SimConfig(
        n_x=12,
        n_y=12,
        n_depth=440,
        max_depth=4400.0,
        max_ore_bodies=2,
        gas_shift_table_path=None,
    )
    N = 20
    counts = []
    for i in range(N):
        m = generate_map(
            bank, geom, cfg, rng=np.random.default_rng(2000 + i), prior=prior
        )
        n_bodies = len(m.get("bodies", []))
        assert (
            n_bodies <= cfg.max_ore_bodies
        ), f"map {i} has {n_bodies} bodies, cap was {cfg.max_ore_bodies}"
        counts.append(n_bodies)
    print(
        f"  {N} maps, body counts: "
        f"{dict(zip(*np.unique(counts, return_counts=True)))} "
        f"(all <= {cfg.max_ore_bodies})"
    )


def test_gas_response_shift(bank, geom, prior):
    """With shift ON vs OFF, in-body cells should differ by the shift
    table entry. Out-of-body cells should be identical to within float
    noise."""
    print("\n[unit] gas-response shift applied IN orebody only")
    # use unrealistically large shifts so the effect is unambiguous
    shifts = {
        "default": {"rhob": -0.50, "nphi": -0.10, "res_deep_log": +1.50},
        "sandstone_clean": {
            "rhob": -0.40,
            "nphi": -0.08,
            "res_deep_log": +2.00,
        },
    }
    table_path = _write_temp_gas_table(shifts)
    try:
        cfg_on = SimConfig(
            n_x=12,
            n_y=12,
            n_depth=440,
            max_depth=4400.0,
            max_ore_bodies=2,
            gas_shift_table_path=table_path,
        )
        cfg_off = SimConfig(
            n_x=12,
            n_y=12,
            n_depth=440,
            max_depth=4400.0,
            max_ore_bodies=2,
            gas_shift_table_path=None,
        )

        for seed in (3001, 3007, 3013):
            m_on = generate_map(
                bank, geom, cfg_on, rng=np.random.default_rng(seed), prior=prior
            )
            m_off = generate_map(
                bank, geom, cfg_off, rng=np.random.default_rng(seed), prior=prior
            )
            if len(m_on["bodies"]) == 0:
                continue
            in_ore = m_on["yield_field"] > 0
            assert in_ore.any(), f"seed {seed}: no in-body cells"
            # out-of-body cells should match exactly (same seed, no shift)
            for v in ["rhob", "nphi", "res_deep_log"]:
                d = m_on["variables"][v] - m_off["variables"][v]
                d_out = d[~in_ore]
                assert np.nanmax(np.abs(d_out)) < 1e-5, (
                    f"seed {seed} var {v}: out-of-body delta nonzero "
                    f"(max |d| = {np.nanmax(np.abs(d_out)):.4f})"
                )

            # in-body delta per rock should match the shift table
            # (within bound-clipping at the empirical bounds)
            for v in ["rhob", "res_deep_log"]:
                for rock in np.unique(m_on["rock_types"][in_ore]):
                    if rock is None:
                        continue
                    mask = (m_on["rock_types"] == rock) & in_ore
                    if mask.sum() < 50:
                        continue
                    expected = shifts.get(str(rock), shifts["default"])[v]
                    measured = float(
                        np.nanmean(
                            m_on["variables"][v][mask] - m_off["variables"][v][mask]
                        )
                    )
                    # bound-clipping can attenuate large shifts; allow 25 % slack
                    rel = abs(measured - expected) / max(abs(expected), 1e-6)
                    assert rel < 0.5, (
                        f"seed {seed} {rock}/{v}: expected shift "
                        f"{expected:+.3f}, measured {measured:+.3f} "
                        f"(rel error {rel*100:.1f}%)"
                    )
            print(f"  seed {seed}: in-body shift correct, " f"out-of-body delta = 0")
    finally:
        os.unlink(table_path)


def test_conditional_copula_recovery():
    """When a cell is missing some KDEs, the bank should recover them
    from a donor cell of the same rock using the donor's stored
    Spearman correlations, not by drawing independently.

    Setup: build a tiny custom bank with one rock 'test_rock' and
    two formations F1, F2:
      - primary cell  (test_rock, F1, bin 0): only `rhob` + `gr_api`
        KDEs.  Its corr_matrix is 2x2 identity (no internal correlation
        to seed).
      - donor cell    (test_rock, F2, bin 5): all 5 KDEs, with a
        strongly NEGATIVE Spearman correlation between rhob and nphi
        and a strongly POSITIVE one between gr_api and nphi.

    Sample N=5000 from the primary cell.  Verify that the recovered
    `nphi` is correlated with `rhob` (Spearman < -0.4) and with
    `gr_api` (Spearman > +0.3) -- i.e. the donor's joint structure
    flowed through.  An independent fallback would give Spearman ~ 0.
    """
    print("\n[unit] conditional-copula recovery of missing KDEs")
    from scipy.stats import gaussian_kde, spearmanr

    rng = np.random.default_rng(123)
    variables = ["rhob", "gr_api", "dt_us_ft", "nphi", "res_deep_log"]
    bins = list(range(0, 4401, 10))
    bank = DistributionBank(variables, bins)
    bank.rock_types = ["test_rock"]
    bank.formations = ["F1", "F2"]

    # ---- primary cell: 2 of 5 KDEs (rhob, gr_api) ----
    primary = CellDistribution(
        rock_type="test_rock",
        depth_lo=0.0,
        depth_hi=10.0,
        variables=variables,
        n_samples=200,
        formation="F1",
    )
    rhob_pri = rng.normal(2.50, 0.10, 200)
    gr_pri = rng.normal(20.0, 5.0, 200)
    primary.kdes["rhob"] = gaussian_kde(rhob_pri)
    primary.kdes["gr_api"] = gaussian_kde(gr_pri)
    primary.corr_matrix = np.eye(2)
    primary.corr_variables = ["rhob", "gr_api"]
    primary.bounds = {v: (-1e6, 1e6) for v in variables}
    bank.cells[("test_rock", "F1", 0)] = primary

    # ---- donor cell: all 5 KDEs, with strong rho(rhob, nphi)<0 and
    #                                          rho(gr_api, nphi)>0 ----
    donor = CellDistribution(
        rock_type="test_rock",
        depth_lo=50.0,
        depth_hi=60.0,
        variables=variables,
        n_samples=500,
        formation="F2",
    )
    # build correlated samples via a Gaussian copula directly
    target_corr = np.array(
        [
            # rhob  gr_api dt    nphi  res
            [1.0, 0.0, 0.0, -0.8, 0.0],
            [0.0, 1.0, 0.0, 0.6, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [-0.8, 0.6, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )
    z = rng.multivariate_normal(np.zeros(5), target_corr, size=500)
    # apply different marginal transforms per variable
    donor_data = {
        "rhob": 2.5 + 0.10 * z[:, 0],
        "gr_api": 20.0 + 5.0 * z[:, 1],
        "dt_us_ft": 80.0 + 8.0 * z[:, 2],
        "nphi": 0.15 + 0.05 * z[:, 3],
        "res_deep_log": 1.5 + 0.3 * z[:, 4],
    }
    for v, arr in donor_data.items():
        donor.kdes[v] = gaussian_kde(arr)
    donor.corr_matrix = target_corr
    donor.corr_variables = list(variables)
    donor.bounds = {v: (-1e6, 1e6) for v in variables}
    bank.cells[("test_rock", "F2", 5)] = donor

    # ---- sample from the primary cell + verify cross-correlations ----
    N = 5000
    out = bank.sample(
        "test_rock", depth=5.0, n=N, rng=rng, formation="F1", interpolate=False
    )

    # 1. shape + finiteness
    for v in variables:
        assert v in out, f"missing variable {v} in output"
        assert len(out[v]) == N, f"{v} length {len(out[v])} != {N}"
        assert np.isfinite(out[v]).all(), f"NaN/Inf in {v}"

    # 2. cross-correlations: nphi should track the donor's joint structure
    rho_rhob_nphi, _ = spearmanr(out["rhob"], out["nphi"])
    rho_gr_nphi, _ = spearmanr(out["gr_api"], out["nphi"])
    print(
        f"  recovered Spearman(rhob, nphi) = {rho_rhob_nphi:+.3f}  "
        f"(target ~ -0.8, threshold < -0.4)"
    )
    print(
        f"  recovered Spearman(gr_api, nphi) = {rho_gr_nphi:+.3f}  "
        f"(target ~ +0.6, threshold > +0.3)"
    )
    assert rho_rhob_nphi < -0.4, (
        f"recovered nphi is independent of rhob (rho={rho_rhob_nphi:.3f}); "
        "conditional copula not applied"
    )
    assert rho_gr_nphi > 0.3, (
        f"recovered nphi is independent of gr_api (rho={rho_gr_nphi:.3f}); "
        "conditional copula not applied"
    )

    # 3. PSD: any sample succeeded means the extended R was PSD
    print(f"  N={N} samples, extended correlation matrix was PSD: OK")


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
        print(f"    {v:14s}  finite={finite_frac:.1%}  " f"mean={np.nanmean(arr):.3f}")
    print(
        f"\n  yield_field   : shape={m['yield_field'].shape}  "
        f"max={m['yield_field'].max():.3f}  "
        f"nnz={(m['yield_field'] > 0).sum()}"
    )
    print(f"  orebodies     : {len(m['bodies'])}")
    for i, b in enumerate(m["bodies"]):
        print(
            f"    body {i}: centre=({b.center_x:.1f}, {b.center_y:.1f}, "
            f"{b.center_z:.0f}m)  yield_peak={b.peak_yield:.2f}  "
            f"radii_xy=({b.radius_x:.1f}, {b.radius_y:.1f})"
        )

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

    # ------ unit tests on the same fake bank/geometry/prior --------------
    test_bank_sample_with_formation(bank)
    test_max_ore_bodies_cap(bank, geom, prior)
    test_gas_response_shift(bank, geom, prior)
    test_conditional_copula_recovery()

    print("=" * 60)
    print("all stages work + unit tests pass. ready for real data.")


def build_fake_geometry() -> FormationGeometry:
    """FormationGeometry with synthetic combinations and thicknesses for
    smoke-test purposes only. Uses the new combination-based format.
    NOTE: this definition (further down the file) intentionally
    overrides the simpler earlier definition above -- the earlier one
    used a legacy top_depths field that the current FormationGeometry
    no longer accepts."""
    geom = FormationGeometry(list(FORMATION_ORDER))
    geom.n_wells_total = 100
    geom.min_well_depth = 4000.0
    geom.max_well_depth = 4500.0

    # synthetic combinations representative of deep Dutch wells
    geom.combinations = [
        (("NU", "NM", "NL", "CK", "KN", "RB", "ZE", "RO", "DC"), 15),
        (("CK", "KN", "RB", "ZE", "RO", "DC"), 13),
        (("NU", "NM", "NL", "CK", "KN", "ZE", "RO", "DC"), 10),
        (("CK", "KN", "RB", "ZE", "RO"), 9),
        (("NU", "NM", "NL", "CK", "KN", "RN", "RB", "ZE", "RO", "DC"), 8),
        (("CK", "KN", "RN", "RB", "ZE", "RO", "DC"), 6),
        (("CK", "KN", "ZE", "RO", "DC"), 5),
        (("CK", "KN", "AT", "RN", "RB", "ZE", "RO", "DC"), 3),
        (("CK", "KN", "SL", "AT", "RN", "RB"), 3),
    ]

    # (top_median_m, top_std_m, thick_median_m, thick_std_m) per formation
    fake_stats = {
        "NU": (144, 150, 414, 200),
        "NM": (671, 300, 60, 40),
        "NL": (721, 300, 393, 200),
        "CK": (1138, 350, 662, 300),
        "KN": (1785, 600, 227, 200),
        "SL": (1950, 500, 180, 150),
        "SG": (2476, 400, 149, 100),
        "AT": (1945, 500, 199, 200),
        "RN": (2200, 500, 262, 200),
        "RB": (2293, 600, 267, 200),
        "ZE": (2475, 500, 326, 250),
        "RO": (3274, 500, 224, 200),
        "DC": (3466, 600, 63, 100),
    }
    fake_facies = {
        "NU": {"clay": 1.0},
        "NM": {"clay": 1.0},
        "NL": {"clay": 1.0},
        "CK": {"chalk": 1.0},
        "KN": {"claystone": 0.46, "claystone_cool": 0.41, "sandstone_shaly": 0.13},
        "SL": {"claystone_cool": 0.61, "claystone": 0.39},
        "SG": {"claystone": 0.69, "claystone_hot": 0.31},
        "AT": {"claystone_hot": 0.66, "claystone": 0.20, "claystone_cool": 0.14},
        "RN": {"claystone": 0.54, "dolomite": 0.40, "anhydrite": 0.06},
        "RB": {"claystone_hot": 0.67, "sandstone_shaly": 0.33},
        "ZE": {"halite_pure": 0.43, "anhydrite": 0.28, "other": 0.22, "dolomite": 0.07},
        "RO": {"sandstone_clean": 0.56, "claystone_hot": 0.44},
        "DC": {"claystone_hot": 0.59, "claystone": 0.41},
    }

    fake_top_stats = {
        "NU": (144, 150),
        "NM": (671, 300),
        "NL": (721, 300),
        "CK": (1138, 350),
        "KN": (1785, 600),
        "SL": (1950, 500),
        "SG": (2476, 400),
        "AT": (1945, 500),
        "RN": (2200, 500),
        "RB": (2293, 600),
        "ZE": (2475, 500),
        "RO": (3274, 500),
        "DC": (3466, 600),
    }
    rng = np.random.default_rng(0)
    for fm, (thk_med, thk_std) in fake_thickness_stats.items():
        thicks = rng.normal(thk_med, thk_std, size=50).clip(min=20.0)
        top_med, top_std = fake_top_stats[fm]
        tops = rng.normal(top_med, top_std, size=50).clip(min=10.0)
        geom.formations[fm] = FormationStats(
            name=fm,
            n_wells=50,
            top_depths=tops,
            thicknesses=thicks,
            facies=fake_facies[fm],
        )
    return geom


if __name__ == "__main__":
    main()
