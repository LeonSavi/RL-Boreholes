"""
Fit DistributionBank, DiscoveryPrior, and FormationGeometry from cleaned
samples.parquet. Save all three pickles and run sanity checks.

Outputs:
  data/clean/distributions.pkl        — petrophysical distributions
  data/clean/discovery_prior.pkl      — rock-type-vs-depth prior conditioned
                                         on hc_discovery=True
  data/clean/formation_geometry.pkl   — empirical formation depth, thickness,
                                         and facies (replaces the hand-coded
                                         DUTCH_COLUMN in stratigraphy.py)
"""
from simulator.distributions import DistributionBank, DiscoveryPrior
from simulator.formation_geometry import FormationGeometry, FORMATION_ORDER

DATA               = "data/clean/samples.parquet"
OUTPUT_DISTR       = "data/clean/distributions.pkl"
OUTPUT_PRIOR       = "data/clean/discovery_prior.pkl"
OUTPUT_GEOM        = "data/clean/formation_geometry.pkl"

VARIABLES = [
    "rhob", "gr_api", "dt_us_ft", "nphi", "res_deep_log",
    # dropped:
    #   pef      — only 6% of wells have it; the encoder learned a
    #              near-constant zero channel for the other 94%.
    #   msus_si  — LILY-only, no NLOG equivalent
    #   drho     — density-correction, tool-quality not rock physics
    #   sp_mv    — drilling-mud electrochemistry, well medians span 230 mV
    #   cali_in  — borehole diameter, indicates washouts not lithology
]

DEPTH_BINS = list(range(0, 6001, 10))
# 10 m bins (matches the encoder's cell size exactly; what John asked
# for explicitly). Chosen from scripts/diagnostics/bank_bin_width.py:
#   width   bins  coverage  median-n
#    10 m   440   46.7 %       384
#    20 m   220   52.1 %       653
#    50 m    88   59.3 %     1,200
#   100 m    44   62.2 %     2,076
#   400 m    11   68.9 %     6,712   (the old default)
# Decision rule: smallest width whose populated-cell coverage stays
# >= 30 % AND median samples per populated cell stays >= 100 (KDE
# bandwidth comfort). 10 m clears both bars. Empty cells fall through
# the nearest-populated-cell fallback in
# DistributionBank._resolve_cell, so no sampling-code changes needed.


# Physically-meaningful pairs to check per rock type.
# Sign expectations (from rock physics):
#   rhob × dt_us_ft       — negative (denser = faster sound)
#   rhob × nphi           — negative (denser = lower porosity) [except halite ~0]
#   rhob × gr_api         — weak positive (shalier = slightly denser) or ~0
#   nphi × dt_us_ft       — positive (both respond to porosity)
#   gr_api × res_deep_log — negative in clean sands, ~0 elsewhere
CHECK_PAIRS = [
    ("rhob", "dt_us_ft"),
    ("rhob", "nphi"),
    ("rhob", "gr_api"),
    ("nphi", "dt_us_ft"),
    ("gr_api", "res_deep_log"),
]

# With 10 m bins + formation key: (rock, formation, bin_idx).
# Cells picked from the dominant (rock, formation) pairs at typical
# Dutch reservoir depths.
CHECK_CELLS = [
    ("claystone_hot",   "DC", 220),   # Carboniferous source rock
    ("claystone_hot",   "RB", 220),   # Triassic Bunt source
    ("claystone_cool",  "KN", 180),   # Cretaceous marl
    ("sandstone_clean", "RO", 260),   # Slochteren reservoir
    ("sandstone_shaly", "RB", 220),   # Bunt sandstone
    ("halite_pure",     "ZE", 260),   # Zechstein cap
    ("dolomite",        "ZE", 220),   # Zechstein carbonate
    ("chalk",           "CK", 180),   # Chalk Group
    ("anhydrite",       "ZE", 260),   # Zechstein anhydrite
]


def fit_and_save():
    print("\n[1/3] fitting DistributionBank ...")
    bank = DistributionBank.fit(
        DATA, variables=VARIABLES, depth_bins=DEPTH_BINS,
    )
    bank.save(OUTPUT_DISTR)

    print("\n[2/3] fitting DiscoveryPrior ...")
    prior = DiscoveryPrior.fit(
        DATA,
        depth_bins=DEPTH_BINS,
        rock_types=bank.rock_types,
    )
    prior.save(OUTPUT_PRIOR)

    print("\n[3/3] fitting FormationGeometry ...")
    geom = FormationGeometry.fit(
        DATA, formation_order=FORMATION_ORDER,
    )
    geom.save(OUTPUT_GEOM)

    return bank, prior, geom


def distribution_check(bank: DistributionBank):
    print("\n" + "=" * 60)
    print("DISTRIBUTION SANITY CHECK")
    print("=" * 60)
    for key in CHECK_CELLS:
        cell = bank.cells.get(key)
        if cell is None:
            print(f"\n{key[0]} @ bin {key[1]}: NO CELL")
            continue
        if cell.corr_matrix is None:
            print(f"\n{key[0]} @ bin {key[1]}: no correlation matrix "
                  f"(n={cell.n_samples:,})")
            continue
        print(f"\n{key[0]:18s} @ bin {key[1]} (n={cell.n_samples:,}):")
        vars_ = cell.corr_variables
        for a, b in CHECK_PAIRS:
            if a in vars_ and b in vars_:
                i, j = vars_.index(a), vars_.index(b)
                print(f"  {a:14s} x {b:14s}  {cell.corr_matrix[i, j]:+.3f}")


def prior_check(prior: DiscoveryPrior):
    print("\n" + "=" * 60)
    print("DISCOVERY PRIOR SANITY CHECK")
    print("=" * 60)
    print(f"  positive wells: {prior.n_positive_wells}")
    print(f"  positive rows : {prior.n_positive_rows:,}")
    print()
    print(prior.summary().to_string(index=False))


def geometry_check(geom: FormationGeometry):
    print("\n" + "=" * 60)
    print("FORMATION GEOMETRY SANITY CHECK")
    print("=" * 60)
    print(f"  total NLOG wells: {geom.n_wells_total:,}")
    print()
    print(geom.summary().to_string(index=False))


if __name__ == "__main__":
    bank, prior, geom = fit_and_save()
    print(bank.summary())
    distribution_check(bank)
    prior_check(prior)
    geometry_check(geom)
    