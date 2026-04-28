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
    "rhob", "gr_api", "dt_us_ft", "nphi", "pef", "res_deep_log",
    # dropped:
    #   msus_si  — LILY-only, no NLOG equivalent
    #   drho     — density-correction, tool-quality not rock physics
    #   sp_mv    — drilling-mud electrochemistry, well medians span 230 mV
    #   cali_in  — borehole diameter, indicates washouts not lithology
]

DEPTH_BINS = [0, 400, 800, 1200, 1600, 2000, 2400, 2800, 3200,
              3600, 4000, 4400, 4800, 5200, 5600, 6000]


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

# bin 5 = 2000-2400m, bin 6 = 2400-2800m
CHECK_CELLS = [
    ("claystone_hot",   5),
    ("claystone_cool",  4),
    ("sandstone_clean", 6),
    ("sandstone_shaly", 5),
    ("halite_pure",     6),
    ("dolomite",        5),
    ("chalk",           4),
    ("anhydrite",       6),
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
    