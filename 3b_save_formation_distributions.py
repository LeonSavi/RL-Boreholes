"""
Fit the formation-resolution distribution bank.

This is the parallel companion to `3_save_distributions.py`. It fits
the same petrophysical KDE / copula / PSD machinery, but indexed by
`(formation, depth_bin)` instead of `(rock_type_fine, depth_bin)`.

The output (`data/clean/formation_distributions.pkl`) is consumed by
`4b_pull_maps_formation.py` to generate the parallel formation-
resolution training dataset. `DiscoveryPrior` and `FormationGeometry`
are not re-fitted — they're shared with the rock pipeline.

Outputs:
  data/clean/formation_distributions.pkl
"""
from simulator.formation_distributions import FormationDistributionBank


DATA               = "data/clean/samples.parquet"
OUTPUT_DISTR       = "data/clean/formation_distributions.pkl"

VARIABLES = [
    "rhob", "gr_api", "dt_us_ft", "nphi", "res_deep_log",
]

DEPTH_BINS = [0, 400, 800, 1200, 1600, 2000, 2400, 2800, 3200,
              3600, 4000, 4400, 4800, 5200, 5600, 6000]


def fit_and_save():
    print("\nFitting FormationDistributionBank ...")
    bank = FormationDistributionBank.fit(
        DATA, variables=VARIABLES, depth_bins=DEPTH_BINS,
    )
    bank.save(OUTPUT_DISTR)
    return bank


if __name__ == "__main__":
    bank = fit_and_save()
    print()
    print(bank.summary().to_string(index=False))
