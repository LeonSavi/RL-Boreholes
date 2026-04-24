"""
Fit DistributionBank from cleaned samples.parquet and run sanity checks
on the resulting correlations.

Key updates for schema:
  * 6 petrophysics variables (dropped msus_si, drho, sp_mv, cali_in)
  * Refined rock_type_fine labels:
      - sandstone  → sandstone_clean | sandstone_shaly
      - claystone  → claystone_cool | claystone_hot (+ 'claystone' fallback)
      - halite     → halite_pure    (+ 'halite' fallback)
      - carbonate  → dolomite       (+ 'carbonate' fallback)
    Chalk, clay, anhydrite unchanged.
"""
from simulator import DistributionBank
import numpy as np

DATA         = "data/clean/samples.parquet"
OUTPUT_DISTR = "data/clean/distributions.pkl"

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


def save_distributions():
    bank = DistributionBank.fit(
        DATA,
        variables=VARIABLES,
        depth_bins=DEPTH_BINS,
    )
    bank.save(OUTPUT_DISTR)
    return bank


# Physically-meaningful pairs to check per rock type.
# Sign expectations (from rock physics):
#   rhob × dt_us_ft  — negative (denser = faster sound)
#   rhob × nphi      — negative (denser = lower porosity) [except halite ~0]
#   rhob × gr_api    — weak positive (shalier = slightly denser) or ~0
#   nphi × dt_us_ft  — positive (both respond to porosity)
#   gr_api × res_deep_log — negative in clean sands, ~0 elsewhere
CHECK_PAIRS = [
    ("rhob", "dt_us_ft"),
    ("rhob", "nphi"),
    ("rhob", "gr_api"),
    ("nphi", "dt_us_ft"),
    ("gr_api", "res_deep_log"),
]

# Cells to inspect — refined rock labels.
# bin indices are into DEPTH_BINS; bin 5 = 2000-2400m, bin 6 = 2400-2800m.
# Each pick below targets a well-populated cell in its reservoir depth range.
CHECK_CELLS = [
    ("claystone_hot",   5),   # Ten Boer / Solling / Carboniferous shale zone
    ("claystone_cool",  4),   # Rijnland / Delfland marls (shallower)
    ("sandstone_clean", 6),   # Slochteren Rotliegend reservoir
    ("sandstone_shaly", 5),   # Buntsandstein / Silverpit
    ("halite_pure",     6),   # Zechstein H members
    ("dolomite",        5),   # Zechstein + Muschelkalk carbonates
    ("chalk",           4),   # Ekofisk/Ommelanden zone
    ("anhydrite",       6),
]


def distribution_check():
    bank = DistributionBank.load(OUTPUT_DISTR)
    print("\n" + "=" * 60)
    print("CORRELATION SANITY CHECK")
    print("=" * 60)

    for key in CHECK_CELLS:
        cell = bank.cells.get(key)
        if cell is None:
            print(f"\n{key[0]} @ bin {key[1]}: NO CELL (skipped — "
                  f"perhaps not enough samples at this depth)")
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


if __name__ == "__main__":
    bank = save_distributions()
    print(bank.summary())
    distribution_check()