from simulator import DistributionBank
import numpy as np


DATA = "data/clean/samples.parquet"
OUTPUT_DISTR = "data/clean/distributions.pkl"

def save_distributions():
    bank = DistributionBank.fit(
        DATA,
        variables=["rhob", "gr_api", "dt_us_ft", "nphi", "pef",
                "cali_in", "res_deep_log", "sp_mv", "drho",
                  #"msus_si" magnetic sus removed because it exists only in LILY
                  ],
        depth_bins=[0, 400, 800, 1200, 1600, 2000, 2400, 2800, 3200, 3600, 4000, 4400, 4800, 5200, 5600, 6000],
    )
    bank.save(OUTPUT_DISTR)

    return bank

def distribution_check():

    # sanity check for distributions

    bank = DistributionBank.load(OUTPUT_DISTR)

    # Pick three well-populated cells with full correlation matrices
    for key in [("claystone", 5), ("sandstone", 6), ("halite", 6)]:
        # bin 5 = 2000-2400m, bin 6 = 2400-2800m
        cell = bank.cells.get(key)
        if cell is None or cell.corr_matrix is None:
            continue

        print(f"\n{key[0]} @ bin {key[1]} (n={cell.n_samples:,}):")
        vars_ = cell.corr_variables

        # print correlations between the most physically-meaningful pairs
        pairs = [("rhob", "dt_us_ft"), ("rhob", "nphi"), ("rhob", "gr_api"),
                ("nphi", "dt_us_ft"), ("gr_api", "res_deep_log")]
        for a, b in pairs:
            if a in vars_ and b in vars_:
                i, j = vars_.index(a), vars_.index(b)
                print(f"  {a:14s} x {b:14s}  {cell.corr_matrix[i, j]:+.3f}")


if __name__ == '__main__':

    bank = save_distributions()

    print(bank.summary())

    distribution_check()