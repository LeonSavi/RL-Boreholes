"""
Compare simulator facies sequences vs real NLOG wells.

Picks N real NLOG wells and N simulator columns (each from a different
map). For each, prints a run-length summary of the facies sequence
within the formation of interest.

Usage:
    python compare_facies.py
    python compare_facies.py --formations ZE --n_real 5 --n_sim 5
    python compare_facies.py --formations CK KN RB ZE RO --n_sim 10
"""
from __future__ import annotations

import argparse
from collections import Counter

import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulator import (
    DistributionBank, DiscoveryPrior, FormationGeometry,
    MapGenerator, SimConfig,
)


def _print_run_summary(rocks: list[str], depths: list[float]) -> None:
    if not rocks:
        return
    runs = []
    cur_rock = rocks[0]
    cur_top = depths[0]
    cur_n = 1
    for i in range(1, len(rocks)):
        if rocks[i] == cur_rock:
            cur_n += 1
        else:
            runs.append((cur_rock, cur_top, depths[i - 1], cur_n))
            cur_rock = rocks[i]
            cur_top = depths[i]
            cur_n = 1
    runs.append((cur_rock, cur_top, depths[-1], cur_n))

    total = sum(r[3] for r in runs)
    print(f"  RUN-LENGTH SUMMARY ({len(runs)} runs over {total} cells):")
    for rock, top, bot, n in runs:
        bar = "#" * min(n, 30)
        print(f"    {top:>5.0f}-{bot:<5.0f}m  ({n:>3d} cells)  "
              f"{rock:<20s} {bar}")
    counts = Counter(rocks)
    print(f"  facies composition: ", end="")
    for r, c in counts.most_common():
        print(f"{r}({100*c/len(rocks):.0f}%) ", end="")
    print()


def _sample_real_wells(
    df: pd.DataFrame,
    formation: str,
    n_wells: int,
    rng: np.random.Generator,
    min_thickness_m: float = 100.0,
) -> list[str]:
    sub = df[df["formation"] == formation]
    well_thick = sub.groupby("borehole")["depth"].agg(["min", "max"])
    well_thick["thk"] = well_thick["max"] - well_thick["min"]
    qualifying = well_thick[well_thick["thk"] >= min_thickness_m]
    if len(qualifying) == 0:
        return []
    chosen = rng.choice(
        qualifying.index.values,
        size=min(n_wells, len(qualifying)),
        replace=False,
    )
    return [str(b) for b in chosen]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--formations", nargs="+",
                   default=["CK", "KN", "RB", "ZE", "RO"])
    p.add_argument("--n_real", type=int, default=3)
    p.add_argument("--n_sim", type=int, default=5,
                   help="simulator columns — each drawn from a different "
                        "map so we see real composition diversity")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    print("loading samples.parquet...")
    df = pd.read_parquet("data/clean/samples.parquet")
    df = df[df["dataset"] == "NLOG"]
    df = df.drop_duplicates(subset=["dataset", "borehole", "depth"],
                              keep="first")
    df = df.sort_values(["borehole", "depth"])
    print(f"  {len(df):,} rows total\n")

    print("loading simulator...")
    bank = DistributionBank.load("data/clean/distributions.pkl")
    prior = DiscoveryPrior.load("data/clean/discovery_prior.pkl")
    geom = FormationGeometry.load("data/clean/formation_geometry.pkl")

    n_maps_to_generate = max(args.n_sim * 4, 20)
    print(f"  generating {n_maps_to_generate} simulator maps...\n")
    gen = MapGenerator(
        bank, geom, SimConfig(), seed=args.seed, prior=prior,
    )
    sim_maps = [next(gen) for _ in range(n_maps_to_generate)]

    for formation in args.formations:
        print("=" * 78)
        print(f"FORMATION: {formation}")
        print("=" * 78)

        # ---------- REAL NLOG WELLS ----------
        wells = _sample_real_wells(df, formation, args.n_real, rng)
        if not wells:
            print(f"  no qualifying real wells for {formation}\n")
            continue

        for well in wells:
            sub = df[(df["borehole"] == well)
                       & (df["formation"] == formation)]
            sub = sub.sort_values("depth")
            depths = sub["depth"].values
            rocks = sub["rock_type_fine"].astype(str).values.tolist()
            print(f"\n--- REAL well {well} ({formation}) ---")
            print(f"  interval: {depths[0]:.0f}-{depths[-1]:.0f}m  "
                  f"({len(depths)} cells)")
            _print_run_summary(rocks, list(depths))

        # ---------- SIMULATOR COLUMNS ----------
        n_sim_printed = 0
        ix, iy = 16, 16
        for map_idx, m in enumerate(sim_maps):
            if n_sim_printed >= args.n_sim:
                break
            formations_arr = m["formations"]
            rock_types = m["rock_types"]
            depth_axis = m["depth_axis"]
            col_fm = formations_arr[ix, iy, :]
            mask = (col_fm == formation)
            if not mask.any():
                continue
            idxs = np.where(mask)[0]
            d_top = depth_axis[idxs[0]]
            d_bot = depth_axis[idxs[-1]]
            thk = d_bot - d_top
            if thk < 100.0:
                continue
            rocks_in_fm = rock_types[ix, iy, idxs].astype(str).tolist()
            depths_in_fm = depth_axis[idxs].tolist()
            print(f"\n--- SIM map {map_idx} (cell {ix},{iy}) "
                  f"({formation}) ---")
            print(f"  interval: {d_top:.0f}-{d_bot:.0f}m  "
                  f"({len(rocks_in_fm)} cells)")
            _print_run_summary(rocks_in_fm, depths_in_fm)
            n_sim_printed += 1

        if n_sim_printed == 0:
            print(f"\n  (no simulator map contained {formation} at "
                  f"cell ({ix},{iy}))")
        elif n_sim_printed < args.n_sim:
            print(f"\n  (only {n_sim_printed} maps contained {formation}; "
                  f"asked for {args.n_sim})")
        print()


if __name__ == "__main__":
    main()