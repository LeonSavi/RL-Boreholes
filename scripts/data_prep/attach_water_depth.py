"""
Look up per-well water depth for offshore NLOG boreholes.

Inputs
------
data/clean/samples.parquet
    Must carry `borehole`, `x_rd`, `y_rd`, `location_type`.
    `x_rd, y_rd` are Dutch RD New coordinates (EPSG:28992).
--bathymetry PATH
    GeoTIFF covering the Dutch shelf. Both work:
      * EMODnet DTM 2022 — https://emodnet.ec.europa.eu/geoviewer/
        Bathymetry layer -> Download -> bbox ~51-55 N, 2-8 E.
      * GEBCO 2024 — https://download.gebco.net/ subset by bbox.
    Both store elevation as negative-down (sea floor = negative).

Output
------
data/clean/well_water_depth.parquet
    One row per offshore well with columns:
      borehole, water_depth_m (>= 0), lon, lat
plots/analysis/well_water_depth_map.png
    Scatter of wells on a (lon, lat) grid, coloured by water depth,
    so the user can spot-check that depths grow as you move from
    coast into the central / northern North Sea.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PARQUET_IN = Path("data/clean/samples.parquet")
PARQUET_OUT = Path("data/clean/well_water_depth.parquet")
PLOT_OUT = Path("plots/analysis/well_water_depth_map.png")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=Path, default=PARQUET_IN)
    p.add_argument("--bathymetry", type=Path, required=True,
                   help="GeoTIFF covering the Dutch shelf "
                        "(EMODnet or GEBCO; both negative-down).")
    p.add_argument("--out", type=Path, default=PARQUET_OUT)
    p.add_argument("--plot", type=Path, default=PLOT_OUT)
    args = p.parse_args()

    if not args.bathymetry.exists():
        raise FileNotFoundError(
            f"bathymetry GeoTIFF not found: {args.bathymetry}\n"
            f"Download a Dutch-shelf tile (51-55 N, 2-8 E) from "
            f"EMODnet or GEBCO and pass --bathymetry pointing at it."
        )

    # Lazy imports so the module is importable without pyproj/rasterio.
    import pyproj
    import rasterio

    print(f"loading wells from: {args.samples}")
    df = pd.read_parquet(
        args.samples,
        columns=["borehole", "x_rd", "y_rd", "location_type"],
    )
    df = df.drop_duplicates(subset=["borehole"])
    print(f"  {len(df):,} unique boreholes total")

    offshore = df[df["location_type"] == "offshore"].copy()
    print(f"  {len(offshore):,} offshore boreholes")
    if len(offshore) == 0:
        raise RuntimeError("no offshore wells found — check samples.parquet schema")

    # RD New (EPSG:28992) -> WGS84 (EPSG:4326). always_xy=True so the
    # transform takes (x, y) and emits (lon, lat) — matches the order
    # rasterio.sample expects for an EPSG:4326 raster.
    transformer = pyproj.Transformer.from_crs(28992, 4326, always_xy=True)
    xs = offshore["x_rd"].to_numpy()
    ys = offshore["y_rd"].to_numpy()
    lons, lats = transformer.transform(xs, ys)
    offshore["lon"] = lons
    offshore["lat"] = lats

    print(f"opening bathymetry raster: {args.bathymetry}")
    with rasterio.open(args.bathymetry) as src:
        print(f"  raster CRS: {src.crs}  shape: {src.shape}  "
              f"bounds: {tuple(round(b, 2) for b in src.bounds)}")
        # sample() takes (x, y) tuples matching the raster CRS. If the
        # raster is EPSG:4326 the order is (lon, lat). If it's
        # something else, we'd need an extra reproject — but EMODnet
        # and GEBCO subsets are normally delivered in 4326.
        if src.crs is None or src.crs.to_epsg() != 4326:
            print(f"  WARNING: expected EPSG:4326 raster, got {src.crs}. "
                  f"If the values look wrong, reproject the GeoTIFF first.")
        elev = np.array(
            [val[0] for val in src.sample(zip(lons, lats))],
            dtype=np.float64,
        )
        nodata = src.nodata

    if nodata is not None:
        elev = np.where(elev == nodata, np.nan, elev)

    n_nan = int(np.isnan(elev).sum())
    if n_nan:
        print(f"  WARNING: {n_nan} wells fell outside raster coverage "
              f"(NaN elevation). Will be dropped from output.")

    # Convention: bathymetry stores sea floor as negative elevation,
    # so water_depth = -elev (clamped to 0 in case a well sample lands
    # on land where elev > 0).
    water_depth = np.where(np.isfinite(elev),
                            np.maximum(0.0, -elev),
                            np.nan)
    offshore["water_depth_m"] = water_depth
    offshore = offshore.dropna(subset=["water_depth_m"])

    print(f"\nwater-depth distribution over {len(offshore):,} wells:")
    desc = offshore["water_depth_m"].describe(
        percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
    for k, v in desc.items():
        print(f"  {k:>6s}  {v:8.2f} m")

    suspicious_deep = (offshore["water_depth_m"] > 1000.0).sum()
    if suspicious_deep:
        print(f"\n  WARNING: {suspicious_deep} wells with water_depth > 1000 m. "
              f"The Dutch shelf is shallow — likely a CRS or sign bug.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["borehole", "water_depth_m", "lon", "lat"]
    offshore[cols].to_parquet(args.out, index=False)
    print(f"\nwrote {args.out}  ({len(offshore):,} rows)")

    # Sanity scatter -----------------------------------------------------
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not available; skipping scatter plot)")
        return
    fig, ax = plt.subplots(figsize=(7, 8))
    sc = ax.scatter(
        offshore["lon"], offshore["lat"],
        c=offshore["water_depth_m"], cmap="Blues",
        s=18, edgecolor="black", linewidth=0.3,
    )
    cbar = fig.colorbar(sc, ax=ax, label="water depth [m]")
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    ax.set_title(f"Offshore NLOG wells: water depth from bathymetry\n"
                 f"({len(offshore):,} wells, source: {args.bathymetry.name})")
    ax.grid(alpha=0.3)
    args.plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.plot, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.plot}")


if __name__ == "__main__":
    main()
