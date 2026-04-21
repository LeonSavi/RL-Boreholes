"""
data_layer.py
=============

One-line access to the unified sample table produced by pull_data.py.
Use this in every downstream script instead of re-parsing raw data.

Example
-------
    from data_layer import load_samples, load_wide

    df = load_samples()
    sandstone = df[df.rock_type == "sandstone"]

    # For crossplots — one row per (well, depth), columns per measurement
    wide = load_wide()
    wide[["rhob", "gr_api", "dt_us_ft"]].plot.scatter(...)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

CLEAN_DIR = Path("data/clean")


def load_samples(clean_dir: Path = CLEAN_DIR) -> pd.DataFrame:
    """Load the long-form unified samples table."""
    path = clean_dir / "samples.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python pull_data.py` first.")
    return pd.read_parquet(path)


def load_wide(clean_dir: Path = CLEAN_DIR,
              index_cols: list[str] | None = None) -> pd.DataFrame:
    """Pivot the long-form table so each row is one (well, depth) and
    each measurement becomes its own column. Useful for crossplots."""
    df = load_samples(clean_dir)
    if index_cols is None:
        index_cols = ["dataset", "borehole", "depth", "rock_type",
                      "formation", "lith_principal"]
    present = [c for c in index_cols if c in df.columns]
    return (df.pivot_table(index=present, columns="measurement",
                            values="value", aggfunc="first")
              .reset_index())
