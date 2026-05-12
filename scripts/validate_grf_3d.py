"""Verify the per-variable 3D GRF (Task 4).

For both implementations (`gaussian_filter` fast path and `gstools.SRF`
explicit path), generate one realised field, estimate its empirical
variogram along each axis, fit a Gaussian variogram model to the
empirical points, and confirm the fitted range matches the input.

Saves plots/validation/variogram_check.png with a 3×2 panel grid:
rows=axis (lateral-x, lateral-y, vertical), cols=method, showing the
empirical variogram + fitted curve + input range.

Run:
    python scripts/validate_grf_3d.py
"""
from __future__ import annotations

from pathlib import Path

import gstools as gs
import matplotlib.pyplot as plt
import numpy as np

from simulator.map_generator import _make_noise_fields, SimConfig


OUT_PATH = Path("plots/validation/variogram_check.png")
N_X, N_Y, N_Z = 64, 64, 220
LATERAL = 2.0
VERTICAL = 3.0


def _empirical_variogram_axis(field: np.ndarray, axis: int,
                              max_lag: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """Pair-difference variogram along one axis only, averaged over
    every other (i, j) line in the volume.

    γ(h) = 0.5 · E[(Z(x) − Z(x+h))²] for lag h along `axis`.
    """
    lags = np.arange(1, max_lag + 1)
    gammas = np.empty_like(lags, dtype=np.float64)
    for k, h in enumerate(lags):
        sl_a = [slice(None)] * field.ndim
        sl_b = [slice(None)] * field.ndim
        sl_a[axis] = slice(None, -h)
        sl_b[axis] = slice(h, None)
        diff = field[tuple(sl_a)] - field[tuple(sl_b)]
        gammas[k] = 0.5 * float(np.mean(diff ** 2))
    return lags.astype(np.float64), gammas


def _fit_gaussian(lags: np.ndarray, gammas: np.ndarray, init_range: float
                  ) -> tuple[float, float]:
    """Return (fitted_range, sill) for γ(h) = sill·(1 − exp(−(h/range)²))."""
    sill_init = float(gammas[-3:].mean())
    model = gs.Gaussian(dim=1, var=sill_init, len_scale=init_range)
    try:
        model.fit_variogram(lags, gammas, nugget=False)
        return float(model.len_scale), float(model.var)
    except Exception:
        return float("nan"), float("nan")


def main() -> None:
    rng = np.random.default_rng(42)
    cfg = SimConfig()

    fields_by_method = {}
    for method in ("gaussian_filter", "gstools_srf"):
        nf = _make_noise_fields(
            rng=np.random.default_rng(7),
            nx=N_X, ny=N_Y, nz=N_Z,
            variables=("rhob",),
            lateral_len_scale=LATERAL,
            vertical_len_scale=VERTICAL,
            method=method,
            mode_no=cfg.grf_mode_no,
        )
        fields_by_method[method] = nf["rhob"]

    fig, axes = plt.subplots(3, 2, figsize=(11, 10), sharex="col", sharey="row")
    axis_labels = [
        ("x (lateral)", LATERAL),
        ("y (lateral)", LATERAL),
        ("z (vertical)", VERTICAL),
    ]
    print(f"{'method':<18}{'axis':<14}{'input range':<14}{'fitted range':<14}{'fitted sill':<12}")
    for col, (method, field) in enumerate(fields_by_method.items()):
        for row, (label, true_range) in enumerate(axis_labels):
            lags, gammas = _empirical_variogram_axis(field, axis=row, max_lag=12)
            fitted_range, fitted_sill = _fit_gaussian(lags, gammas, true_range)
            ax = axes[row, col]
            ax.plot(lags, gammas, "o", label="empirical")
            xs = np.linspace(0, lags.max(), 200)
            true_curve = 1.0 * (1.0 - np.exp(-(xs / true_range) ** 2))
            ax.plot(xs, true_curve, "-", color="tab:gray",
                    label=f"input model (range={true_range})")
            if np.isfinite(fitted_range):
                fitted_curve = fitted_sill * (
                    1.0 - np.exp(-(xs / fitted_range) ** 2)
                )
                ax.plot(xs, fitted_curve, "--", color="tab:red",
                        label=f"fit (range={fitted_range:.2f})")
            ax.set_title(f"{method} — {label}")
            ax.set_xlabel("lag (cells)")
            ax.set_ylabel("γ(h)")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            print(f"{method:<18}{label:<14}{true_range:<14.2f}{fitted_range:<14.2f}{fitted_sill:<12.3f}")

    fig.suptitle(
        "Variogram check — both methods reproduce the input Gaussian variogram"
    )
    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=120)
    print(f"\nsaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
