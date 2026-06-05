"""
Latent-space validation for the trained BoreholeAutoencoder — CORRECTED.

Key fix vs. the previous version
--------------------------------
The previous script used `MapGenerator`, which produces a 2D (nx=32, ny=32)
grid of boreholes per map — and ALL 1024 boreholes within one map share
the same base stratigraphic column (perturbed only by ±20m smooth layer
wiggles). To get 3000 boreholes the script generated just 3 maps, so the
validation set was really 3 distinct stratigraphies, each duplicated
~1024 times.

That's why the latent plot showed three perfectly separated clusters
regardless of rock-type colouring: the encoder was correctly clustering
the three underlying geologies, but the random-window rock-type labels
were independent of which of the three a borehole came from.

Fix: call `sample_column(rng, max_depth)` directly N times to get N
INDEPENDENT stratigraphies. No 2D map at all. Every borehole has its own
freshly-drawn formation stack. This is the right validation object for
a 1D encoder.

What the script produces
------------------------
A single UMAP reduction of the encoded latents, plotted as a 2x2 figure
with four different colourings of the SAME points:

  (A) Dominant rock type in the middle 50% of the column
       → does the encoder cluster rocks that look alike petrophysically?

  (B) Dominant formation in the middle 50% of the column
       → does the encoder cluster by stratigraphic position rather than
         rock type? (e.g. RO-sandstone and RB-sandstone both look like
         sandstone but sit at very different depths/pressures)

  (C) Deepest formation reached
       → does the encoder's cluster structure just reflect how deep the
         well goes? (The most common confound when "rock type" fails to
         explain clusters.)

  (D) Rock type at a fixed deep window (bottom 25%)
       → isolates the rocks that only appear at depth (halite_pure,
         claystone_hot, sandstone_clean/RO) and checks whether they
         cluster.

Between these four panels you can diagnose whether the encoder learned
(a) rocks, (b) formations, (c) borehole-depth envelope, or (d) something
else entirely.

Usage
-----
    python latent_validation.py \\
        --checkpoint checkpoints/ae.pt \\
        --distributions data/clean/distributions.pkl \\
        --n-boreholes 3000 \\
        --out latent_validation.png
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from encoder.autoencoder import load_checkpoint, standardise
from simulator.distributions import DistributionBank
from simulator.formation_geometry import FormationGeometry
from simulator.map_generator import SimConfig
from simulator.stratigraphy import sample_column


# ---------------------------------------------------------------------------
# 1. borehole generation — INDEPENDENT columns (no 2D map)
# ---------------------------------------------------------------------------
def generate_independent_boreholes(
    n_boreholes: int,
    variables: list[str],
    bank: DistributionBank,
    geometry: FormationGeometry,
    sim_cfg: SimConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate N boreholes, each with an independently sampled
    stratigraphic column.

    Returns
    -------
    values : (N, V, D) float32 — the petrophysical log values
    rocks  : (N, D) object — rock type per (borehole, depth)
    forms  : (N, D) object — formation per (borehole, depth)
    depth_axis : (D,) float32
    """
    nz = sim_cfg.n_depth
    depth_axis = np.linspace(0, sim_cfg.max_depth, nz, dtype=np.float32)

    # Step 1: sample N independent stratigraphic columns and rasterise.
    print(f"  [1/2] sampling {n_boreholes} independent columns...")
    rocks = np.empty((n_boreholes, nz), dtype=object)
    forms = np.empty((n_boreholes, nz), dtype=object)
    for i in range(n_boreholes):
        col = sample_column(rng, geometry, max_depth=sim_cfg.max_depth)
        r, f = col.rasterise(depth_axis)
        rocks[i, :] = r
        forms[i, :] = f

    # Step 2: populate variable values. For each depth slice, batch
    # by rock type (one bank.sample call per (rock, depth) group).
    # This matches map_generator.py's approach but operates on a 1D
    # array of boreholes rather than a 2D (x, y) grid.
    print(f"  [2/2] populating variable values per (rock, depth) group...")
    values = {
        v: np.full((n_boreholes, nz), np.nan, dtype=np.float32)
        for v in variables
    }

    for z_idx in range(nz):
        depth = float(depth_axis[z_idx])
        col = rocks[:, z_idx]
        fm_col = forms[:, z_idx]
        # v3.1: bank keys on (rock, formation, depth_bin) -- batch by
        # (rock, formation) so each bank.sample() draws from the correct
        # formation-specific cell.
        pairs = set(zip(col.tolist(), fm_col.tolist()))
        for rock, formation in pairs:
            if rock is None:
                continue
            mask = (col == rock) & (fm_col == formation)
            idxs = np.where(mask)[0]
            n_cells = len(idxs)
            if n_cells == 0:
                continue
            try:
                samples = bank.sample(
                    str(rock), depth, n=n_cells, rng=rng,
                    formation=str(formation) if formation is not None else None,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"    [warn] bank.sample failed for "
                      f"({rock}, {formation}, {depth:.0f}m): {exc!r}")
                continue
            for v in variables:
                if v not in samples:
                    continue
                draws = samples[v]
                if np.all(np.isnan(draws)):
                    continue
                values[v][idxs, z_idx] = draws.astype(np.float32)

    # stack to (N, V, D)
    stacked = np.stack([values[v] for v in variables], axis=1)
    return stacked, rocks, forms, depth_axis


# ---------------------------------------------------------------------------
# 2. labelling — several strategies on the same boreholes
# ---------------------------------------------------------------------------
def dominant_in_window(
    col: np.ndarray, lo: int, hi: int,
    min_homogeneity: float = 0.6,
) -> str:
    """Most common non-'other' value in col[lo:hi]. Returns 'mixed' if
    no value reaches `min_homogeneity` fraction of the window."""
    window = col[lo:hi]
    counts = Counter(window)
    counts.pop("other", None)
    if not counts:
        return "other"
    top, n = counts.most_common(1)[0]
    if n / len(window) < min_homogeneity:
        return "mixed"
    return str(top)


def compute_all_labels(
    rocks: np.ndarray, forms: np.ndarray, nz: int,
) -> dict[str, list[str]]:
    """Four label schemes for the same set of boreholes."""
    mid_lo, mid_hi = nz // 4, 3 * nz // 4        # middle 50%
    deep_lo, deep_hi = 3 * nz // 4, nz            # bottom 25%

    rock_middle = [
        dominant_in_window(rocks[i], mid_lo, mid_hi) for i in range(len(rocks))
    ]
    form_middle = [
        dominant_in_window(forms[i], mid_lo, mid_hi) for i in range(len(rocks))
    ]
    rock_deep = [
        dominant_in_window(rocks[i], deep_lo, deep_hi) for i in range(len(rocks))
    ]
    # "deepest formation reached" = the formation at the last depth index
    # whose value isn't 'other'.
    form_deepest = []
    for i in range(len(rocks)):
        fs = forms[i]
        last = "other"
        for f in fs[::-1]:
            if f != "other":
                last = str(f)
                break
        form_deepest.append(last)

    return {
        "rock (middle 50%)":     rock_middle,
        "formation (middle 50%)": form_middle,
        "formation (deepest)":    form_deepest,
        "rock (bottom 25%)":      rock_deep,
    }


# ---------------------------------------------------------------------------
# 3. encoding + dimensionality reduction
# ---------------------------------------------------------------------------
def encode_in_batches(
    model, values: np.ndarray, stats, variables, device, batch_size: int = 256,
) -> np.ndarray:
    """Feed (N, V, D) values through the encoder in batches."""
    std = standardise(values, stats, variables)
    std = np.nan_to_num(std, nan=0.0).astype(np.float32)
    latents = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(std), batch_size):
            x = torch.from_numpy(std[i : i + batch_size]).to(device)
            z = model.encoder(x).cpu().numpy()
            latents.append(z)
    return np.concatenate(latents, axis=0)


def reduce_to_2d(Z: np.ndarray, method: str = "umap") -> np.ndarray:
    if method == "umap":
        try:
            import umap
            # n_neighbors=75 instead of default 15 — for 3000 points we
            # want global structure to win over very-local neighbourhoods,
            # otherwise UMAP tends to manufacture tight round blobs.
            reducer = umap.UMAP(
                n_neighbors=75, min_dist=0.1, random_state=42, verbose=False,
            )
            return reducer.fit_transform(Z)
        except ImportError:
            print("  [warn] umap-learn not installed; using PCA instead")
            method = "pca"
    if method == "pca":
        from sklearn.decomposition import PCA
        return PCA(n_components=2, random_state=42).fit_transform(Z)
    if method == "tsne":
        from sklearn.manifold import TSNE
        return TSNE(
            n_components=2, random_state=42, perplexity=30,
        ).fit_transform(Z)
    raise ValueError(f"unknown reducer {method}")


# ---------------------------------------------------------------------------
# 4. plotting — 2x2 figure, same points, four colourings
# ---------------------------------------------------------------------------
ROCK_COLOURS = {
    "sandstone_clean":  "#d35400",
    "sandstone_shaly":  "#e67e22",
    "sandstone":        "#f39c12",
    "claystone_hot":    "#8e44ad",
    "claystone_cool":   "#9b59b6",
    "claystone":        "#bb8fce",
    "clay":             "#3498db",
    "chalk":            "#ecf0f1",
    "halite_pure":      "#16a085",
    "halite":           "#48c9b0",
    "anhydrite":        "#2c3e50",
    "dolomite":         "#c0392b",
    "carbonate":        "#e74c3c",
    "limestone":        "#f1c40f",
    "siltstone":        "#95a5a6",
    "mudstone":         "#7f8c8d",
    "basalt":           "#1b2631",
    "mixed":            "#dddddd",
    "other":            "#cccccc",
    "rare":             "#aaaaaa",
}

# Dutch formations, ordered shallow (top) to deep (bottom).  Using a
# diverging-ish palette keeps shallow=blue, deep=red so panel (C)
# reveals any depth-envelope structure by eye.
FORMATION_ORDER = ["NU", "NM", "NL", "CK", "KN", "SL", "SG", "AT",
                   "RN", "RB", "ZE", "RO", "DC"]


def formation_colour(fm: str) -> str:
    if fm in ("other", "mixed", "rare"):
        return "#cccccc"
    if fm not in FORMATION_ORDER:
        return "#777777"
    # map index along the stratigraphic column to a perceptual colourmap
    idx = FORMATION_ORDER.index(fm)
    frac = idx / max(1, len(FORMATION_ORDER) - 1)
    from matplotlib import cm
    rgba = cm.viridis(frac)
    return "#%02x%02x%02x" % tuple(int(c * 255) for c in rgba[:3])


def _plot_one_panel(
    ax, Z2: np.ndarray, labels: list[str],
    palette: dict | callable, title: str,
    min_count: int = 15,
) -> None:
    counts = Counter(labels)
    kept = {k for k, n in counts.items() if n >= min_count}
    display_labels = [l if l in kept else "rare" for l in labels]

    unique = sorted(set(display_labels))
    # plot 'mixed' / 'rare' / 'other' in the background first
    bg = [u for u in ("other", "rare", "mixed") if u in unique]
    fg = [u for u in unique if u not in bg]
    order = bg + fg

    for val in order:
        idxs = [i for i, l in enumerate(display_labels) if l == val]
        if not idxs:
            continue
        if callable(palette):
            c = palette(val)
        else:
            c = palette.get(val, "#777777")
        alpha = 0.25 if val in ("mixed", "rare", "other") else 0.80
        edge = "black" if val == "chalk" else "none"
        lw = 0.25 if edge == "black" else 0.0
        n_display = (counts[val] if val != "rare"
                     else sum(1 for l in labels if l not in kept))
        ax.scatter(
            Z2[idxs, 0], Z2[idxs, 1],
            c=c, edgecolors=edge, linewidths=lw, s=14, alpha=alpha,
            label=f"{val} (n={n_display})",
        )
    ax.set_xlabel("latent dim 1")
    ax.set_ylabel("latent dim 2")
    ax.set_title(title)
    ax.legend(
        loc="center left", bbox_to_anchor=(1.01, 0.5),
        frameon=False, fontsize=7,
    )


def plot_four_panels(
    Z2: np.ndarray,
    label_sets: dict[str, list[str]],
    out_path: Path,
    n_boreholes: int,
    method: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    panel_configs = [
        ("rock (middle 50%)",     ROCK_COLOURS),
        ("formation (middle 50%)", formation_colour),
        ("formation (deepest)",    formation_colour),
        ("rock (bottom 25%)",      ROCK_COLOURS),
    ]
    for ax, (key, palette) in zip(axes.ravel(), panel_configs):
        _plot_one_panel(ax, Z2, label_sets[key], palette, key)

    fig.suptitle(
        f"Borehole autoencoder latent space — INDEPENDENT columns "
        f"(N={n_boreholes}, {method.upper()}, n_neighbors=75)",
        fontsize=13, y=1.00,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"\n  → saved {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. entrypoint
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path,
                   default=Path("checkpoints/ae.pt"))
    p.add_argument("--distributions", type=Path,
                   default=Path("data/clean/distributions.pkl"))
    p.add_argument("--n-boreholes", type=int, default=3000,
                   help="number of INDEPENDENT boreholes to encode")
    p.add_argument("--out", type=Path,
               default=Path("plots/encoders/latent_validation.png"))
    p.add_argument("--method", choices=["umap", "pca", "tsne"], default="umap")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    print(f"loading checkpoint: {args.checkpoint}")
    model, stats, variables = load_checkpoint(args.checkpoint, device=device)
    model.eval()
    print(f"  variables: {variables}")
    print(f"  latent dim: {model.cfg.latent_dim}")

    print(f"loading distributions: {args.distributions}")
    bank = DistributionBank.load(args.distributions)
    geom = FormationGeometry.load("data/clean/formation_geometry.pkl")
    sim_cfg = SimConfig()
    rng = np.random.default_rng(args.seed)

    print(f"\ngenerating {args.n_boreholes} independent boreholes...")
    values, rocks, forms, depth_axis = generate_independent_boreholes(
        n_boreholes=args.n_boreholes,
        variables=variables,
        bank=bank,
        geometry=geom,
        sim_cfg=sim_cfg,
        rng=rng,
    )
    print(f"  values shape:     {values.shape}")
    print(f"  NaN fraction:     {np.isnan(values).mean():.2%}")

    print(f"\nencoding through the model...")
    Z = encode_in_batches(model, values, stats, variables, device)
    print(f"  latents shape:    {Z.shape}")

    print(f"\ncomputing label sets...")
    label_sets = compute_all_labels(rocks, forms, nz=values.shape[-1])
    for name, labels in label_sets.items():
        top5 = Counter(labels).most_common(5)
        top5_str = ", ".join(f"{k}={v}" for k, v in top5)
        print(f"  {name:28s} top5: {top5_str}")

    print(f"\nreducing to 2D via {args.method}...")
    Z2 = reduce_to_2d(Z, method=args.method)

    plot_four_panels(
        Z2=Z2, label_sets=label_sets,
        out_path=args.out,
        n_boreholes=args.n_boreholes,
        method=args.method,
    )

    # Quick quantitative summary: silhouette scores.  A higher score for
    # formation-based labelling than rock-based labelling would confirm
    # the encoder prioritises stratigraphic position over rock type.
    try:
        from sklearn.metrics import silhouette_score
        print("\nsilhouette scores (higher = labels match clusters better):")
        for name, labels in label_sets.items():
            # only score on the non-rare points
            counts = Counter(labels)
            keep = [i for i, l in enumerate(labels) if counts[l] >= 15
                    and l not in ("mixed", "other", "rare")]
            if len(set(labels[i] for i in keep)) < 2:
                print(f"  {name:28s}  (not enough labels)")
                continue
            sub_Z = Z[keep]
            sub_labels = [labels[i] for i in keep]
            score = silhouette_score(sub_Z, sub_labels, sample_size=2000,
                                     random_state=42)
            print(f"  {name:28s}  {score:+.3f}  (n={len(keep)})")
    except ImportError:
        pass


if __name__ == "__main__":
    main()