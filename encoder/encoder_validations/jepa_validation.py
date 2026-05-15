"""
Latent-space validation for the JEPA encoder — parallel to
latent_validation.py.  Same independent-column sampling, same 2x2 UMAP
plot with four colourings, same silhouette scores.  Only difference: we
load a JEPA checkpoint and use model.embed() instead of model.encoder().

Usage
-----
    python jepa_validation.py \
        --checkpoint checkpoints/jepa.pt \
        --distributions data/clean/distributions.pkl \
        --n-boreholes 3000 \
        --out jepa_validation.png
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from encoder.autoencoder import standardise
from simulator.distributions import DistributionBank
from simulator.formation_geometry import FormationGeometry
from simulator.map_generator import SimConfig
from simulator.stratigraphy import sample_column
from encoder.jepa_encoder import load_jepa_checkpoint

# Reuse helpers from the reconstruction validation script.  They live
# in latent_validation.py at the repo root — we import by adding its
# directory to sys.path.  If you've moved things around, just copy the
# helper functions here.
import sys
sys.path.insert(0, str(Path(__file__).parent))
from encoder.encoder_validations.latent_validation import (
    generate_independent_boreholes,
    compute_all_labels,
    reduce_to_2d,
    plot_four_panels,
)


def encode_in_batches_jepa(
    model, values: np.ndarray, stats, variables, device, batch_size: int = 256,
) -> np.ndarray:
    """Feed (N, V, D) values through model.embed() in batches."""
    std = standardise(values, stats, variables)
    std = np.nan_to_num(std, nan=0.0).astype(np.float32)
    latents = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(std), batch_size):
            x = torch.from_numpy(std[i : i + batch_size]).to(device)
            z = model.embed(x).cpu().numpy()     # (B, D_latent)
            latents.append(z)
    return np.concatenate(latents, axis=0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path,
                   default=Path("checkpoints/jepa.pt"))
    p.add_argument("--distributions", type=Path,
                   default=Path("data/clean/distributions.pkl"))
    p.add_argument("--n-boreholes", type=int, default=3000)
    p.add_argument("--out", type=Path,
                   default=Path("plots/encoders/jepa_validation.png"))
    p.add_argument("--method", choices=["umap", "pca", "tsne"], default="umap")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    print(f"loading JEPA checkpoint: {args.checkpoint}")
    model, stats, variables = load_jepa_checkpoint(args.checkpoint, device=device)
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

    print(f"\nencoding through JEPA model (target encoder, mean-pooled)...")
    Z = encode_in_batches_jepa(model, values, stats, variables, device)
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

    try:
        from sklearn.metrics import silhouette_score
        print("\nsilhouette scores (JEPA encoder):")
        for name, labels in label_sets.items():
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