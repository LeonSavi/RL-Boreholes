"""Dataset transformation utilities for belief-model training.

These helpers mutate GeologicalBeliefDataset instances in-place and save
any fitted state (e.g. PCA reducers) to the checkpoint directory.
"""

from __future__ import annotations

import json
from pathlib import Path

from .dataset import GeologicalBeliefDataset
from .utils import LatentPCAReducer


def fit_and_apply_latent_pca(
    train_ds: GeologicalBeliefDataset,
    val_ds: GeologicalBeliefDataset,
    n_components: int,
    checkpoint_dir: Path,
    borehole_encoder: str = "unknown",
    verbose: bool = True,
) -> tuple[LatentPCAReducer, int]:
    """Fit PCA on observed training latents and apply it to both datasets.

    Only cells where mask==1 (observed/drilled) are used for fitting, so
    zero-filled unobserved cells do not bias the principal components.
    Both datasets are modified in-place.

    Saves two files to ``checkpoint_dir``:
      * ``latent_pca_stats.json``   — components, explained variance, etc.
      * ``latent_pca_reducer.pkl``  — serialised reducer for inference reuse

    Parameters
    ----------
    train_ds / val_ds : datasets to transform (modified in-place)
    n_components      : desired number of PCA output dimensions
    checkpoint_dir    : directory where stats and reducer are written
    borehole_encoder  : recorded in the stats JSON for provenance
    verbose           : print reduction summary if True

    Returns
    -------
    (reducer, actual_n_components)
        ``actual_n_components`` may be less than ``n_components`` when the
        latent dim or number of observed samples is smaller than requested.
    """
    inputs_np = train_ds.inputs.numpy()          # (N, 2+D, n_x, n_y)
    obs_mask  = inputs_np[:, 1, :, :] > 0.0     # (N, n_x, n_y) bool
    latent_t  = inputs_np[:, 2:, :, :].transpose(0, 2, 3, 1)  # (N, n_x, n_y, D)
    observed  = latent_t[obs_mask]               # (N_obs, D)
    original_latent_dim = observed.shape[1]

    reducer = LatentPCAReducer(n_components=n_components)
    reducer.fit(observed)

    train_ds.apply_latent_pca(reducer)
    val_ds.apply_latent_pca(reducer)

    actual_k = reducer.n_output_components
    evr = reducer.explained_variance_ratio

    pca_meta = {
        "original_latent_dim": original_latent_dim,
        "n_components":        actual_k,
        "explained_variance_ratio":  evr.tolist() if evr is not None else [],
        "explained_variance_total":  float(evr.sum()) if evr is not None else 0.0,
        "encoder_variant":     borehole_encoder,
        "n_observed_samples":  int(observed.shape[0]),
    }
    pca_path = checkpoint_dir / "latent_pca_stats.json"
    with open(pca_path, "w") as f:
        json.dump(pca_meta, f, indent=2)

    reducer.save(checkpoint_dir / "latent_pca_reducer.pkl")

    if verbose:
        print(
            f"  PCA reduction : {original_latent_dim} → {actual_k}"
            f"  (fitted on {observed.shape[0]:,} observed cells)"
        )
        if evr is not None:
            print(f"  expl. variance: {evr.sum():.3f}")
        print(f"  PCA stats     -> {pca_path}")

    return reducer, actual_k
