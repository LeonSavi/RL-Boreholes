from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

from decision_simulator.resources import (
    check_device,
    check_encoder_path,
    check_sim_paths,
    load_decision_resources,
    resolve_resource_paths,
)

from .dataset import build_dataset_from_cache
from .map_cache import NpzMapCacheStore
from .models.unet_belief import UNetBelief
from .training import (
    build_training_config,
    build_map_belief_training_config,
    export_history,
    load_belief_checkpoint,
    load_map_belief_checkpoint,
    save_experiment_config,
    train_neural_belief,
    train_map_belief,
    validate_by_drill_bins,
)


def train_belief_from_colab(
    storage_root: str | Path,
    checkpoint_dir: str | Path,
    device: str = "cuda",
    debug: bool = False,
    borehole_encoder: Literal["jepa", "autoencoder", "none"] = "jepa",
    plot_dir: str | Path | None = None,
    **overrides,
) -> tuple[UNetBelief, list[dict]]:
    """Train the neural geological belief updater from a Colab notebook.

    Example
    -------
    >>> from decision_simulator.neural_belief.colab import train_belief_from_colab

    # JEPA encoder (default)
    >>> model, history = train_belief_from_colab(
    ...     storage_root="/content/drive/MyDrive/thesis",
    ...     checkpoint_dir="checkpoints/belief_jepa",
    ...     device="cuda",
    ...     debug=True,
    ... )

    # Autoencoder
    >>> model, history = train_belief_from_colab(
    ...     storage_root="/content/drive/MyDrive/thesis",
    ...     checkpoint_dir="checkpoints/belief_autoencoder",
    ...     device="cuda",
    ...     borehole_encoder="autoencoder",
    ... )

    # No encoder (ore + mask only, in_channels=2)
    >>> model, history = train_belief_from_colab(
    ...     storage_root="/content/drive/MyDrive/thesis",
    ...     checkpoint_dir="checkpoints/belief_none",
    ...     device="cuda",
    ...     borehole_encoder="none",
    ... )

    Parameters
    ----------
    storage_root
        Absolute root for all data and checkpoint paths, e.g.
        ``"/content/drive/MyDrive/thesis"``.  All relative paths are resolved
        against this directory.
    checkpoint_dir
        Where ``belief_best.pt``, ``belief_last.pt``, ``training_history.json``,
        and ``training_history.csv`` are saved.  Relative paths are resolved
        against ``storage_root``.
    device
        ``"cuda"`` or ``"cpu"``.  Raises a clear error when CUDA is requested
        but unavailable.
    debug
        If True, use a minimal config (2 maps, 2 epochs, base_channels=16) to
        verify the full pipeline end-to-end without a full training run.
    borehole_encoder
        Which encoder provides latent embeddings per borehole:

        ``"jepa"``        — JEPA encoder; ``in_channels = 2 + latent_dim``
        ``"autoencoder"`` — 1-D CNN autoencoder; ``in_channels = 2 + latent_dim``
        ``"none"``        — no encoder; input is ore + mask only, ``in_channels = 2``
    **overrides
        Any field of :class:`NeuralBeliefTrainingConfig` by name, e.g.
        ``n_epochs=75``.  Unknown field names raise ``ValueError``.
        ``in_channels`` and ``latent_dim`` are set automatically from the
        encoder type; explicit overrides take precedence.

    Returns
    -------
    tuple[UNetBelief, list[dict]]
        ``(model, history)`` — the trained model loaded with its best weights,
        and the per-epoch training history.
    """
    check_device(device)

    root = Path(storage_root).expanduser().resolve()

    jepa_path, ae_path, distributions, formation_geo, discovery = (
        resolve_resource_paths(root)
    )
    check_encoder_path(borehole_encoder, jepa_path, ae_path)
    check_sim_paths(distributions, formation_geo)

    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.is_absolute():
        ckpt_dir = root / ckpt_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    plot_dir_resolved: Path | None = None
    if plot_dir is not None:
        plot_dir_resolved = Path(plot_dir)
        if not plot_dir_resolved.is_absolute():
            plot_dir_resolved = root / plot_dir_resolved

    resources, latent_dim = load_decision_resources(
        borehole_encoder=borehole_encoder,
        jepa_path=jepa_path,
        ae_path=ae_path,
        distributions_path=distributions,
        formation_geometry_path=formation_geo,
        discovery_prior_path=discovery,
        device=device,
    )

    # in_channels and latent_dim are derived from the encoder;
    # explicit user overrides take precedence.
    encoder_defaults = {"in_channels": 2 + latent_dim, "latent_dim": latent_dim}
    cfg = build_training_config(debug, {**encoder_defaults, **overrides})
    cfg.borehole_encoder = borehole_encoder

    print(f"\nborehole_encoder : {borehole_encoder}")
    print(f"latent_dim       : {latent_dim}")
    print(f"in_channels      : {cfg.in_channels}")
    print(f"Training config  :\n{cfg}\n")

    model, _ = train_neural_belief(
        resources=resources,
        cfg=cfg,
        device=device,
        checkpoint_dir=ckpt_dir,
        plot_dir=plot_dir_resolved,
        verbose=True,
    )

    # Smoke test: verify the saved checkpoint loads cleanly.
    _, _, _, history = load_belief_checkpoint(
        ckpt_dir / "belief_best.pt", device=device
    )
    print(
        f"\nSmoke test passed: checkpoint loaded with {len(history)} epoch(s) of history."
    )

    export_history(history, ckpt_dir)
    save_experiment_config(ckpt_dir, cfg, cache_path=None, sim_cfg=None)

    return model, history


def compare_belief_encoders_from_colab(
    storage_root: str | Path,
    output_root: str | Path = "checkpoints/belief_comparison",
    device: str = "cuda",
    borehole_encoder_variants: tuple[str, ...] = ("none", "autoencoder", "jepa"),
    map_encoder_variants: tuple[str, ...] = ("unet",),
    debug: bool = False,
    map_pool_path: str | Path = "data/train_maps",
    **overrides,
) -> pd.DataFrame:
    """Train and compare all combinations of borehole and map encoder variants.

    Iterates over every combination of ``borehole_encoder_variants`` ×
    ``map_encoder_variants``.  For each combination both a non-shuffled and a
    shuffled-latent run are executed automatically, giving a built-in spatial
    control.  Resources are loaded once per borehole encoder; datasets are built
    once per (borehole encoder, shuffled) pair and reused across map encoders.

    The map pool must already exist at ``map_pool_path``, generated in advance
    by ``generate_training_maps.py``.

    Seed defaults to 42; pass ``seed=N`` in ``overrides`` to change it
    consistently across all runs.

    Example
    -------
    >>> from decision_simulator.neural_belief.colab import compare_belief_encoders_from_colab

    # Compare UNet vs Transformer, all three borehole encoders:
    >>> df = compare_belief_encoders_from_colab(
    ...     storage_root="/content/drive/MyDrive/thesis",
    ...     borehole_encoder_variants=("none", "jepa"),
    ...     map_encoder_variants=("unet", "transformer"),
    ...     debug=True,
    ... )
    >>> print(df)

    Parameters
    ----------
    storage_root
        Absolute root for all data and checkpoint paths.
    output_root
        Parent directory for per-run subdirectories, resolved against
        ``storage_root`` if relative.  Each run is saved under
        ``<output_root>/<map_encoder>_[shuffled_]<borehole_encoder>/``.
    device
        ``"cuda"`` or ``"cpu"``.
    borehole_encoder_variants
        Which borehole encoders to evaluate: ``"none"``, ``"autoencoder"``,
        and/or ``"jepa"``.  Controls which latent embeddings are fed to the
        map encoder.  Both shuffled and non-shuffled embeddings are always run.
    map_encoder_variants
        Which map encoder models to evaluate: ``"unet"`` and/or
        ``"transformer"``.
    debug
        Run all combinations with a minimal config (2 maps, 2 epochs) for a
        quick end-to-end check.
    map_pool_path
        Path to the pre-generated npz map pool directory, resolved against
        ``storage_root`` if relative.
    **overrides
        Forwarded to every training run.  ``in_channels`` and ``latent_dim``
        are always derived from the encoder and cannot be overridden here.

    Returns
    -------
    pd.DataFrame
        One row per run with columns ``map_encoder``, ``borehole_encoder``,
        ``shuffled``, ``best_epoch``, ``best_val_mse``, ``best_val_mae``,
        ``best_val_corr``, plus per-drill-bin metrics.
        Also written to ``<output_root>/comparison_summary.csv``.
    """
    check_device(device)
    root = Path(storage_root).expanduser().resolve()

    out_root = Path(output_root)
    if not out_root.is_absolute():
        out_root = root / out_root
    out_root.mkdir(parents=True, exist_ok=True)

    resolved_pool_path = Path(map_pool_path)
    if not resolved_pool_path.is_absolute():
        resolved_pool_path = root / resolved_pool_path

    jepa_path, ae_path, distributions, formation_geo, discovery = (
        resolve_resource_paths(root)
    )
    check_sim_paths(distributions, formation_geo)

    # Both config types share the same dataset fields, so UNet config suffices here.
    base_cfg = build_training_config(
        debug,
        {k: v for k, v in overrides.items() if k not in ("in_channels", "latent_dim")},
    )

    store = NpzMapCacheStore(resolved_pool_path)
    train_cache, val_cache = store.load_train_val_split(
        n_train_maps=base_cfg.n_train_maps,
        n_val_maps=base_cfg.n_val_maps,
    )

    # ---- pre-build all datasets -------------------------------------------------
    # Resources and datasets depend only on (borehole_encoder, is_shuffled), not
    # on the map encoder, so we build them once and reuse across map encoders.
    resources_by_encoder: dict[str, tuple] = {}
    datasets_by_key: dict[tuple, tuple] = {}

    for borehole_encoder in borehole_encoder_variants:
        check_encoder_path(borehole_encoder, jepa_path, ae_path)
        resources, latent_dim = load_decision_resources(
            borehole_encoder=borehole_encoder,
            jepa_path=jepa_path,
            ae_path=ae_path,
            distributions_path=distributions,
            formation_geometry_path=formation_geo,
            discovery_prior_path=discovery,
            device=device,
        )
        resources_by_encoder[borehole_encoder] = (resources, latent_dim)

        for is_shuffled in [False, True]:
            print(f"\nBuilding {'shuffled ' if is_shuffled else ''}datasets  borehole_encoder={borehole_encoder} ...")
            train_ds = build_dataset_from_cache(
                train_cache, resources, device,
                verbose=True,
                samples_per_map=base_cfg.samples_per_map,
                shuffle_latents=is_shuffled,
                shuffle_seed=base_cfg.seed,
            )
            val_ds = build_dataset_from_cache(
                val_cache, resources, device,
                verbose=True,
                samples_per_map=base_cfg.val_samples_per_map,
                shuffle_latents=is_shuffled,
                shuffle_seed=base_cfg.seed + 10000,
            )
            print(f"  train={len(train_ds)}  val={len(val_ds)}")
            datasets_by_key[(borehole_encoder, is_shuffled)] = (train_ds, val_ds)

    # ---- training loop ----------------------------------------------------------
    rows: list[dict] = []

    for map_encoder in map_encoder_variants:
        for borehole_encoder in borehole_encoder_variants:
            resources, latent_dim = resources_by_encoder[borehole_encoder]

            for is_shuffled in [False, True]:
                shuffle_label = "shuffled_" if is_shuffled else ""
                run_label = f"{map_encoder}_{shuffle_label}{borehole_encoder}"
                train_ds, val_ds = datasets_by_key[(borehole_encoder, is_shuffled)]

                print(f"\n{'=' * 60}")
                print(f"  map_encoder={map_encoder}  borehole_encoder={borehole_encoder}  shuffled={is_shuffled}")
                print(f"{'=' * 60}")

                ckpt_dir = out_root / run_label
                ckpt_dir.mkdir(parents=True, exist_ok=True)

                is_transformer = map_encoder == "transformer"
                if is_transformer:
                    cfg = build_map_belief_training_config(
                        debug,
                        {"latent_dim": latent_dim, **{k: v for k, v in overrides.items() if k != "in_channels"}},
                    )
                else:
                    cfg = build_training_config(
                        debug,
                        {"in_channels": 2 + latent_dim, "latent_dim": latent_dim, **overrides},
                    )
                cfg.borehole_encoder = f"{shuffle_label}{borehole_encoder}"

                print(f"  latent_dim    : {latent_dim}  in_channels : {cfg.in_channels}")
                if is_shuffled:
                    print("  (shuffled-latent control: spatial latent positions are permuted)")

                if is_transformer:
                    model, normalizer = train_map_belief(
                        resources=resources, cfg=cfg, device=device,
                        checkpoint_dir=ckpt_dir, plot_dir=ckpt_dir / "plots",
                        verbose=True, train_ds=train_ds, val_ds=val_ds,
                    )
                    _, _, _, history = load_map_belief_checkpoint(
                        ckpt_dir / "map_belief_best.pt", device=device
                    )
                else:
                    model, normalizer = train_neural_belief(
                        resources=resources, cfg=cfg, device=device,
                        checkpoint_dir=ckpt_dir, plot_dir=ckpt_dir / "plots",
                        verbose=True, train_ds=train_ds, val_ds=val_ds,
                    )
                    _, _, _, history = load_belief_checkpoint(
                        ckpt_dir / "belief_best.pt", device=device
                    )

                print(f"\nSmoke test passed: {len(history)} epoch(s) in history.")
                export_history(history, ckpt_dir)
                save_experiment_config(ckpt_dir, cfg, cache_path=resolved_pool_path, sim_cfg=None)

                best = min(history, key=lambda r: r["val_mse"])
                bin_metrics = validate_by_drill_bins(model, val_ds, normalizer, device)
                rows.append(
                    {
                        "map_encoder": map_encoder,
                        "borehole_encoder": borehole_encoder,
                        "shuffled": is_shuffled,
                        "best_epoch": best["epoch"],
                        "best_val_mse": best["val_mse"],
                        "best_val_mae": best["val_mae"],
                        "best_val_corr": best["val_corr"],
                        **bin_metrics,
                    }
                )

    df = pd.DataFrame(rows)

    csv_path = out_root / "comparison_summary.csv"
    df.to_csv(csv_path, index=False)

    print(f"\n{'=' * 60}")
    print("  Comparison summary")
    print(f"{'=' * 60}")
    print(df.to_string(index=False))
    print(f"\nSummary -> {csv_path}")

    return df
