from __future__ import annotations

import json
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

from .datasets import GeologicalBeliefDataset
from .map_cache import NpzMapCacheStore
from .map_hdf5 import HDF5MapStore
from .models.map_encoders.unet_belief import UNetBelief

# ---------------------------------------------------------------------------
# Data source flag
# ---------------------------------------------------------------------------
# True  → read maps from a pre-built HDF5 shard  (HDF5MapStore,    default)
# False → read maps from an npz pool directory    (NpzMapCacheStore, legacy)
USE_HDF5_STORE: bool = True

from .training import (
    NeuralBeliefTrainingConfig,
    MapBeliefTrainingConfig,
    E2EMapBeliefTrainingConfig,
    E2EMapDataset,
    build_training_config,
    export_history,
    save_experiment_config,
    load_belief_checkpoint,
    load_map_belief_checkpoint,
    train_neural_belief,
    train_map_belief,
    train_end_to_end_map_belief,
    load_e2e_map_belief_checkpoint,
)

_DEBUG_UNET: dict = {
    "n_train_maps": 2,
    "samples_per_map": 2,
    "n_val_maps": 1,
    "val_samples_per_map": 2,
    "n_epochs": 2,
    "batch_size": 2,
    "base_channels": 16,
    "n_sequences_per_map": 2,
    "prefix_steps": [1, 3, 5],
}
_DEBUG_MAP: dict = {
    "n_train_maps": 2,
    "samples_per_map": 2,
    "n_val_maps": 1,
    "val_samples_per_map": 2,
    "n_epochs": 2,
    "batch_size": 2,
    "d_model": 64,
    "n_heads": 4,
    "n_encoder_layers": 1,
    "d_ff": 128,
    "head_hidden_dim": 32,
    "n_sequences_per_map": 2,
    "prefix_steps": [1, 3, 5],
}
_DEBUG_E2E: dict = {
    "n_train_maps": 2,
    "samples_per_map": 4,
    "n_val_maps": 1,
    "val_samples_per_map": 4,
    "n_epochs": 2,
    "batch_size": 2,
    "d_model": 64,
    "n_heads": 4,
    "n_encoder_layers": 1,
    "d_ff": 128,
    "head_hidden_dim": 32,
    "bh_d_model": 32,
    "bh_n_heads": 4,
    "bh_n_layers": 1,
    "latent_dim": 32,
}

# Fields present in NeuralBeliefTrainingConfig but not in MapBeliefTrainingConfig.
# These are silently dropped when building a transformer config from **overrides.
_UNET_ONLY_OVERRIDE_FIELDS = frozenset(
    {
        "in_channels",
        "base_channels",
        "use_latent_pca",
        "use_coordinate_channels",
    }
)


def train_belief_from_colab(
    storage_root: str | Path,
    checkpoint_dir: str | Path,
    map_pool_path: str | Path = "data/train_maps",
    device: str = "cuda",
    debug: bool = False,
    borehole_encoder: Literal["jepa", "autoencoder", "none"] = "jepa",
    plot_dir: str | Path | None = None,
    n_orebodies: int | None = None,
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
    cfg: NeuralBeliefTrainingConfig = build_training_config(
        NeuralBeliefTrainingConfig,
        {**(_DEBUG_UNET if debug else {}), **encoder_defaults, **overrides},
    )
    cfg.borehole_encoder = borehole_encoder

    print(f"\nborehole_encoder : {borehole_encoder}")
    print(f"latent_dim       : {latent_dim}")
    print(f"in_channels      : {cfg.in_channels}")
    print(f"Training config  :\n{cfg}\n")

    resolved_pool_path = Path(map_pool_path)
    if not resolved_pool_path.is_absolute():
        resolved_pool_path = root / resolved_pool_path
    store = HDF5MapStore(resolved_pool_path) if USE_HDF5_STORE else NpzMapCacheStore(resolved_pool_path)

    train_cache, val_cache = store.load_map_data(
        n_train_maps=cfg.n_train_maps,
        n_val_maps=cfg.n_val_maps,
        n_orebodies=n_orebodies,
        seed=overrides.get("seed", cfg.seed),
    )
    train_ds = train_cache.build_geo_train_maps(resources, device, verbose=True)
    val_ds = val_cache.build_geo_train_maps(resources, device, verbose=True)

    model, _ = train_neural_belief(
        cfg=cfg,
        device=device,
        checkpoint_dir=ckpt_dir,
        plot_dir=plot_dir_resolved,
        verbose=True,
        train_ds=train_ds,
        val_ds=val_ds,
    )

    # Smoke test: verify the saved checkpoint loads cleanly.
    _, _, _, history = load_belief_checkpoint(
        ckpt_dir / "belief_best.pt", device=device
    )
    print(
        f"\nSmoke test passed: checkpoint loaded with {len(history)} epoch(s) of history."
    )

    export_history(history, ckpt_dir)
    save_experiment_config(ckpt_dir, cfg, cache_path=None)

    return model, history


def compare_belief_encoders_from_colab(
    storage_root: str | Path,
    output_root: str | Path = "checkpoints/belief_comparison",
    device: str = "cuda",
    borehole_encoder_variants: tuple[str, ...] = ("none", "autoencoder", "jepa"),
    map_encoder_variants: tuple[str, ...] = ("unet",),
    debug: bool = False,
    map_pool_path: str | Path = "data/train_maps",
    n_orebodies: int | None = None,
    run_shuffled: bool = False,
    use_sequential_dataset: bool = False,
    n_sequences_per_map: int = 3,
    prefix_steps: list[int] | None = None,
    sequential_seed: int = 42,
    training_variants: list[dict] | None = None,
    **overrides,
) -> pd.DataFrame:
    """Train and compare all combinations of borehole and map encoder variants.

    Iterates over every combination of ``borehole_encoder_variants`` ×
    ``map_encoder_variants``.  When ``run_shuffled=True``, both a non-shuffled
    and a shuffled-latent run are executed, giving a built-in spatial control.
    Resources are loaded once per borehole encoder; datasets are built
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
    n_orebodies
        Stratify the map sample by ore body count.  ``None`` (default) uses
        a contiguous slice of the pool.  Set to 1, 2, or 3 to draw maps
        in equal proportions across body counts 0…n_orebodies.  Requires
        ``n_bodies_index.npy`` to be present in the pool (written by
        ``generate_training_maps.py``).
    training_variants
        Optional list of per-variant override dicts.  Each dict is merged
        on top of ``overrides`` for that variant's training run, producing
        one additional dimension in the comparison grid.  A ``"_label"`` key
        in the dict names the variant (used in directory names and the
        ``variant`` column); if absent the variant is labelled ``v0``,
        ``v1``, …  When ``None`` (default), a single run is executed per
        combination (backward-compatible).

        Example — compare FP-penalty on vs off::

            training_variants=[
                {"_label": "fp_off"},
                {"_label": "fp_on",
                 "use_false_positive_penalty": True,
                 "false_positive_weight": 0.05},
            ]
    **overrides
        Forwarded to every training run.  ``in_channels`` and ``latent_dim``
        are always derived from the encoder and cannot be overridden here.

    Returns
    -------
    pd.DataFrame
        One row per run with columns ``map_encoder``, ``borehole_encoder``,
        ``shuffled``, ``variant`` (when ``training_variants`` is set),
        ``best_epoch``, ``best_val_mse``, ``best_val_mae``, ``best_val_corr``,
        plus per-drill-bin metrics and no-ore FP metrics.
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
        NeuralBeliefTrainingConfig,
        {
            **(_DEBUG_UNET if debug else {}),
            **{
                k: v
                for k, v in overrides.items()
                if k not in ("in_channels", "latent_dim")
            },
        },
    )

    resolved_steps = prefix_steps or [1, 2, 3, 5, 8, 10, 15]

    # Overrides injected into every per-run config when sequential mode is on.
    seq_cfg_overrides: dict = (
        {
            "use_sequential_dataset": True,
            "n_sequences_per_map": n_sequences_per_map,
            "prefix_steps": resolved_steps,
            "sequential_seed": sequential_seed,
        }
        if use_sequential_dataset
        else {}
    )

    store = HDF5MapStore(resolved_pool_path) if USE_HDF5_STORE else NpzMapCacheStore(resolved_pool_path)

    train_cache, val_cache = store.load_map_data(
        n_train_maps=base_cfg.n_train_maps,
        n_val_maps=base_cfg.n_val_maps,
        n_orebodies=n_orebodies,
        seed=overrides.get("seed", 42),
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

        for is_shuffled in ([False, True] if run_shuffled else [False]):
            shuffle_tag = "shuffled " if is_shuffled else ""
            mode_tag = "sequential " if use_sequential_dataset else ""
            print(
                f"\nBuilding {mode_tag}{shuffle_tag}datasets  borehole_encoder={borehole_encoder} ..."
            )
            if use_sequential_dataset:
                train_ds = train_cache.build_geo_train_maps(
                    resources,
                    device,
                    n_sequences_per_map=n_sequences_per_map,
                    max_drills=base_cfg.max_drills,
                    prefix_steps=resolved_steps,
                    seed=sequential_seed,
                    verbose=True,
                    shuffle_latents=is_shuffled,
                    shuffle_seed=base_cfg.seed,
                )
                val_ds = val_cache.build_geo_train_maps(
                    resources,
                    device,
                    n_sequences_per_map=n_sequences_per_map,
                    max_drills=base_cfg.max_drills,
                    prefix_steps=resolved_steps,
                    seed=sequential_seed + 1,
                    verbose=True,
                    shuffle_latents=is_shuffled,
                    shuffle_seed=base_cfg.seed + 10000,
                )
            else:
                train_ds = train_cache.build_geo_train_maps(
                    resources,
                    device,
                    verbose=True,
                    shuffle_latents=is_shuffled,
                    shuffle_seed=base_cfg.seed,
                )
                val_ds = val_cache.build_geo_train_maps(
                    resources,
                    device,
                    verbose=True,
                    shuffle_latents=is_shuffled,
                    shuffle_seed=base_cfg.seed + 10000,
                )
            print(f"  train={len(train_ds)}  val={len(val_ds)}")
            datasets_by_key[(borehole_encoder, is_shuffled)] = (train_ds, val_ds)

    # ---- training loop ----------------------------------------------------------
    _variant_list = training_variants if training_variants is not None else [{}]
    _multi_variant = training_variants is not None

    rows: list[dict] = []

    for vi, variant_overrides in enumerate(_variant_list):
        variant_label = str(variant_overrides.get("_label", f"v{vi}"))
        # Pure training overrides — strip the internal _label key
        variant_train = {k: v for k, v in variant_overrides.items() if k != "_label"}

        for map_encoder in map_encoder_variants:
            for borehole_encoder in borehole_encoder_variants:
                resources, latent_dim = resources_by_encoder[borehole_encoder]

                for is_shuffled in ([False, True] if run_shuffled else [False]):
                    shuffle_label = "shuffled_" if is_shuffled else ""
                    base_run_label = f"{map_encoder}_{shuffle_label}{borehole_encoder}"
                    run_label = (
                        f"{base_run_label}_{variant_label}"
                        if _multi_variant
                        else base_run_label
                    )
                    train_ds, val_ds = datasets_by_key[(borehole_encoder, is_shuffled)]

                    print(f"\n{'=' * 60}")
                    print(
                        f"  map_encoder={map_encoder}  borehole_encoder={borehole_encoder}"
                        f"  shuffled={is_shuffled}"
                        + (f"  variant={variant_label}" if _multi_variant else "")
                    )
                    print(f"{'=' * 60}")

                    ckpt_dir = out_root / run_label
                    ckpt_dir.mkdir(parents=True, exist_ok=True)

                    # Effective overrides = shared overrides merged with variant-specific ones
                    effective_overrides = {**overrides, **variant_train}

                    is_transformer = map_encoder == "transformer"
                    if is_transformer:
                        cfg = build_training_config(
                            MapBeliefTrainingConfig,
                            {
                                **(_DEBUG_MAP if debug else {}),
                                "latent_dim": latent_dim,
                                **{
                                    k: v
                                    for k, v in effective_overrides.items()
                                    if k not in _UNET_ONLY_OVERRIDE_FIELDS
                                },
                                **seq_cfg_overrides,
                            },
                        )
                    else:
                        cfg = build_training_config(
                            NeuralBeliefTrainingConfig,
                            {
                                **(_DEBUG_UNET if debug else {}),
                                "in_channels": 2 + latent_dim,
                                "latent_dim": latent_dim,
                                **effective_overrides,
                                **seq_cfg_overrides,
                            },
                        )
                    cfg.borehole_encoder = f"{shuffle_label}{borehole_encoder}"

                    print(
                        f"  latent_dim    : {latent_dim}  in_channels : {cfg.in_channels}"
                    )
                    if is_shuffled:
                        print(
                            "  (shuffled-latent control: spatial latent positions are permuted)"
                        )

                    # Wrap in new objects so normalisation applied during training
                    # does not mutate the cached datasets for subsequent runs.
                    train_ds_run = GeologicalBeliefDataset(
                        train_ds.inputs,
                        train_ds.targets,
                        train_ds.drill_counts,
                        train_ds.metadata,
                    )
                    val_ds_run = GeologicalBeliefDataset(
                        val_ds.inputs,
                        val_ds.targets,
                        val_ds.drill_counts,
                        val_ds.metadata,
                    )

                    if is_transformer:
                        train_map_belief(
                            cfg=cfg,
                            device=device,
                            checkpoint_dir=ckpt_dir,
                            plot_dir=ckpt_dir / "plots",
                            verbose=True,
                            train_ds=train_ds_run,
                            val_ds=val_ds_run,
                        )
                        _, _, _, history = load_map_belief_checkpoint(
                            ckpt_dir / "map_belief_best.pt", device=device
                        )
                    else:
                        train_neural_belief(
                            cfg=cfg,
                            device=device,
                            checkpoint_dir=ckpt_dir,
                            plot_dir=ckpt_dir / "plots",
                            verbose=True,
                            train_ds=train_ds_run,
                            val_ds=val_ds_run,
                        )
                        _, _, _, history = load_belief_checkpoint(
                            ckpt_dir / "belief_best.pt", device=device
                        )

                    print(f"\nSmoke test passed: {len(history)} epoch(s) in history.")
                    export_history(history, ckpt_dir)
                    save_experiment_config(ckpt_dir, cfg, cache_path=resolved_pool_path)

                    best = min(history, key=lambda r: r["val_mse"])
                    bin_path = ckpt_dir / "val_metrics_by_drills.json"
                    bin_metrics = (
                        json.loads(bin_path.read_text()) if bin_path.exists() else {}
                    )
                    step_path = ckpt_dir / "val_metrics_by_step.json"
                    step_metrics_flat: dict = {}
                    if step_path.exists():
                        for step_str, m in json.loads(step_path.read_text()).items():
                            for metric, val in m.items():
                                step_metrics_flat[f"step{step_str}_{metric}"] = val
                    no_ore_path = ckpt_dir / "val_metrics_no_ore.json"
                    no_ore_flat = (
                        json.loads(no_ore_path.read_text())
                        if no_ore_path.exists()
                        else {}
                    )
                    row: dict = {
                        "map_encoder": map_encoder,
                        "borehole_encoder": borehole_encoder,
                        "shuffled": is_shuffled,
                        "best_epoch": best["epoch"],
                        "best_val_mse": best["val_mse"],
                        "best_val_mae": best["val_mae"],
                        "best_val_corr": best["val_corr"],
                        **bin_metrics,
                        **step_metrics_flat,
                        **no_ore_flat,
                    }
                    if _multi_variant:
                        row["variant"] = variant_label
                    rows.append(row)

    df = pd.DataFrame(rows)

    csv_path = out_root / "comparison_summary.csv"
    df.to_csv(csv_path, index=False)

    print(f"\n{'=' * 60}")
    print("  Comparison summary")
    print(f"{'=' * 60}")
    print(df.to_string(index=False))
    print(f"\nSummary -> {csv_path}")

    return df


def train_sequential_belief_from_colab(
    storage_root: str | Path,
    checkpoint_dir: str | Path,
    map_pool_path: str | Path = "data/train_maps",
    device: str = "cuda",
    debug: bool = False,
    borehole_encoder: Literal["jepa", "autoencoder", "none"] = "jepa",
    map_encoder: Literal["unet", "transformer"] = "unet",
    n_train_maps: int = 50,
    n_val_maps: int = 10,
    n_sequences_per_map: int = 3,
    prefix_steps: list[int] | None = None,
    sequential_seed: int = 42,
    plot_dir: str | Path | None = None,
    n_orebodies: int | None = None,
    **overrides,
) -> tuple[object, list[dict]]:
    """Train a neural geological belief updater with sequential drill observations.

    For each geological map, generates ``n_sequences_per_map`` ordered drill
    sequences and creates prefix samples at each checkpoint in ``prefix_steps``.
    This teaches the model to refine its belief as more drilling evidence
    accumulates, rather than treating each sample as an independent observation.

    Example
    -------
    >>> from decision_simulator.neural_belief.colab import train_sequential_belief_from_colab

    >>> model, history = train_sequential_belief_from_colab(
    ...     storage_root="/content/drive/MyDrive/thesis",
    ...     checkpoint_dir="checkpoints/belief_sequential",
    ...     map_pool_path="data/train_maps",
    ...     device="cuda",
    ...     borehole_encoder="jepa",
    ...     n_sequences_per_map=3,
    ...     prefix_steps=[1, 2, 3, 5, 8, 10, 15],
    ...     debug=False,
    ... )

    Parameters
    ----------
    storage_root
        Absolute root for all data and checkpoint paths.
    checkpoint_dir
        Where checkpoints and metric files are saved.  Relative paths are
        resolved against ``storage_root``.
    map_pool_path
        Path to the pre-generated npz map pool directory.  Relative paths are
        resolved against ``storage_root``.
    device
        ``"cuda"`` or ``"cpu"``.
    debug
        If True, use a minimal config (2 maps, 2 epochs) to verify the full
        pipeline end-to-end without a full training run.
    borehole_encoder
        Which encoder provides latent embeddings per borehole:
        ``"jepa"``, ``"autoencoder"``, or ``"none"``.
    map_encoder
        Which map encoder model to train: ``"unet"`` or ``"transformer"``.
    n_train_maps
        Number of maps to use for training.
    n_val_maps
        Number of maps to use for validation.
    n_sequences_per_map
        Number of independent drill orderings per map.
    prefix_steps
        Drill-count checkpoints at which to create samples.
        Defaults to ``[1, 2, 3, 5, 8, 10, 15]``.
    sequential_seed
        RNG seed for drill sequence generation.
    plot_dir
        If given, save validation plots here after training.
    n_orebodies
        Stratify the map sample by ore body count (requires
        ``n_bodies_index.npy`` in the pool).  ``None`` uses a contiguous slice.
    **overrides
        Any field of the training config by name.

    Returns
    -------
    tuple[model, history]
        The trained model loaded with its best weights and the per-epoch history.
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

    resolved_pool = Path(map_pool_path)
    if not resolved_pool.is_absolute():
        resolved_pool = root / resolved_pool

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

    # In debug mode use a small map count for both cache loading and config.
    effective_n_train = 2 if debug else n_train_maps
    effective_n_val = 1 if debug else n_val_maps

    store = HDF5MapStore(resolved_pool) if USE_HDF5_STORE else NpzMapCacheStore(resolved_pool)

    train_cache, val_cache = store.load_map_data(
        n_train_maps=effective_n_train,
        n_val_maps=effective_n_val,
        n_orebodies=n_orebodies,
        seed=overrides.get("seed", 42),
    )

    resolved_steps = prefix_steps or [1, 2, 3, 5, 8, 10, 15]

    is_transformer = map_encoder == "transformer"

    # n_train_maps / n_val_maps are intentionally excluded: the sequential path
    # builds the dataset directly from the cache, so those config fields are unused.
    sequential_overrides = {
        "use_sequential_dataset": True,
        "n_sequences_per_map": n_sequences_per_map,
        "prefix_steps": resolved_steps,
        "sequential_seed": sequential_seed,
    }

    if is_transformer:
        cfg = build_training_config(
            MapBeliefTrainingConfig,
            {
                **(_DEBUG_MAP if debug else {}),
                "latent_dim": latent_dim,
                **{
                    k: v
                    for k, v in overrides.items()
                    if k not in _UNET_ONLY_OVERRIDE_FIELDS
                },
                **sequential_overrides,
            },
        )
        cfg.borehole_encoder = borehole_encoder
        print(f"\nborehole_encoder : {borehole_encoder}")
        print("map_encoder      : transformer")
        print(f"latent_dim       : {latent_dim}")
        print(f"n_sequences      : {cfg.n_sequences_per_map}")
        print(f"prefix_steps     : {cfg.prefix_steps}")
        print(f"Training config  :\n{cfg}\n")
    else:
        encoder_defaults = {"in_channels": 2 + latent_dim, "latent_dim": latent_dim}
        cfg = build_training_config(
            NeuralBeliefTrainingConfig,
            {
                **(_DEBUG_UNET if debug else {}),
                **encoder_defaults,
                **overrides,
                **sequential_overrides,
            },
        )
        cfg.borehole_encoder = borehole_encoder
        print(f"\nborehole_encoder : {borehole_encoder}")
        print("map_encoder      : unet")
        print(f"latent_dim       : {latent_dim}")
        print(f"in_channels      : {cfg.in_channels}")
        print(f"n_sequences      : {cfg.n_sequences_per_map}")
        print(f"prefix_steps     : {cfg.prefix_steps}")
        print(f"Training config  :\n{cfg}\n")

    print("Building sequential training dataset ...")
    train_ds = train_cache.build_geo_train_maps(
        resources,
        device,
        n_sequences_per_map=n_sequences_per_map,
        max_drills=cfg.max_drills,
        prefix_steps=resolved_steps,
        seed=sequential_seed,
        verbose=True,
    )
    print("Building sequential validation dataset ...")
    val_ds = val_cache.build_geo_train_maps(
        resources,
        device,
        n_sequences_per_map=n_sequences_per_map,
        max_drills=cfg.max_drills,
        prefix_steps=resolved_steps,
        seed=sequential_seed + 1,
        verbose=True,
    )

    if is_transformer:

        model, _ = train_map_belief(
            cfg=cfg,
            device=device,
            checkpoint_dir=ckpt_dir,
            plot_dir=plot_dir_resolved,
            verbose=True,
            train_ds=train_ds,
            val_ds=val_ds,
        )
        _, _, _, history = load_map_belief_checkpoint(
            ckpt_dir / "map_belief_best.pt", device=device
        )
    else:
        model, _ = train_neural_belief(
            cfg=cfg,
            device=device,
            checkpoint_dir=ckpt_dir,
            plot_dir=plot_dir_resolved,
            verbose=True,
            train_ds=train_ds,
            val_ds=val_ds,
        )
        _, _, _, history = load_belief_checkpoint(
            ckpt_dir / "belief_best.pt", device=device
        )

    print(
        f"\nSmoke test passed: checkpoint loaded with {len(history)} epoch(s) of history."
    )
    export_history(history, ckpt_dir)
    save_experiment_config(ckpt_dir, cfg, cache_path=resolved_pool)

    return model, history


def train_end_to_end_from_colab(
    storage_root: str | Path,
    checkpoint_dir: str | Path,
    device: str = "cuda",
    debug: bool = False,
    norm_stats_from: Literal["jepa", "autoencoder", "none"] = "jepa",
    map_pool_path: str | Path = "data/dataset_complete",
    n_orebodies: int | None = None,
    use_sequential_dataset: bool = False,
    n_sequences_per_map: int = 3,
    prefix_steps: list[int] | None = None,
    **overrides,
) -> tuple[object, list[dict]]:
    """Train the end-to-end map belief transformer from a Colab notebook.

    Trains EndToEndMapBeliefTransformer with a full-map MSE reconstruction
    objective.  The borehole encoder and map belief transformer are trained
    jointly from scratch — no pre-trained JEPA or autoencoder weights are
    required.  The ``norm_stats_from`` parameter controls which checkpoint's
    per-variable normalization statistics are used to standardise raw borehole
    inputs.

    Example
    -------
    >>> from decision_simulator.neural_belief.colab import train_end_to_end_from_colab

    # Default: use JEPA norm stats, train from scratch
    >>> model, history = train_end_to_end_from_colab(
    ...     storage_root="/content/drive/MyDrive/thesis",
    ...     checkpoint_dir="checkpoints/e2e_map",
    ...     device="cuda",
    ...     debug=True,
    ... )

    Parameters
    ----------
    storage_root
        Absolute root for all data and checkpoint paths.  Relative paths are
        resolved against this directory.
    checkpoint_dir
        Where ``e2e_map_belief_best.pt``, ``e2e_map_belief_last.pt``, and
        ``training_history.{json,csv}`` are saved.  Relative paths are
        resolved against ``storage_root``.
    device
        ``"cuda"`` or ``"cpu"``.  Raises a clear error when CUDA is requested
        but unavailable.
    debug
        If True, use a minimal config (2 maps, 2 epochs, tiny model) to
        verify the full pipeline end-to-end without a full training run.
    norm_stats_from
        Which encoder checkpoint to load per-variable normalization statistics
        from.  The encoder itself is NOT used.

        ``"jepa"``        — JEPA checkpoint norm stats (recommended)
        ``"autoencoder"`` — Autoencoder checkpoint norm stats
        ``"none"``        — skip normalization (raw geological values)
    **overrides
        Any field of :class:`E2EMapBeliefTrainingConfig` by name.  Common
        overrides:

        * ``n_epochs=100`` — train longer
        * ``use_false_positive_penalty=True`` — penalise false positives
        * ``batch_size=4`` — reduce if GPU memory is tight

    Returns
    -------
    tuple[EndToEndMapBeliefTransformer, list[dict]]
        ``(model, history)`` — the trained model loaded with its best weights,
        and the per-epoch training history.
    """
    check_device(device)

    root = Path(storage_root).expanduser().resolve()

    jepa_path, ae_path, distributions, formation_geo, discovery = (
        resolve_resource_paths(root)
    )
    check_encoder_path(norm_stats_from, jepa_path, ae_path)
    check_sim_paths(distributions, formation_geo)

    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.is_absolute():
        ckpt_dir = root / ckpt_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    resolved_pool = Path(map_pool_path)
    if not resolved_pool.is_absolute():
        resolved_pool = root / resolved_pool

    print(
        f"\nLoading norm stats from '{norm_stats_from}' checkpoint "
        "(encoder is not used — model trains its own borehole encoder) …"
    )
    resources, _ = load_decision_resources(
        borehole_encoder=norm_stats_from,
        jepa_path=jepa_path,
        ae_path=ae_path,
        distributions_path=distributions,
        formation_geometry_path=formation_geo,
        discovery_prior_path=discovery,
        device=device,
    )

    seq_overrides: dict = {}
    if use_sequential_dataset:
        seq_overrides = {
            "use_sequential_dataset": True,
            "n_sequences_per_map": n_sequences_per_map,
            "prefix_steps": prefix_steps or [1, 2, 3, 5, 8, 10, 15],
        }

    cfg = build_training_config(
        E2EMapBeliefTrainingConfig,
        {**(_DEBUG_E2E if debug else {}), **overrides, **seq_overrides},
    )

    # ---- load map cache -------------------------------------------------------
    store = HDF5MapStore(resolved_pool) if USE_HDF5_STORE else NpzMapCacheStore(resolved_pool)
    seed = overrides.get("seed", cfg.seed)

    train_cache, val_cache = store.load_map_data(
        n_train_maps=cfg.n_train_maps,
        n_val_maps=cfg.n_val_maps,
        n_orebodies=n_orebodies,
        seed=seed,
    )

    cfg.n_x = train_cache.n_x
    cfg.n_y = train_cache.n_y

    print(f"\nnorm_stats_from  : {norm_stats_from}")
    print(f"map_pool_path    : {resolved_pool}")
    print(f"n_orebodies      : {n_orebodies}")
    print(f"n_train_maps     : {cfg.n_train_maps}  n_val_maps: {cfg.n_val_maps}")
    print(f"latent_dim       : {cfg.latent_dim}")
    print(f"grid             : {cfg.n_x} × {cfg.n_y}")
    print(f"sequential       : {cfg.use_sequential_dataset}")
    if cfg.use_sequential_dataset:
        print(f"n_sequences      : {cfg.n_sequences_per_map}")
        print(f"prefix_steps     : {cfg.prefix_steps}")
    print(f"Training config  :\n{cfg}\n")

    print("Building training dataset ...")
    train_ds = E2EMapDataset.from_cache(
        train_cache, resources, cfg, verbose=True, is_val=False
    )
    print("Building validation dataset ...")
    val_ds = E2EMapDataset.from_cache(
        val_cache, resources, cfg, verbose=True, is_val=True
    )

    model, _ = train_end_to_end_map_belief(
        resources=resources,
        cfg=cfg,
        device=device,
        checkpoint_dir=ckpt_dir,
        plot_dir=ckpt_dir / "plots",
        verbose=True,
        train_ds=train_ds,
        val_ds=val_ds,
    )

    # Smoke test: verify the saved checkpoint loads cleanly.
    _, _, _, history = load_e2e_map_belief_checkpoint(
        ckpt_dir / "e2e_map_belief_best.pt", device=device
    )
    print(
        f"\nSmoke test passed: checkpoint loaded with {len(history)} epoch(s) of history."
    )

    return model, history
