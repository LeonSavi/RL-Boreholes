from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Literal

import pandas as pd
import torch

from decision_simulator.resources import (
    check_device,
    check_encoder_path,
    check_sim_paths,
    load_decision_resources,
    resolve_resource_paths,
)

from .datasets import GeologicalBeliefDataset
from .map_hdf5 import HDF5MapDirectory, HDF5MapStore
from .models import UNetBelief
from .training_utils import TargetNormalizer

from .training import (
    NeuralBeliefTrainingConfig,
    MapBeliefTrainingConfig,
    E2EMapBeliefConfig,
    PatchBoreholeConfig,
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
    train_patch_borehole_transformer,
    load_patch_borehole_checkpoint,
    PatchBoreholeCLSConfig,
    train_patch_borehole_cls_transformer,
    load_patch_borehole_cls_checkpoint,
    VariableAwarePatchBoreholeConfig,
    train_variable_aware_patch_borehole_transformer,
    load_variable_aware_patch_borehole_checkpoint,
    VariableAwarePatchBoreholeUncertaintyConfig,
    train_variable_aware_patch_uncertainty_borehole_transformer,
    load_variable_aware_patch_uncertainty_borehole_checkpoint,
    GuidedExplorationConfig,
    GuidedE2EMapDataset,
    train_guided_exploration_belief,
    load_guided_belief_checkpoint,
    CatVarConfig,
    CatVarE2EMapDataset,
    train_cat_var_encoder,
    load_cat_var_checkpoint,
)
from .training.belief_models.end_to_end.train_guided_exploration_belief import _load_guide_model

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
_DEBUG_PATCH: dict = {**_DEBUG_E2E, "bh_patch_size": 10}
_DEBUG_CLS: dict = {**_DEBUG_E2E, "bh_patch_size": 10}
_DEBUG_VAR_AWARE: dict = {**_DEBUG_E2E, "bh_patch_size": 10}
_DEBUG_VAR_AWARE_UNCERTAINTY: dict = {**_DEBUG_E2E, "bh_patch_size": 10}
_DEBUG_CAT_VAR: dict = {**_DEBUG_E2E, "bh_patch_size": 10}

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
    store = HDF5MapDirectory(resolved_pool_path) if resolved_pool_path.is_dir() else HDF5MapStore(resolved_pool_path)

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

    store = HDF5MapDirectory(resolved_pool_path) if resolved_pool_path.is_dir() else HDF5MapStore(resolved_pool_path)

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

                    # Build and apply normalizer from this run's training targets.
                    normalizer_run = TargetNormalizer(mode=cfg.norm_mode)
                    normalizer_run.fit(train_ds_run.targets.numpy())
                    train_ds_run.apply_target_normalizer(normalizer_run)
                    val_ds_run.apply_target_normalizer(normalizer_run)

                    if is_transformer:
                        train_map_belief(
                            cfg=cfg,
                            device=device,
                            checkpoint_dir=ckpt_dir,
                            plot_dir=ckpt_dir / "plots",
                            verbose=True,
                            train_ds=train_ds_run,
                            val_ds=val_ds_run,
                            normalizer=normalizer_run,
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
                            normalizer=normalizer_run,
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

    store = HDF5MapDirectory(resolved_pool) if resolved_pool.is_dir() else HDF5MapStore(resolved_pool)

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
    bh_encoder_model: Literal["cnn", "patch", "cls", "variable_aware", "variable_aware_uncertainty", "cat_var_encoder"] = "cnn",
    guided_training: bool = False,
    guide_ckpt_path: str | Path | None = None,
    labels_dir: str | Path | None = None,
    **overrides,
) -> tuple[object, list[dict]]:
    """Train the end-to-end map belief transformer from a Colab notebook.

    Trains EndToEndMapBeliefTransformer (``bh_encoder_model="cnn"``),
    PatchBoreholeEndToEndMapBeliefTransformer (``bh_encoder_model="patch"``), or
    PatchBoreholeCLSEndToEndMapBeliefTransformer (``bh_encoder_model="cls"``) with a
    full-map MSE reconstruction objective.  The borehole encoder and map belief
    transformer are trained jointly from scratch — no pre-trained JEPA or
    autoencoder weights are required.  The ``norm_stats_from`` parameter
    controls which checkpoint's per-variable normalization statistics are used
    to standardise raw borehole inputs.

    Example
    -------
    >>> from decision_simulator.neural_belief.colab import train_end_to_end_from_colab

    # CNN front-end (default)
    >>> model, history = train_end_to_end_from_colab(
    ...     storage_root="/content/drive/MyDrive/thesis",
    ...     checkpoint_dir="checkpoints/e2e_cnn",
    ...     device="cuda",
    ...     debug=True,
    ... )

    # Patch tokenisation front-end (mean pooling)
    >>> model, history = train_end_to_end_from_colab(
    ...     storage_root="/content/drive/MyDrive/thesis",
    ...     checkpoint_dir="checkpoints/e2e_patch",
    ...     device="cuda",
    ...     bh_encoder_model="patch",
    ...     bh_patch_size=20,
    ... )

    # Patch tokenisation front-end (CLS-token pooling)
    >>> model, history = train_end_to_end_from_colab(
    ...     storage_root="/content/drive/MyDrive/thesis",
    ...     checkpoint_dir="checkpoints/e2e_cls",
    ...     device="cuda",
    ...     bh_encoder_model="cls",
    ...     bh_patch_size=20,
    ... )

    Parameters
    ----------
    storage_root
        Absolute root for all data and checkpoint paths.  Relative paths are
        resolved against this directory.
    checkpoint_dir
        Where checkpoints and ``training_history.{json,csv}`` are saved.
        Relative paths are resolved against ``storage_root``.
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
    bh_encoder_model
        Which borehole encoder front-end to use:

        ``"cnn"``                        — 1D CNN downsampling + transformer (default, current baseline)
        ``"patch"``                      — depth-patch tokenisation + mean pooling
        ``"cls"``                        — depth-patch tokenisation + learned CLS-token pooling
        ``"variable_aware"``             — one token per (variable, depth patch) + variable embedding + CLS pooling
        ``"variable_aware_uncertainty"`` — same as ``"variable_aware"`` with a parallel uncertainty head that predicts per-cell prediction error
        ``"cat_var_encoder"``            — same as ``"variable_aware_uncertainty"`` but also embeds per-depth rock-type labels via a soft one-hot projection

        When ``"patch"``, ``"cls"``, ``"variable_aware"``, ``"variable_aware_uncertainty"``, or
        ``"cat_var_encoder"``, pass ``bh_patch_size=N`` in ``**overrides`` to control the patch size (default 20).
    labels_dir
        Directory containing ``labels_vocab.pkl`` and ``labels_NNNNN.npz`` files produced by
        ``generate_training_maps.py``.  Only used when ``bh_encoder_model="cat_var_encoder"``.
        If ``None``, rock IDs default to zero (the ``"other"`` / unknown category).
        ``n_rock_types`` is inferred automatically from ``labels_vocab.pkl`` in this directory.
    **overrides
        Any field of the selected training config by name.  Common overrides:

        * ``n_epochs=100`` — train longer
        * ``use_false_positive_penalty=True`` — penalise false positives
        * ``batch_size=4`` — reduce if GPU memory is tight
        * ``bh_patch_size=20`` — patch size (``"patch"`` and ``"cls"`` only)

    Returns
    -------
    tuple[model, list[dict]]
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

    is_patch = bh_encoder_model == "patch"
    is_cls = bh_encoder_model == "cls"
    is_var_aware = bh_encoder_model == "variable_aware"
    is_var_aware_uncertainty = bh_encoder_model == "variable_aware_uncertainty"
    is_cat_var = bh_encoder_model == "cat_var_encoder"
    if is_cat_var:
        cfg_class = CatVarConfig
        debug_defaults = _DEBUG_CAT_VAR if debug else {}
    elif is_var_aware_uncertainty:
        cfg_class = VariableAwarePatchBoreholeUncertaintyConfig
        debug_defaults = _DEBUG_VAR_AWARE_UNCERTAINTY if debug else {}
    elif is_var_aware:
        cfg_class = VariableAwarePatchBoreholeConfig
        debug_defaults = _DEBUG_VAR_AWARE if debug else {}
    elif is_cls:
        cfg_class = PatchBoreholeCLSConfig
        debug_defaults = _DEBUG_CLS if debug else {}
    elif is_patch:
        cfg_class = PatchBoreholeConfig
        debug_defaults = _DEBUG_PATCH if debug else {}
    else:
        cfg_class = E2EMapBeliefConfig
        debug_defaults = _DEBUG_E2E if debug else {}

    cfg = build_training_config(
        cfg_class,
        {**debug_defaults, **overrides, **seq_overrides},
    )

    # ---- load map cache -------------------------------------------------------
    store = HDF5MapDirectory(resolved_pool) if resolved_pool.is_dir() else HDF5MapStore(resolved_pool)
    seed = overrides.get("seed", cfg.seed)

    train_cache, val_cache = store.load_map_data(
        n_train_maps=cfg.n_train_maps,
        n_val_maps=cfg.n_val_maps,
        n_orebodies=n_orebodies,
        seed=seed,
    )

    cfg.n_x = train_cache.n_x
    cfg.n_y = train_cache.n_y

    # ---- guided-exploration curriculum ----------------------------------------
    resolved_guide_path: str | None = None
    if guided_training and guide_ckpt_path is not None:
        _gp = Path(guide_ckpt_path)
        if not _gp.is_absolute():
            _gp = root / _gp
        resolved_guide_path = str(_gp)

    guided_cfg: GuidedExplorationConfig | None = None
    if guided_training:
        # Build a GuidedExplorationConfig for curriculum parameters.
        # Filter overrides to only fields known to GuidedExplorationConfig so
        # that model-specific keys (e.g. bh_patch_size) do not raise errors.
        _guided_valid = {f.name for f in dataclasses.fields(GuidedExplorationConfig)}
        _guided_overrides = {
            k: v for k, v in {**overrides, **seq_overrides}.items()
            if k in _guided_valid
        }
        guided_cfg = build_training_config(
            GuidedExplorationConfig,
            {
                **(_DEBUG_E2E if debug else {}),
                **_guided_overrides,
                **({"guide_ckpt_path": resolved_guide_path} if resolved_guide_path else {}),
            },
        )
        guided_cfg.n_x = train_cache.n_x
        guided_cfg.n_y = train_cache.n_y

    print(f"\nnorm_stats_from  : {norm_stats_from}")
    print(f"bh_encoder_model : {bh_encoder_model}")
    if is_patch or is_cls or is_var_aware or is_var_aware_uncertainty or is_cat_var:
        print(f"bh_patch_size    : {cfg.bh_patch_size}")
    if is_cat_var:
        print(f"labels_dir       : {labels_dir}")
    print(f"map_pool_path    : {resolved_pool}")
    print(f"n_orebodies      : {n_orebodies}")
    print(f"n_train_maps     : {cfg.n_train_maps}  n_val_maps: {cfg.n_val_maps}")
    print(f"latent_dim       : {cfg.latent_dim}")
    print(f"grid             : {cfg.n_x} × {cfg.n_y}")
    print(f"sequential       : {cfg.use_sequential_dataset}")
    if cfg.use_sequential_dataset:
        print(f"n_sequences      : {cfg.n_sequences_per_map}")
        print(f"prefix_steps     : {cfg.prefix_steps}")
    if guided_training:
        print("guided_training  : True")
        print(f"p_guided         : {guided_cfg.p_guided:.0%}")
        print(f"mode split       : A={guided_cfg.p_mode_a:.0%}  AB={guided_cfg.p_mode_ab:.0%}  AC={guided_cfg.p_mode_ac:.0%}")
        print(f"phase1_drills    : {guided_cfg.phase1_drills}")
        print(f"guide model      : {resolved_guide_path or 'heuristic (distance-from-drills)'}")
    print(f"Training config  :\n{cfg}\n")

    is_cnn = not (is_patch or is_cls or is_var_aware or is_var_aware_uncertainty or is_cat_var)

    # Resolve labels_dir once — used for both dataset building and vocab inference.
    _resolved_labels: Path | None = None
    if is_cat_var and labels_dir is not None:
        _resolved_labels = Path(labels_dir)
        if not _resolved_labels.is_absolute():
            _resolved_labels = root / _resolved_labels

    if guided_training and is_cnn:
        # train_guided_exploration_belief builds datasets internally from the caches,
        # so we skip the pre-build step here.
        train_ds = val_ds = None
    elif guided_training:
        # For non-CNN models build guided datasets externally and pass them in.
        # Load the guide model from checkpoint if one was given; otherwise fall
        # back to the distance-from-drills heuristic (guide_model=None).
        _guide_model = (
            _load_guide_model(resolved_guide_path, device)
            if resolved_guide_path is not None
            else None
        )
        if _guide_model is not None:
            print(f"  guide model loaded from {resolved_guide_path}")
        print("Building guided training dataset ...")
        train_ds = GuidedE2EMapDataset.from_cache_guided(
            train_cache, resources, guided_cfg,
            guide_model=_guide_model, normalizer=None,
            device=device, verbose=True, is_val=False,
        )
        print("Building guided validation dataset (random sequences) ...")
        val_ds = GuidedE2EMapDataset.from_cache_guided(
            val_cache, resources, guided_cfg,
            guide_model=_guide_model, normalizer=None,
            device=device, verbose=True, is_val=True,
        )
    else:
        print("Building training dataset ...")
        if is_cat_var:
            train_ds = CatVarE2EMapDataset.from_cache(
                train_cache, resources, cfg, labels_dir=_resolved_labels, verbose=True, is_val=False
            )
            print("Building validation dataset ...")
            val_ds = CatVarE2EMapDataset.from_cache(
                val_cache, resources, cfg, labels_dir=_resolved_labels, verbose=True, is_val=True
            )
        else:
            train_ds = E2EMapDataset.from_cache(
                train_cache, resources, cfg, verbose=True, is_val=False
            )
            print("Building validation dataset ...")
            val_ds = E2EMapDataset.from_cache(
                val_cache, resources, cfg, verbose=True, is_val=True
            )

    if is_cat_var:
        trained_model, _, run_dir = train_cat_var_encoder(
            resources=resources,
            cfg=cfg,
            device=device,
            checkpoint_dir=ckpt_dir,
            labels_dir=_resolved_labels,
            plot_dir=ckpt_dir,
            verbose=True,
            train_ds=train_ds,
            val_ds=val_ds,
        )
        _, _, _, history = load_cat_var_checkpoint(
            run_dir / "cat_var_best.pt", device=device
        )
    elif is_var_aware_uncertainty:
        trained_model, _, run_dir = train_variable_aware_patch_uncertainty_borehole_transformer(
            resources=resources,
            cfg=cfg,
            device=device,
            checkpoint_dir=ckpt_dir,
            plot_dir=ckpt_dir,
            verbose=True,
            train_ds=train_ds,
            val_ds=val_ds,
        )
        _, _, _, history = load_variable_aware_patch_uncertainty_borehole_checkpoint(
            run_dir / "variable_aware_patch_uncertainty_best.pt", device=device
        )
    elif is_var_aware:
        trained_model, _ = train_variable_aware_patch_borehole_transformer(
            resources=resources,
            cfg=cfg,
            device=device,
            checkpoint_dir=ckpt_dir,
            plot_dir=ckpt_dir / "plots",
            verbose=True,
            train_ds=train_ds,
            val_ds=val_ds,
        )
        _, _, _, history = load_variable_aware_patch_borehole_checkpoint(
            ckpt_dir / "variable_aware_patch_best.pt", device=device
        )
    elif is_cls:
        trained_model, _ = train_patch_borehole_cls_transformer(
            resources=resources,
            cfg=cfg,
            device=device,
            checkpoint_dir=ckpt_dir,
            plot_dir=ckpt_dir / "plots",
            verbose=True,
            train_ds=train_ds,
            val_ds=val_ds,
        )
        _, _, _, history = load_patch_borehole_cls_checkpoint(
            ckpt_dir / "patch_borehole_cls_best.pt", device=device
        )
    elif is_patch:
        trained_model, _ = train_patch_borehole_transformer(
            resources=resources,
            cfg=cfg,
            device=device,
            checkpoint_dir=ckpt_dir,
            plot_dir=ckpt_dir / "plots",
            verbose=True,
            train_ds=train_ds,
            val_ds=val_ds,
        )
        _, _, _, history = load_patch_borehole_checkpoint(
            ckpt_dir / "patch_borehole_best.pt", device=device
        )
    else:
        if guided_training:
            trained_model, _ = train_guided_exploration_belief(
                resources=resources,
                cfg=guided_cfg,
                device=device,
                checkpoint_dir=ckpt_dir,
                train_cache=train_cache,
                val_cache=val_cache,
                plot_dir=ckpt_dir / "plots",
                verbose=True,
            )
            _, _, _, history = load_guided_belief_checkpoint(
                ckpt_dir / "guided_belief_best.pt", device=device
            )
        else:
            trained_model, _ = train_end_to_end_map_belief(
                resources=resources,
                cfg=cfg,
                device=device,
                checkpoint_dir=ckpt_dir,
                plot_dir=ckpt_dir / "plots",
                verbose=True,
                train_ds=train_ds,
                val_ds=val_ds,
            )
            _, _, _, history = load_e2e_map_belief_checkpoint(
                ckpt_dir / "e2e_map_belief_best.pt", device=device
            )

    print(
        f"\nSmoke test passed: checkpoint loaded with {len(history)} epoch(s) of history."
    )

    return trained_model, history


def load_variable_aware_uncertainty_model_from_colab(
    checkpoint_path: str | Path,
    device: str = "cuda",
) -> object:
    """Load a trained VariableAwarePatchBoreholeUncertaintyEndToEndMapBeliefTransformer.

    Parameters
    ----------
    checkpoint_path
        Path to ``variable_aware_patch_uncertainty_best.pt``.
    device
        ``"cuda"`` or ``"cpu"``.

    Returns
    -------
    model
        Trained model in eval mode on ``device``.

    Example
    -------
    >>> model = load_variable_aware_uncertainty_model_from_colab(
    ...     "/content/drive/MyDrive/Thesis/checkpoints/variable_aware_uncertainty_no_FP"
    ...     "/2026-06-02_07-43-54/variable_aware_patch_uncertainty_best.pt",
    ...     device="cuda",
    ... )
    """
    check_device(device)
    model, _, _, _ = load_variable_aware_patch_uncertainty_borehole_checkpoint(
        Path(checkpoint_path), device=device
    )
    model.eval()
    return model


def load_sample_borehole_from_colab(
    storage_root: str | Path,
    hdf5_path: str | Path = "data/dataset_HDF5/maps_00000_00499.h5",
    map_idx: int = 0,
    cell_idx: int = 0,
    norm_stats_from: Literal["jepa", "autoencoder", "none"] = "jepa",
    device: str = "cpu",
) -> tuple["torch.Tensor", list[str]]:
    """Load and standardize a single borehole from an HDF5 map file.

    Reads one cell from the ``boreholes`` dataset (shape ``(N, n_x*n_y, V, D)``),
    standardizes it with the same per-variable norm stats used during training,
    and returns it ready to pass to :func:`inspect_borehole_attention`.

    Parameters
    ----------
    storage_root
        Absolute root for all data and checkpoint paths.
    hdf5_path
        Path to the ``.h5`` shard file.  Relative paths are resolved against
        ``storage_root``.
    map_idx
        Which map (row index) to read from the HDF5 file.
    cell_idx
        Which cell within that map; cells are stored in row-major order
        (flattened ``n_x * n_y``).
    norm_stats_from
        Which encoder checkpoint to load normalization statistics from.
        Must match what was used during training.
    device
        Device for the returned tensor (``"cpu"`` is fine for inspection).

    Returns
    -------
    borehole : torch.Tensor  (V, D)
        Standardized borehole tensor.
    variable_names : list[str]
        Geological variable names in V-dimension order, for use as
        ``var_names`` in :func:`inspect_borehole_attention`.

    Example
    -------
    >>> borehole, var_names = load_sample_borehole_from_colab(
    ...     storage_root="/content/drive/MyDrive/Thesis",
    ...     hdf5_path="data/dataset_HDF5/maps_00000_00499.h5",
    ...     map_idx=0,
    ...     cell_idx=12,
    ... )
    >>> latent, cls_attn = inspect_borehole_attention(model, borehole.to("cuda"), var_names=var_names)
    """
    import h5py
    import numpy as np
    from .models.belief_models.borehole_encoders.autoencoder import standardise

    root = Path(storage_root).expanduser().resolve()

    hdf5_resolved = Path(hdf5_path)
    if not hdf5_resolved.is_absolute():
        hdf5_resolved = root / hdf5_resolved

    jepa_path, ae_path, distributions, formation_geo, discovery = (
        resolve_resource_paths(root)
    )
    check_encoder_path(norm_stats_from, jepa_path, ae_path)

    resources, _ = load_decision_resources(
        borehole_encoder=norm_stats_from,
        jepa_path=jepa_path,
        ae_path=ae_path,
        distributions_path=distributions,
        formation_geometry_path=formation_geo,
        discovery_prior_path=discovery,
        device="cpu",
    )

    with h5py.File(hdf5_resolved, "r") as hf:
        bh_raw = hf["boreholes"][map_idx, cell_idx].astype(np.float32)  # (V, D)

    if resources.norm_stats:
        bh = standardise(bh_raw, resources.norm_stats, resources.variable_names)
    else:
        bh = bh_raw

    bh = np.nan_to_num(bh, nan=0.0).astype(np.float32)
    return torch.from_numpy(bh).to(device), resources.variable_names


def extract_borehole_attention_from_colab(
    checkpoint_path: str | Path,
    storage_root: str | Path,
    output_dir: str | Path = "attention_analysis",
    hdf5_path: str | Path = "data/dataset_HDF5/maps_00000_00499.h5",
    n_per_class: int = 50,
    norm_stats_from: Literal["jepa", "autoencoder", "none"] = "jepa",
    device: str = "cuda",
    seed: int = 42,
) -> dict:
    """Extract CLS-token attention tensors from multiple validation boreholes.

    Samples ``n_per_class`` no-ore maps and ``n_per_class`` ore maps from the
    HDF5 dataset (stratified by ``n_bodies``), runs each borehole through the
    trained ``VariableAwarePatchBoreholeTransformerEncoder`` with attention
    capture enabled, and saves the results to ``output_dir``.

    Parameters
    ----------
    checkpoint_path
        Path to ``variable_aware_patch_uncertainty_best.pt``.
    storage_root
        Absolute root for all data and resource paths.  Relative ``hdf5_path``
        values are resolved against this directory.
    output_dir
        Where to write the three output files.  Created if absent.  Relative
        paths are resolved against the current working directory.
    hdf5_path
        Path to the ``.h5`` shard.  Relative paths are resolved against
        ``storage_root``.
    n_per_class
        Boreholes to extract per class (no-ore and ≥1-ore-body).  Actual
        counts may be smaller when the dataset is imbalanced.
    norm_stats_from
        Which encoder checkpoint to load per-variable normalization statistics
        from.  Must match what was used during training.
    device
        ``"cuda"`` or ``"cpu"``.
    seed
        RNG seed for reproducible map / cell selection.

    Returns
    -------
    dict with keys:
        ``cls_attn``  — ``torch.Tensor`` ``(n_samples, n_layers, n_heads, V, P)``
        ``latent``    — ``torch.Tensor`` ``(n_samples, latent_dim)``
        ``metadata``  — ``pd.DataFrame``  with one row per borehole

    Saved files
    -----------
    ``<output_dir>/cls_attn.pt``   — stacked CLS-attention tensor
    ``<output_dir>/latent.pt``     — stacked latent embeddings
    ``<output_dir>/metadata.csv``  — per-borehole metadata

    Example
    -------
    >>> from decision_simulator.neural_belief.colab import (
    ...     extract_borehole_attention_from_colab,
    ... )
    >>> result = extract_borehole_attention_from_colab(
    ...     checkpoint_path=(
    ...         "/content/drive/MyDrive/Thesis/checkpoints/"
    ...         "variable_aware_uncertainty_no_FP/2026-06-02_07-43-54/"
    ...         "variable_aware_patch_uncertainty_best.pt"
    ...     ),
    ...     storage_root="/content/drive/MyDrive/Thesis",
    ...     output_dir="attention_analysis",
    ...     n_per_class=50,
    ...     device="cuda",
    ... )
    >>> cls_attn = result["cls_attn"]   # (100, n_layers, n_heads, 5, 22)
    >>> latent   = result["latent"]     # (100, latent_dim)
    >>> meta     = result["metadata"]   # pd.DataFrame
    """
    import h5py
    import numpy as np
    from .models.belief_models.borehole_encoders.autoencoder import standardise
    from .models.belief_models.borehole_encoder_components.attention_utils import (
        extract_cls_attention,
    )

    check_device(device)

    root = Path(storage_root).expanduser().resolve()
    out_dir = Path(output_dir)
    if not out_dir.is_absolute():
        out_dir = Path.cwd() / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    hdf5_resolved = Path(hdf5_path)
    if not hdf5_resolved.is_absolute():
        hdf5_resolved = root / hdf5_resolved

    # --- load model -----------------------------------------------------------
    print("Loading model from checkpoint …")
    model, _, _, _ = load_variable_aware_patch_uncertainty_borehole_checkpoint(
        Path(checkpoint_path), device=device
    )
    model.eval()
    enc = model.bh_encoder
    n_variables = enc.cfg.n_variables
    n_patches = enc.n_patches
    print(f"  n_variables  : {n_variables}")
    print(f"  n_patches    : {n_patches}")

    # --- load norm stats ------------------------------------------------------
    jepa_path, ae_path, distributions, formation_geo, discovery = (
        resolve_resource_paths(root)
    )
    check_encoder_path(norm_stats_from, jepa_path, ae_path)
    resources, _ = load_decision_resources(
        borehole_encoder=norm_stats_from,
        jepa_path=jepa_path,
        ae_path=ae_path,
        distributions_path=distributions,
        formation_geometry_path=formation_geo,
        discovery_prior_path=discovery,
        device="cpu",
    )

    # --- sample map indices ---------------------------------------------------
    rng = np.random.default_rng(seed)

    with h5py.File(hdf5_resolved, "r") as hf:
        n_maps_in_file = int(hf["boreholes"].shape[0])
        n_cells = int(hf["boreholes"].shape[1])
        n_x = int(hf.attrs.get("pool_n_x", int(round(n_cells ** 0.5))))
        n_y = int(hf.attrs.get("pool_n_y", int(round(n_cells ** 0.5))))
        n_bodies_arr = hf["n_bodies"][:].astype(np.int32)    # (N,)
        map_index_arr = hf["map_index"][:].astype(np.int32)  # (N,)

    no_ore_idxs = np.where(n_bodies_arr == 0)[0]
    ore_idxs    = np.where(n_bodies_arr  > 0)[0]

    n_no_ore = min(n_per_class, len(no_ore_idxs))
    n_ore    = min(n_per_class, len(ore_idxs))

    selected_no_ore = rng.choice(no_ore_idxs, size=n_no_ore, replace=False)
    selected_ore    = rng.choice(ore_idxs,    size=n_ore,    replace=False)
    selected_idxs   = np.concatenate([selected_no_ore, selected_ore])
    ore_labels      = np.array([0] * n_no_ore + [1] * n_ore, dtype=np.int32)
    cell_idxs       = rng.integers(0, n_cells, size=len(selected_idxs))

    print(f"\n  HDF5 maps  : {n_maps_in_file}  (no-ore={len(no_ore_idxs)}, ore={len(ore_idxs)})")
    print(f"  Selected   : no-ore={n_no_ore}  ore={n_ore}  total={len(selected_idxs)}")
    print(f"  Grid       : {n_x} × {n_y}  ({n_cells} cells per map)\n")

    # --- extract attention for each borehole ----------------------------------
    all_cls_attn: list[torch.Tensor] = []
    all_latent:   list[torch.Tensor] = []
    meta_rows:    list[dict]         = []

    with h5py.File(hdf5_resolved, "r") as hf:
        boreholes_ds    = hf["boreholes"]     # (N, n_cells, V, D)
        yield_target_ds = hf["yield_target"]  # (N, n_x, n_y)

        for sample_i, (map_idx, cell_idx, ore_label) in enumerate(
            zip(selected_idxs.tolist(), cell_idxs.tolist(), ore_labels.tolist())
        ):
            bh_raw = boreholes_ds[map_idx, cell_idx].astype(np.float32)  # (V, D)
            cell_x = cell_idx // n_y
            cell_y = cell_idx  % n_y
            ore_val = float(yield_target_ds[map_idx, cell_x, cell_y])

            if resources.norm_stats:
                bh = standardise(bh_raw, resources.norm_stats, resources.variable_names)
            else:
                bh = bh_raw
            bh = np.nan_to_num(bh, nan=0.0).astype(np.float32)

            bh_t = torch.from_numpy(bh).unsqueeze(0).to(device)  # (1, V, D)

            with torch.no_grad():
                latent, attn_weights = model.encode_borehole_with_attention(bh_t)

            if attn_weights is None:
                raise RuntimeError(
                    "Model returned no attention weights. "
                    "Ensure bh_n_layers > 0 in the model config."
                )

            # cls_attn: (n_layers, 1, n_heads, V, P) → squeeze batch dim
            cls_attn_sq = extract_cls_attention(
                attn_weights, n_variables, n_patches
            )[:, 0, :, :, :].cpu()  # (n_layers, n_heads, V, P)

            all_cls_attn.append(cls_attn_sq)
            all_latent.append(latent[0].cpu())  # (latent_dim,)

            meta_rows.append({
                "sample_idx":     sample_i,
                "map_idx":        map_idx,
                "map_global_idx": int(map_index_arr[map_idx]),
                "cell_idx":       cell_idx,
                "cell_x":         cell_x,
                "cell_y":         cell_y,
                "ore_label":      int(ore_label),
                "n_bodies":       int(n_bodies_arr[map_idx]),
                "ore_val":        ore_val,
            })

            if (sample_i + 1) % 10 == 0 or (sample_i + 1) == len(selected_idxs):
                print(
                    f"  [{sample_i + 1:>3}/{len(selected_idxs)}]  "
                    f"map={map_idx:>4}  cell={cell_idx:>4}  "
                    f"label={int(ore_label)}  ore_val={ore_val:.4f}"
                )

    # --- stack and save -------------------------------------------------------
    cls_attn_tensor = torch.stack(all_cls_attn)  # (n_samples, n_layers, n_heads, V, P)
    latent_tensor   = torch.stack(all_latent)    # (n_samples, latent_dim)
    metadata_df     = pd.DataFrame(meta_rows)

    torch.save(cls_attn_tensor, out_dir / "cls_attn.pt")
    torch.save(latent_tensor,   out_dir / "latent.pt")
    metadata_df.to_csv(out_dir / "metadata.csv", index=False)

    n_no_ore_saved = int((metadata_df["ore_label"] == 0).sum())
    n_ore_saved    = int((metadata_df["ore_label"] == 1).sum())

    print(f"\n{'=' * 50}")
    print(f"  Extracted boreholes : {len(meta_rows)}")
    print(f"  No-ore samples      : {n_no_ore_saved}")
    print(f"  Ore samples         : {n_ore_saved}")
    print(f"  cls_attn shape      : {tuple(cls_attn_tensor.shape)}")
    print(f"  latent shape        : {tuple(latent_tensor.shape)}")
    print(f"{'=' * 50}")
    print(f"\nSaved to  '{out_dir}':")
    print(f"  cls_attn.pt  → {out_dir / 'cls_attn.pt'}")
    print(f"  latent.pt    → {out_dir / 'latent.pt'}")
    print(f"  metadata.csv → {out_dir / 'metadata.csv'}")

    return {
        "cls_attn": cls_attn_tensor,
        "latent":   latent_tensor,
        "metadata": metadata_df,
    }


def inspect_borehole_attention(
    model: object,
    borehole: "torch.Tensor",
    var_names: list[str] | None = None,
    layer_idx: int = -1,
    sample_idx: int = 0,
    plot: bool = True,
    title: str | None = None,
) -> tuple["torch.Tensor", "torch.Tensor | None"]:
    """Extract CLS-token attention from the borehole encoder and optionally plot it.

    Only works with models that use VariableAwarePatchBoreholeTransformerEncoder
    (i.e. ``bh_encoder_model="variable_aware"`` or ``"variable_aware_uncertainty"``).

    Parameters
    ----------
    model      : trained end-to-end model with a ``bh_encoder`` attribute and an
                 ``encode_borehole_with_attention()`` method (e.g.
                 VariableAwarePatchBoreholeUncertaintyEndToEndMapBeliefTransformer).
    borehole   : (V, D) or (B, V, D) — standardised raw borehole on the correct device.
    var_names  : geological variable names for the y-axis labels.
                 Falls back to "Var 0", "Var 1", … when None.
    layer_idx  : which transformer layer to visualize; -1 = last layer.
    sample_idx : which batch element to plot when borehole is a batch.
    plot       : if True, render the CLS-attention heatmap inline.
    title      : custom plot title.

    Returns
    -------
    latent   : (B, latent_dim) — borehole latent embedding
    cls_attn : (n_layers, B, n_heads, n_variables, n_patches) or None

    Example (Colab)
    ---------------
    >>> model, history = train_end_to_end_from_colab(
    ...     ..., bh_encoder_model="variable_aware_uncertainty"
    ... )
    >>> # borehole: a single standardised borehole tensor (V, D)
    >>> latent, cls_attn = inspect_borehole_attention(
    ...     model, borehole, var_names=resources.variable_names
    ... )
    """
    from .models.belief_models.borehole_encoder_components.attention_utils import (
        extract_cls_attention,
        plot_cls_attention,
    )

    model.eval()
    with torch.no_grad():
        latent, attn_weights = model.encode_borehole_with_attention(borehole)

    if attn_weights is None:
        print("No attention weights (model has bh_n_layers=0).")
        return latent, None

    enc = model.bh_encoder
    cls_attn = extract_cls_attention(attn_weights, enc.cfg.n_variables, enc.n_patches)

    if plot:
        plot_cls_attention(
            cls_attn,
            layer_idx=layer_idx,
            sample_idx=sample_idx,
            var_names=var_names,
            title=title,
        )

    return latent, cls_attn

