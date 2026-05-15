from __future__ import annotations

import csv
import dataclasses
import json
from pathlib import Path
from typing import Literal

import pandas as pd
import torch

from simulator.distributions import DiscoveryPrior, DistributionBank
from simulator.formation_geometry import FormationGeometry
from decision_simulator.resources import DecisionSimulationResources

from .model import UNetBelief
from .training import (
    NeuralBeliefTrainingConfig,
    load_belief_checkpoint,
    train_neural_belief,
)

# Default resource paths, resolved against storage_root at runtime.
_JEPA_REL = Path("checkpoints/jepa.pt")
_AE_REL = Path("checkpoints/ae.pt")
_DISTRIBUTIONS_REL = Path("data/clean/distributions.pkl")
_FORMATION_GEO_REL = Path("data/clean/formation_geometry.pkl")
_DISCOVERY_REL = Path("data/clean/discovery_prior.pkl")


def train_belief_from_colab(
    storage_root: str | Path,
    checkpoint_dir: str | Path,
    device: str = "cuda",
    debug: bool = False,
    borehole_encoder: Literal["jepa", "autoencoder", "none"] = "jepa",
    jepa_checkpoint: str | Path | None = None,
    autoencoder_checkpoint: str | Path | None = None,
    distribution_bank_path: str | Path | None = None,
    formation_geometry_path: str | Path | None = None,
    discovery_prior_path: str | Path | None = None,
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
    jepa_checkpoint
        Path to the JEPA ``.pt`` file.  Defaults to
        ``<storage_root>/checkpoints/jepa.pt``.  Required when
        ``borehole_encoder="jepa"``.
    autoencoder_checkpoint
        Path to the autoencoder ``.pt`` file.  Defaults to
        ``<storage_root>/checkpoints/ae.pt``.  Required when
        ``borehole_encoder="autoencoder"``.
    distribution_bank_path
        Path to ``distributions.pkl``.  Defaults to
        ``<storage_root>/data/clean/distributions.pkl``.
    formation_geometry_path
        Path to ``formation_geometry.pkl``.  Defaults to
        ``<storage_root>/data/clean/formation_geometry.pkl``.
    discovery_prior_path
        Path to ``discovery_prior.pkl``.  Defaults to
        ``<storage_root>/data/clean/discovery_prior.pkl``.  Optional — if the
        file does not exist, a uniform placement prior is used.
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
    _check_device(device)

    root = Path(storage_root).expanduser().resolve()

    jepa_path, ae_path, distributions, formation_geo, discovery = _resolve_paths(
        root,
        jepa_checkpoint,
        autoencoder_checkpoint,
        distribution_bank_path,
        formation_geometry_path,
        discovery_prior_path,
    )
    _check_encoder_path(borehole_encoder, jepa_path, ae_path)
    _check_sim_paths(distributions, formation_geo)

    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.is_absolute():
        ckpt_dir = root / ckpt_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    plot_dir_resolved: Path | None = None
    if plot_dir is not None:
        plot_dir_resolved = Path(plot_dir)
        if not plot_dir_resolved.is_absolute():
            plot_dir_resolved = root / plot_dir_resolved

    resources, latent_dim = _load_encoder_resources(
        borehole_encoder,
        jepa_path,
        ae_path,
        distributions,
        formation_geo,
        discovery,
        device,
    )

    # in_channels and latent_dim are derived from the encoder;
    # explicit user overrides take precedence.
    encoder_defaults = {"in_channels": 2 + latent_dim, "latent_dim": latent_dim}
    cfg = _build_config(debug, {**encoder_defaults, **overrides})

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

    _export_history(history, ckpt_dir)

    return model, history


# ---------------------------------------------------------------------------
# Resource loading
# ---------------------------------------------------------------------------


def _load_encoder_resources(
    borehole_encoder: str,
    jepa_path: Path,
    ae_path: Path,
    distributions: Path,
    formation_geo: Path,
    discovery: Path,
    device: str,
) -> tuple[DecisionSimulationResources, int]:
    """Load simulator components and the requested borehole encoder.

    Returns (resources, latent_dim).
    """
    distribution_bank = DistributionBank.load(distributions)
    formation_geometry = FormationGeometry.load(formation_geo)
    discovery_prior = DiscoveryPrior.load(discovery) if discovery.exists() else None

    if borehole_encoder == "jepa":
        from encoder.jepa_encoder import load_jepa_checkpoint

        jepa_model, norm_stats, variable_names = load_jepa_checkpoint(
            jepa_path, device=device
        )
        jepa_model.eval()
        latent_dim: int = jepa_model.cfg.latent_dim
        resources = DecisionSimulationResources(
            jepa_model=jepa_model,
            norm_stats=norm_stats,
            variable_names=variable_names,
            distribution_bank=distribution_bank,
            formation_geometry=formation_geometry,
            discovery_prior=discovery_prior,
            # borehole_encoder_fn=None → encode_full_latent_map falls back to jepa_model.embed
        )

    elif borehole_encoder == "autoencoder":
        from encoder.autoencoder import load_checkpoint as load_ae_checkpoint

        ae_model, norm_stats, variable_names = load_ae_checkpoint(
            ae_path, device=device
        )
        ae_model.eval()
        latent_dim = ae_model.cfg.latent_dim
        resources = DecisionSimulationResources(
            jepa_model=None,
            norm_stats=norm_stats,
            variable_names=variable_names,
            distribution_bank=distribution_bank,
            formation_geometry=formation_geometry,
            discovery_prior=discovery_prior,
            borehole_encoder_fn=ae_model.encoder,  # BoreholeEncoder: (B,V,D) → (B,latent_dim)
        )

    else:  # "none"
        latent_dim = 0
        resources = DecisionSimulationResources(
            jepa_model=None,
            norm_stats={},
            variable_names=[],
            distribution_bank=distribution_bank,
            formation_geometry=formation_geometry,
            discovery_prior=discovery_prior,
            # borehole_encoder_fn=None + jepa_model=None → returns (n_x, n_y, 0) latent map
        )

    return resources, latent_dim


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _resolve_paths(
    root: Path,
    jepa_checkpoint: str | Path | None,
    autoencoder_checkpoint: str | Path | None,
    distribution_bank_path: str | Path | None,
    formation_geometry_path: str | Path | None,
    discovery_prior_path: str | Path | None,
) -> tuple[Path, Path, Path, Path, Path]:
    jepa = Path(jepa_checkpoint) if jepa_checkpoint is not None else root / _JEPA_REL
    ae = (
        Path(autoencoder_checkpoint)
        if autoencoder_checkpoint is not None
        else root / _AE_REL
    )
    distributions = (
        Path(distribution_bank_path)
        if distribution_bank_path is not None
        else root / _DISTRIBUTIONS_REL
    )
    formation_geo = (
        Path(formation_geometry_path)
        if formation_geometry_path is not None
        else root / _FORMATION_GEO_REL
    )
    discovery = (
        Path(discovery_prior_path)
        if discovery_prior_path is not None
        else root / _DISCOVERY_REL
    )
    return jepa, ae, distributions, formation_geo, discovery


def _check_encoder_path(borehole_encoder: str, jepa_path: Path, ae_path: Path) -> None:
    if borehole_encoder == "jepa" and not jepa_path.exists():
        raise FileNotFoundError(f"JEPA checkpoint not found: {jepa_path}")
    if borehole_encoder == "autoencoder" and not ae_path.exists():
        raise FileNotFoundError(f"Autoencoder checkpoint not found: {ae_path}")


def _check_sim_paths(distributions: Path, formation_geo: Path) -> None:
    for path in (distributions, formation_geo):
        if not path.exists():
            raise FileNotFoundError(f"Required resource not found: {path}")


# ---------------------------------------------------------------------------
# Config / device helpers
# ---------------------------------------------------------------------------


def _check_device(device: str) -> None:
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "device='cuda' was requested but CUDA is not available. "
                "In Colab: Runtime > Change runtime type > GPU. "
                "Or pass device='cpu' to run on CPU."
            )
        print(f"CUDA available : {torch.cuda.is_available()}")
        print(f"GPU            : {torch.cuda.get_device_name(0)}")


def _build_config(debug: bool, overrides: dict) -> NeuralBeliefTrainingConfig:
    valid_fields = {f.name for f in dataclasses.fields(NeuralBeliefTrainingConfig)}
    invalid = set(overrides) - valid_fields
    if invalid:
        raise ValueError(
            f"Unknown NeuralBeliefTrainingConfig field(s): {sorted(invalid)}.\n"
            f"Valid fields: {sorted(valid_fields)}"
        )

    if debug:
        cfg = NeuralBeliefTrainingConfig(
            n_train_maps=2,
            samples_per_map=2,
            n_val_maps=1,
            val_samples_per_map=2,
            n_epochs=2,
            batch_size=2,
            base_channels=16,
        )
    else:
        cfg = NeuralBeliefTrainingConfig()

    for key, value in overrides.items():
        setattr(cfg, key, value)

    return cfg


# ---------------------------------------------------------------------------
# History export
# ---------------------------------------------------------------------------


def compare_belief_encoders_from_colab(
    storage_root: str | Path,
    output_root: str | Path = "checkpoints/belief_comparison",
    device: str = "cuda",
    variants: tuple[str, ...] = ("none", "autoencoder", "jepa"),
    debug: bool = False,
    **overrides,
) -> pd.DataFrame:
    """Train and compare multiple borehole encoder variants on identical data.

    All variants are trained with the same seed, maps, train/val split, drill
    configurations, and model hyperparameters — only the encoder type differs.
    Seed defaults to 42 (``NeuralBeliefTrainingConfig`` default); pass
    ``seed=N`` in ``overrides`` to change it consistently across all variants.

    Example
    -------
    >>> from decision_simulator.neural_belief.colab import compare_belief_encoders_from_colab
    >>> df = compare_belief_encoders_from_colab(
    ...     storage_root="/content/drive/MyDrive/thesis",
    ...     debug=True,
    ... )
    >>> print(df)

    Parameters
    ----------
    storage_root
        Absolute root for all data and checkpoint paths.
    output_root
        Parent directory for per-variant subdirectories. Relative paths are
        resolved against ``storage_root``. Each variant is saved under
        ``<output_root>/belief_<variant>/``.
    device
        ``"cuda"`` or ``"cpu"``.
    variants
        Encoder types to compare, in order.
    debug
        Run all variants with a minimal config (2 maps, 2 epochs) for a
        quick end-to-end check.
    **overrides
        Forwarded unchanged to every ``train_belief_from_colab`` call.
        ``in_channels`` and ``latent_dim`` are always derived from the
        encoder and cannot be overridden here.

    Returns
    -------
    pd.DataFrame
        One row per variant with columns ``encoder``, ``best_epoch``,
        ``best_val_mse``, ``best_val_mae``, ``best_val_corr``.
        Also written to ``<output_root>/comparison_summary.csv``.
    """
    _check_device(device)
    root = Path(storage_root).expanduser().resolve()

    out_root = Path(output_root)
    if not out_root.is_absolute():
        out_root = root / out_root
    out_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []

    for variant in variants:
        print(f"\n{'=' * 60}")
        print(f"  Encoder variant : {variant}")
        print(f"{'=' * 60}")

        ckpt_dir = out_root / f"belief_{variant}"
        plot_dir = ckpt_dir / "plots"

        _, history = train_belief_from_colab(
            storage_root=root,
            checkpoint_dir=ckpt_dir,
            device=device,
            debug=debug,
            borehole_encoder=variant,
            plot_dir=plot_dir,
            **overrides,
        )

        best = min(history, key=lambda r: r["val_mse"])
        rows.append(
            {
                "encoder": variant,
                "best_epoch": best["epoch"],
                "best_val_mse": best["val_mse"],
                "best_val_mae": best["val_mae"],
                "best_val_corr": best["val_corr"],
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


# ---------------------------------------------------------------------------
# History export
# ---------------------------------------------------------------------------


def _export_history(history: list[dict], checkpoint_dir: Path) -> None:
    if not history:
        return

    json_path = checkpoint_dir / "training_history.json"
    with open(json_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"History -> {json_path}")

    csv_path = checkpoint_dir / "training_history.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    print(f"History -> {csv_path}")
