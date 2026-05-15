from __future__ import annotations

import csv
import dataclasses
import json
from pathlib import Path

import torch

from decision_simulator.resources import load_resources

from .model import UNetBelief
from .training import (
    NeuralBeliefTrainingConfig,
    load_belief_checkpoint,
    train_neural_belief,
)

# Relative resource paths — resolved against storage_root at runtime.
_JEPA_REL         = Path("checkpoints/jepa.pt")
_DISTRIBUTIONS_REL = Path("data/clean/distributions.pkl")
_FORMATION_GEO_REL = Path("data/clean/formation_geometry.pkl")
_DISCOVERY_REL     = Path("data/clean/discovery_prior.pkl")


def train_belief_from_colab(
    storage_root: str | Path,
    checkpoint_dir: str | Path,
    device: str = "cuda",
    debug: bool = False,
    jepa_checkpoint: str | Path | None = None,
    distribution_bank_path: str | Path | None = None,
    formation_geometry_path: str | Path | None = None,
    discovery_prior_path: str | Path | None = None,
    **overrides,
) -> tuple[UNetBelief, list[dict]]:
    """Train the neural geological belief updater from a Colab notebook.

    Example
    -------
    >>> from decision_simulator.neural_belief.colab import train_belief_from_colab
    >>> model, history = train_belief_from_colab(
    ...     storage_root="/content/drive/MyDrive/thesis",
    ...     checkpoint_dir="checkpoints/belief_v1",
    ...     device="cuda",
    ...     debug=True,
    ... )

    Override any NeuralBeliefTrainingConfig field via keyword arguments:

    >>> model, history = train_belief_from_colab(
    ...     storage_root="/content/drive/MyDrive/thesis",
    ...     checkpoint_dir="checkpoints/belief_v1",
    ...     device="cuda",
    ...     n_train_maps=100,
    ...     samples_per_map=30,
    ...     n_epochs=75,
    ...     batch_size=32,
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
    jepa_checkpoint
        Path to the JEPA ``.pt`` file.  Defaults to
        ``<storage_root>/checkpoints/jepa.pt``.
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

    Returns
    -------
    tuple[UNetBelief, list[dict]]
        ``(model, history)`` — the trained model loaded with its best weights,
        and the per-epoch training history.
    """
    _check_device(device)

    root = Path(storage_root).expanduser().resolve()

    jepa, distributions, formation_geo, discovery = _resolve_paths(
        root,
        jepa_checkpoint,
        distribution_bank_path,
        formation_geometry_path,
        discovery_prior_path,
    )
    _check_required_paths(jepa, distributions, formation_geo)

    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.is_absolute():
        ckpt_dir = root / ckpt_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    cfg = _build_config(debug, overrides)
    print(f"\nTraining config:\n{cfg}\n")

    resources = load_resources(
        jepa_path=jepa,
        distributions_path=distributions,
        formation_geometry_path=formation_geo,
        discovery_prior_path=discovery,
        device=device,
    )

    model, _ = train_neural_belief(
        resources=resources,
        cfg=cfg,
        device=device,
        checkpoint_dir=ckpt_dir,
        verbose=True,
    )

    # Smoke test: verify the saved checkpoint loads cleanly.
    _, _, _, history = load_belief_checkpoint(ckpt_dir / "belief_best.pt", device=device)
    print(f"\nSmoke test passed: checkpoint loaded with {len(history)} epoch(s) of history.")

    _export_history(history, ckpt_dir)

    return model, history


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_paths(
    root: Path,
    jepa_checkpoint: str | Path | None,
    distribution_bank_path: str | Path | None,
    formation_geometry_path: str | Path | None,
    discovery_prior_path: str | Path | None,
) -> tuple[Path, Path, Path, Path]:
    jepa         = Path(jepa_checkpoint)         if jepa_checkpoint         is not None else root / _JEPA_REL
    distributions = Path(distribution_bank_path) if distribution_bank_path  is not None else root / _DISTRIBUTIONS_REL
    formation_geo = Path(formation_geometry_path) if formation_geometry_path is not None else root / _FORMATION_GEO_REL
    discovery     = Path(discovery_prior_path)    if discovery_prior_path    is not None else root / _DISCOVERY_REL
    return jepa, distributions, formation_geo, discovery


def _check_required_paths(jepa: Path, distributions: Path, formation_geo: Path) -> None:
    for path in (jepa, distributions, formation_geo):
        if not path.exists():
            raise FileNotFoundError(f"Required resource not found: {path}")


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
