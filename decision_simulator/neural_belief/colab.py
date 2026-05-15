from __future__ import annotations

import csv
import dataclasses
import json
from pathlib import Path

import torch

from decision_simulator.config_decision_experiments import (
    DISCOVERY_PRIOR,
    DISTRIBUTIONS,
    FORMATION_GEOMETRY,
    JEPA_CHECKPOINT,
)
from decision_simulator.resources import load_resources

from .model import UNetBelief
from .training import (
    NeuralBeliefTrainingConfig,
    load_belief_checkpoint,
    train_neural_belief,
)


def train_belief_from_colab(
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
    ...     checkpoint_dir="/content/drive/MyDrive/thesis/checkpoints/belief",
    ...     device="cuda",
    ...     debug=True,
    ... )

    Override any NeuralBeliefTrainingConfig field via keyword arguments:

    >>> model, history = train_belief_from_colab(
    ...     checkpoint_dir="/content/drive/MyDrive/thesis/checkpoints/belief",
    ...     device="cuda",
    ...     n_train_maps=100,
    ...     samples_per_map=30,
    ...     n_epochs=75,
    ...     batch_size=32,
    ... )

    Parameters
    ----------
    checkpoint_dir
        Directory where ``belief_best.pt``, ``belief_last.pt``,
        ``training_history.json``, and ``training_history.csv`` are saved.
    device
        ``"cuda"`` or ``"cpu"``.  Raises a clear error when CUDA is requested
        but unavailable.
    debug
        If True, use a minimal config (2 maps, 2 epochs, base_channels=16)
        to verify the full pipeline end-to-end without a full training run.
    jepa_checkpoint
        Path to the JEPA ``.pt`` file.  Defaults to ``checkpoints/jepa.pt``
        relative to the current working directory.
    distribution_bank_path
        Path to ``distributions.pkl``.  Defaults to
        ``data/clean/distributions.pkl``.
    formation_geometry_path
        Path to ``formation_geometry.pkl``.  Defaults to
        ``data/clean/formation_geometry.pkl``.
    discovery_prior_path
        Path to ``discovery_prior.pkl``.  Defaults to
        ``data/clean/discovery_prior.pkl``.  Optional — if the file does not
        exist, a uniform placement prior is used.
    **overrides
        Any field of :class:`NeuralBeliefTrainingConfig` by name, e.g.
        ``n_epochs=75``.  Unknown field names raise ``ValueError``.

    Returns
    -------
    tuple[UNetBelief, list[dict]]
        ``(model, history)`` — the trained model loaded with its best
        weights, and the per-epoch training history.
    """
    _check_device(device)

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    cfg = _build_config(debug, overrides)

    print(f"\nTraining config:\n{cfg}\n")

    resources = load_resources(
        jepa_path=Path(jepa_checkpoint) if jepa_checkpoint is not None else JEPA_CHECKPOINT,
        distributions_path=Path(distribution_bank_path) if distribution_bank_path is not None else DISTRIBUTIONS,
        formation_geometry_path=Path(formation_geometry_path) if formation_geometry_path is not None else FORMATION_GEOMETRY,
        discovery_prior_path=Path(discovery_prior_path) if discovery_prior_path is not None else DISCOVERY_PRIOR,
        device=device,
    )

    model, _ = train_neural_belief(
        resources=resources,
        cfg=cfg,
        device=device,
        checkpoint_dir=checkpoint_dir,
        verbose=True,
    )

    # Smoke test: verify the saved checkpoint loads cleanly.
    _, _, _, history = load_belief_checkpoint(checkpoint_dir / "belief_best.pt", device=device)
    print(f"\nSmoke test passed: checkpoint loaded with {len(history)} epoch(s) of history.")

    _export_history(history, checkpoint_dir)

    return model, history


# ---------------------------------------------------------------------------
# Helpers
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
