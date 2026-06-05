from __future__ import annotations

import torch
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Any, Literal

from decision_simulator.neural_belief.models.belief_models.borehole_encoders.jepa_encoder import JEPAModel, load_jepa_checkpoint

from simulator.distributions import (
    DistributionBank,
    DiscoveryPrior,
)

from simulator.formation_geometry import (
    FormationGeometry,
)

from decision_simulator.config_decision_experiments import (
    JEPA_CHECKPOINT,
    AUTOENCODER_CHECKPOINT,
    DISTRIBUTIONS,
    FORMATION_GEOMETRY,
    DISCOVERY_PRIOR,
)


@dataclass
class DecisionSimulationResources:
    """
    Shared runtime resources used by decision-simulation experiments.
    """

    jepa_model: JEPAModel | None
    norm_stats: dict[str, tuple[float, float]]
    variable_names: list[str]
    distribution_bank: DistributionBank
    formation_geometry: FormationGeometry
    discovery_prior: DiscoveryPrior | None = None
    # Explicit encoder callable; when None, falls back to jepa_model.embed.
    borehole_encoder_fn: Callable[..., Any] | None = field(default=None, repr=False)


def load_decision_resources(
    borehole_encoder: Literal["jepa", "autoencoder", "none"] = "jepa",
    jepa_path: Path = JEPA_CHECKPOINT,
    ae_path: Path = AUTOENCODER_CHECKPOINT,
    distributions_path: Path = DISTRIBUTIONS,
    formation_geometry_path: Path = FORMATION_GEOMETRY,
    discovery_prior_path: Path = DISCOVERY_PRIOR,
    device: str = "cpu",
) -> tuple[DecisionSimulationResources, int]:
    """Load simulator components and the requested borehole encoder.

    Always loads DistributionBank and FormationGeometry.
    Loads DiscoveryPrior only if the file exists.

    Parameters
    ----------
    borehole_encoder
        ``"jepa"``        — JEPA encoder; latent_dim from model config.
        ``"autoencoder"`` — 1-D CNN autoencoder; latent_dim from model config.
        ``"none"``        — no encoder; latent_dim=0.

    Returns
    -------
    tuple[DecisionSimulationResources, int]
        ``(resources, latent_dim)``
    """
    if borehole_encoder not in ("jepa", "autoencoder", "none"):
        raise ValueError(
            f"Unknown borehole_encoder={borehole_encoder!r}. "
            "Expected 'jepa', 'autoencoder', or 'none'."
        )

    for path in (distributions_path, formation_geometry_path):
        if not path.exists():
            raise FileNotFoundError(f"Required resource not found: {path}")

    print("Loading simulator components...")
    distribution_bank = DistributionBank.load(distributions_path)
    formation_geometry = FormationGeometry.load(formation_geometry_path)
    discovery_prior: DiscoveryPrior | None = None
    if discovery_prior_path.exists():
        discovery_prior = DiscoveryPrior.load(discovery_prior_path)
    else:
        print(
            f"  [info] {discovery_prior_path} not found - using uniform ore placement"
        )

    if borehole_encoder == "jepa":
        if not jepa_path.exists():
            raise FileNotFoundError(f"JEPA checkpoint not found: {jepa_path}")
        print("Loading JEPA checkpoint...")
        jepa_model, norm_stats, variable_names = load_jepa_checkpoint(
            jepa_path, device=device
        )
        jepa_model.eval()
        latent_dim: int = jepa_model.cfg.latent_dim
        print(f"  variable_names : {variable_names}")
        print(f"  device         : {device}")
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
        if not ae_path.exists():
            raise FileNotFoundError(f"Autoencoder checkpoint not found: {ae_path}")
        from decision_simulator.neural_belief.models.belief_models.borehole_encoders.autoencoder import load_checkpoint as load_ae_checkpoint

        print("Loading autoencoder checkpoint...")
        ae_model, norm_stats, variable_names = load_ae_checkpoint(
            ae_path, device=device
        )
        ae_model.eval()
        latent_dim = ae_model.cfg.latent_dim
        print(f"  variable_names : {variable_names}")
        print(f"  device         : {device}")
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


def load_resources(
    jepa_path: Path = JEPA_CHECKPOINT,
    distributions_path: Path = DISTRIBUTIONS,
    formation_geometry_path: Path = FORMATION_GEOMETRY,
    discovery_prior_path: Path = DISCOVERY_PRIOR,
    device: str = "cpu",
) -> DecisionSimulationResources:
    """Load JEPA model + simulator components.

    Resources are expensive to initialise — call once per experiment and
    reuse across seeds.

    Returns
    -------
    DecisionSimulationResources
    """
    resources, _ = load_decision_resources(
        borehole_encoder="jepa",
        jepa_path=jepa_path,
        distributions_path=distributions_path,
        formation_geometry_path=formation_geometry_path,
        discovery_prior_path=discovery_prior_path,
        device=device,
    )
    return resources


def resolve_resource_paths(
    root: Path,
) -> tuple[Path, Path, Path, Path, Path]:
    """Return resource paths by combining root with the config-defined relative paths.

    Returns
    -------
    tuple[Path, Path, Path, Path, Path]
        ``(jepa_path, ae_path, distributions_path, formation_geometry_path, discovery_prior_path)``
    """
    return (
        root / JEPA_CHECKPOINT,
        root / AUTOENCODER_CHECKPOINT,
        root / DISTRIBUTIONS,
        root / FORMATION_GEOMETRY,
        root / DISCOVERY_PRIOR,
    )


def check_encoder_path(borehole_encoder: str, jepa_path: Path, ae_path: Path) -> None:
    """Raise FileNotFoundError early if the required encoder checkpoint is missing."""
    if borehole_encoder == "jepa" and not jepa_path.exists():
        raise FileNotFoundError(f"JEPA checkpoint not found: {jepa_path}")
    if borehole_encoder == "autoencoder" and not ae_path.exists():
        raise FileNotFoundError(f"Autoencoder checkpoint not found: {ae_path}")


def check_sim_paths(distributions: Path, formation_geo: Path) -> None:
    """Raise FileNotFoundError early if required simulator data files are missing."""
    for path in (distributions, formation_geo):
        if not path.exists():
            raise FileNotFoundError(f"Required resource not found: {path}")


def check_device(device: str) -> None:
    """Validate the requested device and print GPU info when CUDA is available."""
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "device='cuda' was requested but CUDA is not available. "
                "In Colab: Runtime > Change runtime type > GPU. "
                "Or pass device='cpu' to run on CPU."
            )
        print(f"CUDA available : {torch.cuda.is_available()}")
        print(f"GPU            : {torch.cuda.get_device_name(0)}")
