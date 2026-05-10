from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from encoder.jepa_encoder import JEPAModel, load_jepa_checkpoint

from simulator.distributions import (
    DistributionBank,
    DiscoveryPrior,
)

from simulator.formation_geometry import (
    FormationGeometry,
)

from decision_simulator.config_decision_experiments import (
    JEPA_CHECKPOINT,
    DISTRIBUTIONS,
    FORMATION_GEOMETRY,
    DISCOVERY_PRIOR,
)


@dataclass
class DecisionSimulationResources:
    """
    Shared runtime resources used by decision-simulation experiments.
    """

    jepa_model: JEPAModel
    norm_stats: dict[str, tuple[float, float]]
    variable_names: list[str]
    distribution_bank: DistributionBank
    formation_geometry: FormationGeometry
    discovery_prior: DiscoveryPrior | None = None


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

    The checkpoint contains:
    - trained model weights (`state_dict`)
    - JEPA architecture configuration (`cfg`)
    - variable standardisation statistics (`stats`)
    - ordered variable/channel names (`variables`)

    The returned `norm_stats` and `variable_names` must be reused during
    inference to ensure boreholes are standardised and ordered identically
    to training.

    Returns
    -------
    DecisionSimulationResources
    """
    print("Loading JEPA checkpoint...")
    jepa_model, norm_stats, variable_names = load_jepa_checkpoint(
        jepa_path, device=device
    )
    jepa_model.eval()
    print(f"  variable_names : {variable_names}")
    print(f"  device    : {device}")

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

    return DecisionSimulationResources(
        jepa_model=jepa_model,
        norm_stats=norm_stats,
        variable_names=variable_names,
        distribution_bank=distribution_bank,
        formation_geometry=formation_geometry,
        discovery_prior=discovery_prior,
    )
