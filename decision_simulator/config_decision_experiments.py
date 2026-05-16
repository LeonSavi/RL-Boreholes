from dataclasses import dataclass
from pathlib import Path

METHOD_GREEDY = "greedy_jepa_knn"
METHOD_PARTICLE_BELIEF = "particle_belief_expected_ore"

JEPA_CHECKPOINT = Path("checkpoints/jepa.pt")
AUTOENCODER_CHECKPOINT = Path("checkpoints/ae.pt")
DISTRIBUTIONS = Path("data/clean/distributions.pkl")
FORMATION_GEOMETRY = Path("data/clean/formation_geometry.pkl")
DISCOVERY_PRIOR = Path("data/clean/discovery_prior.pkl")

RESULTS_BASE = Path("decision_simulator/results")


@dataclass
class BaseDecisionConfig:
    drilling_budget: int = 10
    initial_random_drills: int = 3
    mine_threshold: float = 0.7


@dataclass
class GreedyConfig(BaseDecisionConfig):
    k_neighbors: int = 5


@dataclass
class ParticleBeliefConfig(BaseDecisionConfig):
    n_particles: int = 50
    temperature: float = 0.1
