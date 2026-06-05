"""Regenerate plots/map_check_v2.png from the TVD/10m/(rock,formation)
bank. The deck references this image; refresh it so the slide shows a
map made by the current pipeline.
"""
from pathlib import Path
import numpy as np
from simulator import SimConfig, generate_map, DistributionBank, DiscoveryPrior
from simulator.formation_geometry import FormationGeometry
from simulator.visualize import plot_map

OUT = Path("plots/map_check_v2.png")


def main():
    bank = DistributionBank.load("data/clean/distributions.pkl")
    geom = FormationGeometry.load("data/clean/formation_geometry.pkl")
    prior = DiscoveryPrior.load("data/clean/discovery_prior.pkl")
    rng = np.random.default_rng(42)
    m = generate_map(bank, geom, SimConfig(), rng=rng, prior=prior)
    plot_map(m, out_path=OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
