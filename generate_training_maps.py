from pathlib import Path
from decision_simulator.neural_belief.colab import pull_belief_maps_from_colab

pool_path = pull_belief_maps_from_colab(
    storage_root=Path(__file__).parent,
    n_maps=100,
    out="C:/dataset_raw_maps/raw_pool.h5",
)
print(f"\nDone. Pool saved to: {pool_path}")
