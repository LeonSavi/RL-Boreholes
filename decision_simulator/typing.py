from typing import TypedDict
import torch


class DrillObservation(TypedDict):
    location: tuple[int, int]
    latent: torch.Tensor
    ore_value: float
    predicted_ore: float | None
    step: int
