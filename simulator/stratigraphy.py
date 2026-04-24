"""
Stratigraphic column — stage 1b.

Defines the Dutch stratigraphic ordering (oldest at bottom, youngest at top),
picks a sequence of formations with prevalence fit from NLOG, and assigns
each formation to a fine-rock-type for variable sampling.

The "fine rock type" for each formation is chosen from a predefined set of
lithological candidates per formation.  E.g. a Rotliegend cell gets either
"sandstone" (Slochteren-like) or "claystone" (Ten Boer-like) based on
facies probability.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


# Dutch stratigraphic column, youngest at top (index 0) -> oldest at bottom.
# For each formation: its NLOG code, typical thickness range (m), and the
# fine rock types it can contain with their relative probabilities.
#
# Probabilities are based on NLOG coverage: e.g. in RO, Slochteren sandstone
# is more common than Ten Boer claystone.  These are approximate; tune if
# you refit from data later.
DUTCH_COLUMN = [
    # (formation, thickness_mean, thickness_std, {rock_type: probability})
    #
    # Rock type facies probabilities updated for the refined
    # rock_type_fine scheme (v5): sandstone → sandstone_clean /
    # sandstone_shaly, claystone → claystone_cool / claystone_hot,
    # halite → halite_pure, carbonate → dolomite for ZE/RN where the
    # carbonates are dolomitic.  Probabilities are set from the
    # stratigraphic interpretation — e.g. Rotliegend is mostly
    # Slochteren clean reservoir sand with some Ten Boer hot shale.
    ("NU",  300,  150, {"clay": 0.85, "sandstone_shaly": 0.15}),
    ("NM",   80,   40, {"clay": 0.95, "sandstone_shaly": 0.05}),
    ("NL",  200,  100, {"clay": 0.70, "sandstone_shaly": 0.30}),
    ("CK",  400,  200, {"chalk": 0.95, "claystone_cool": 0.05}),
    ("KN",  250,  150, {"claystone_cool": 0.55, "claystone": 0.25,
                       "sandstone_shaly": 0.20}),
    ("SL",   80,   50, {"claystone_cool": 0.80, "claystone": 0.15,
                       "sandstone_shaly": 0.05}),
    ("SG",   40,   30, {"claystone_hot": 0.50, "claystone_cool": 0.40,
                       "sandstone_shaly": 0.10}),  # Kimmeridge-like source rocks
    ("AT",   80,   50, {"claystone_cool": 0.60, "claystone_hot": 0.35,
                       "sandstone_shaly": 0.05}),  # Altena
    ("RN",  150,   80, {"claystone": 0.45, "dolomite": 0.30,
                       "anhydrite": 0.10, "sandstone_shaly": 0.15}),
    ("RB",  300,  150, {"sandstone_shaly": 0.60, "claystone_hot": 0.30,
                       "sandstone_clean": 0.10}),  # Buntsandstein mostly muddy
    ("ZE",  600,  400, {"halite_pure": 0.50, "anhydrite": 0.20,
                       "dolomite": 0.25, "claystone": 0.05}),
    ("RO",  400,  200, {"sandstone_clean": 0.55, "claystone_hot": 0.35,
                       "sandstone_shaly": 0.10}),  # Slochteren + Ten Boer
    ("DC",  500,  250, {"claystone_hot": 0.80, "sandstone_shaly": 0.20}),  # coal measures
]


@dataclass
class StratigraphicColumn:
    """One realisation of a vertical stratigraphic profile — an ordered list
    of layers with (formation, rock_type, depth_top, depth_bottom)."""
    layers: list[tuple[str, str, float, float]]

    def rock_type_at(self, depth: float) -> str | None:
        """Which fine rock type occupies this depth."""
        for _, rock, top, bot in self.layers:
            if top <= depth < bot:
                return rock
        return None

    def formation_at(self, depth: float) -> str | None:
        for fm, _, top, bot in self.layers:
            if top <= depth < bot:
                return fm
        return None

    @property
    def total_depth(self) -> float:
        return max(bot for _, _, _, bot in self.layers)

    def rasterise(self, depth_array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (rock_types, formations) arrays aligned to depth_array."""
        rocks = np.empty(len(depth_array), dtype=object)
        forms = np.empty(len(depth_array), dtype=object)
        for i, d in enumerate(depth_array):
            rocks[i] = self.rock_type_at(d) or "other"
            forms[i] = self.formation_at(d) or "other"
        return rocks, forms


def sample_column(
    rng: np.random.Generator,
    max_depth: float = 3000.0,
    p_formation_present: float = 0.75,
) -> StratigraphicColumn:
    """Sample one vertical stratigraphic column.

    Each formation in the Dutch column is included with probability
    `p_formation_present` (matches NLOG: not all wells see every formation).
    Thickness is sampled from a clamped normal; rock type per layer is
    drawn from that formation's facies distribution.

    Guarantees that the returned column extends to max_depth (extending
    the deepest layer if the sampled formations run short).
    """
    layers = []
    current_depth = 0.0
    for fm, t_mean, t_std, facies in DUTCH_COLUMN:
        if rng.random() > p_formation_present:
            continue
        thickness = max(10.0, rng.normal(t_mean, t_std))
        top = current_depth
        bot = min(current_depth + thickness, max_depth)
        if bot <= top:
            break
        rocks, probs = zip(*facies.items())
        rock = rng.choice(rocks, p=np.array(probs) / sum(probs))
        layers.append((fm, str(rock), top, bot))
        current_depth = bot
        if current_depth >= max_depth:
            break

    # extend the deepest layer to max_depth if we finished early.  real
    # wells bottom out somewhere; for the simulator we want every column
    # filled so downstream (x,y,z) tensors have no "other" gaps.
    if layers and current_depth < max_depth:
        fm, rock, top, _ = layers[-1]
        layers[-1] = (fm, rock, top, max_depth)
    elif not layers:
        # vanishingly unlikely, but handle: pick the deepest formation
        fm, t_mean, _, facies = DUTCH_COLUMN[-1]
        rocks, probs = zip(*facies.items())
        rock = rng.choice(rocks, p=np.array(probs) / sum(probs))
        layers = [(fm, str(rock), 0.0, max_depth)]

    return StratigraphicColumn(layers=layers)


def sample_spatial_column_field(
    rng: np.random.Generator,
    n_x: int,
    n_y: int,
    max_depth: float = 3000.0,
    layer_waviness: float = 20.0,
) -> list[list[StratigraphicColumn]]:
    """Sample a 2D (x, y) grid of stratigraphic columns with lateral
    continuity.

    Strategy: pick a single base column, then perturb layer boundaries
    spatially so neighbouring cells have similar but not identical layer
    depths.  Later we'll use gstools for variable values; layer geometry
    uses a simpler per-boundary 2D wiggle.

    Returns a nested list `columns[x][y]` of StratigraphicColumn objects.
    """
    base = sample_column(rng, max_depth=max_depth)
    n_layers = len(base.layers)

    # sample a smooth 2D perturbation field for each layer boundary.
    # boundaries shift by up to ±layer_waviness metres across the grid.
    boundary_perturbations = np.zeros((n_layers + 1, n_x, n_y))
    for i in range(1, n_layers):
        # low-frequency 2D sinusoidal perturbation
        kx = rng.uniform(0.05, 0.3)
        ky = rng.uniform(0.05, 0.3)
        phase = rng.uniform(0, 2 * np.pi)
        ampl = rng.uniform(0.4, 1.0) * layer_waviness
        xs = np.arange(n_x)[:, None]
        ys = np.arange(n_y)[None, :]
        boundary_perturbations[i] = ampl * np.sin(kx * xs + ky * ys + phase)

    columns = []
    for x in range(n_x):
        row = []
        for y in range(n_y):
            new_layers = []
            current = 0.0
            for li, (fm, rock, top, bot) in enumerate(base.layers):
                perturbed_bot = bot + boundary_perturbations[li + 1, x, y]
                perturbed_bot = max(current + 5.0, perturbed_bot)  # positive thickness
                new_layers.append((fm, rock, current, perturbed_bot))
                current = perturbed_bot
            # extend deepest layer to max_depth so no "other" gaps at bottom
            if new_layers and current < max_depth:
                fm, rock, top, _ = new_layers[-1]
                new_layers[-1] = (fm, rock, top, max_depth)
            row.append(StratigraphicColumn(layers=new_layers))
        columns.append(row)
    return columns