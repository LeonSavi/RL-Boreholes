from __future__ import annotations

import warnings
import numpy as np
import torch

from decision_simulator.neural_belief.models.belief_models.borehole_encoders.autoencoder import standardise
from decision_simulator.resources import DecisionSimulationResources
from decision_simulator.typing import DrillObservation

# Tracks which missing variables have already been warned about this session.
_warned_missing_vars: set[str] = set()


def extract_borehole(
    true_map: dict,
    i: int,
    j: int,
    variables: list[str],
) -> np.ndarray:
    """Extract the raw multi-variable borehole profile at grid location (i, j).

    Stacks one depth profile per variable into a single array, preserving
    the variable ordering required by the encoder.

    Parameters
    ----------
    true_map  : simulator output dict containing a "variables" sub-dict
    i, j      : grid row and column indices
    variables : ordered list of variable names to extract

    Returns
    -------
    np.ndarray of shape (V, D) — V variables, D depth layers, float32

    Notes
    -----
    If a variable in ``variables`` is absent from the map (e.g., the JEPA
    checkpoint was trained with ``pef`` but the current SimConfig no longer
    generates it), a zero channel is substituted and a UserWarning is emitted
    once per missing variable name.  Zero is appropriate for z-scored inputs
    (it represents the training mean), but may bias embeddings if the encoder
    saw a non-zero baseline for that variable during training.
    """
    map_vars = true_map["variables"]
    n_depth = next(iter(map_vars.values())).shape[2]

    channels: list[np.ndarray] = []
    missing: list[str] = []
    for v in variables:
        if v in map_vars:
            channels.append(map_vars[v][i, j, :])
        else:
            channels.append(np.zeros(n_depth, dtype=np.float32))
            missing.append(v)

    new_missing = [v for v in missing if v not in _warned_missing_vars]
    if new_missing:
        _warned_missing_vars.update(new_missing)
        warnings.warn(
            f"extract_borehole: variable(s) {new_missing} not found in the "
            f"generated map. This usually means the JEPA checkpoint was trained "
            f"with a different SimConfig.variables than the one currently in use "
            f"(e.g., 'pef' was removed). Substituting zeros for missing channels. "
            f"Map has: {sorted(map_vars.keys())}. "
            f"Consider retraining the encoder with the current variable set.",
            UserWarning,
            stacklevel=2,
        )

    return np.stack(channels, axis=0).astype(np.float32)


def standardise_borehole(
    borehole: np.ndarray,
    stats: dict[str, tuple[float, float]],
    variables: list[str],
) -> np.ndarray:
    """Z-score normalise a raw borehole profile using training statistics.

    Each variable channel is shifted by its training mean and divided by its
    training standard deviation, then clipped to ±4 sigma to limit the
    influence of extreme values.

    Parameters
    ----------
    borehole  : (V, D) raw borehole array as returned by extract_borehole
    stats     : mapping from variable name to (mean, std) training statistics
    variables : ordered variable names matching the channel axis of borehole

    Returns
    -------
    np.ndarray of shape (V, D), standardised float32
    """
    return standardise(borehole, stats, variables)


def encode_borehole(
    model,
    borehole_std: np.ndarray,
    device: str,
) -> np.ndarray:
    """Encode a standardised borehole profile into a compact latent vector.

    Passes the borehole through the JEPA encoder to produce a fixed-length
    embedding that captures the geophysical signature of the location.
    NaN values (missing measurements) are replaced with zero before encoding.

    Parameters
    ----------
    model        : trained JEPA encoder with an embed() method
    borehole_std : (V, D) standardised borehole array
    device       : torch device string, e.g. "cpu" or "cuda"

    Returns
    -------
    np.ndarray of shape (D_latent,) — the latent embedding, on CPU
    """
    borehole_tensor = (
        torch.from_numpy(np.nan_to_num(borehole_std, nan=0.0)).unsqueeze(0).to(device)
    )  # (1, V, D)
    with torch.no_grad():
        latent_batch = model.embed(borehole_tensor)  # (1, D_latent)
    return latent_batch.squeeze(0).cpu().numpy()


def compute_ore_value(true_map: dict, i: int, j: int) -> float:
    """Return the peak ore yield across all depth layers at location (i, j).

    Used as the ground-truth reward signal for a drilled borehole.

    Parameters
    ----------
    true_map : simulator output dict containing a "yield_field" array
    i, j     : grid row and column indices

    Returns
    -------
    float — maximum ore yield across the depth dimension
    """
    return float(true_map["yield_field"][i, j, :].max())


def drill_at(
    loc: tuple[int, int],
    step: int,
    true_map: dict,
    resources: DecisionSimulationResources,
    device: str,
    pred_ore: float | None = None,
    verbose: bool = True,
) -> DrillObservation:
    """Simulate drilling a borehole at a candidate location and return an observation.

    Conceptually, drilling a borehole at a location reveals the full depth
    profile of geophysical measurements at that point. These measurements are
    standardised and encoded into a latent vector by the JEPA encoder, which
    compresses the high-dimensional profile into a compact representation
    suitable for downstream decision-making. The true ore value at the location
    is also recorded as the ground-truth reward signal.

    Parameters
    ----------
    loc       : (i, j) grid coordinates to drill
    step      : drill step index (1-based), used for logging and result tracking
    true_map  : simulator output dict (ground truth map)
    resources : shared experiment resources (encoder, normalisation stats, etc.)
    device    : torch device string
    pred_ore  : predicted ore value used to select this location, or None for
                random exploration steps
    verbose   : if True, print a one-line summary of the drill result

    Returns
    -------
    dict with keys:
        location      : (i, j) tuple
        latent        : (D_latent,) JEPA embedding of the borehole profile
        ore_value     : ground-truth peak ore yield at this location
        predicted_ore : pred_ore argument (None for random steps → NaN in parquet)
        step          : drill step index
    """
    i, j = loc
    borehole = extract_borehole(true_map, i, j, resources.variable_names)
    borehole_std = standardise_borehole(
        borehole, resources.norm_stats, resources.variable_names
    )
    latent = encode_borehole(resources.jepa_model, borehole_std, device)
    ore_value = compute_ore_value(true_map, i, j)

    if verbose:
        if pred_ore is not None:
            print(
                f"  step={step:02d}  loc=({i:2d},{j:2d})  pred={pred_ore:.4f}  true_ore={ore_value:.4f}"
            )
        else:
            print(
                f"  step={step:02d}  loc=({i:2d},{j:2d})  [random]        true_ore={ore_value:.4f}"
            )

    return {
        "location": (i, j),
        "latent": latent,
        "ore_value": ore_value,
        "predicted_ore": pred_ore,  # None for random steps -> NaN in parquet
        "step": step,
    }


def run_initial_random_drills(
    all_candidate_borehole_coords: list[list[int, int]],
    n_initial_drills: int,
    rng: np.random.Generator,
    drill_kwargs: dict,
    verbose: bool = True,
) -> tuple[list[DrillObservation], list[list[int, int]]]:
    """Randomly select and drill n_initial_drills unique locations.

    Used at the start of every simulation to collect an initial set of
    observations before any informed selection strategy takes over.

    Parameters
    ----------
    all_candidate_borehole_coords   : full list of (i, j) grid locations
    n_initial_drills : number of locations to drill randomly
    rng              : seeded numpy random generator (ensures reproducibility)
    drill_kwargs     : keyword arguments forwarded to drill_at (true_map,
                       resources, device, verbose)

    Returns
    -------
    observations : list[dict]        — one observation dict per drilled location
    unvisited    : list[list[int,int]] — remaining undrilled candidates
    """

    if verbose:
        print(f"\nPhase 1 - random initialisation ({n_initial_drills} drills)")
    init_indices = rng.choice(
        len(all_candidate_borehole_coords), size=n_initial_drills, replace=False
    )
    observations: list[DrillObservation] = []
    drilled: set[tuple[int, int]] = set()
    for step, idx in enumerate(init_indices, start=1):
        loc = all_candidate_borehole_coords[int(idx)]
        obs = drill_at(loc, step, **drill_kwargs)
        observations.append(obs)
        drilled.add(tuple(loc))
    unvisited = [c for c in all_candidate_borehole_coords if tuple(c) not in drilled]
    return observations, unvisited


# ---------------------------------------------------------------------------
# POMDP observation state
# ---------------------------------------------------------------------------


class BoreholeObservationState:
    """Tracks drilled borehole observations for the POMDP belief model.

    Responsibilities
    ----------------
    - Store drilled positions, standardised borehole profiles, and raw ore values.
    - Prevent duplicate drilling.
    - Build model-ready input tensors via to_model_inputs().

    Design note
    -----------
    Ore values are stored raw (in ore-value space). The NeuralBeliefUpdater
    applies target normalisation before passing them to the belief model.
    Borehole profiles are stored already standardised (z-scored per variable)
    as returned by DrillingEnvironment.drill().
    """

    def __init__(self, n_x: int, n_y: int) -> None:
        self.n_x = n_x
        self.n_y = n_y
        self._positions: list[tuple[int, int]] = []
        self._boreholes: list[np.ndarray] = []  # each (V, D), standardised
        self._ore_values: list[float] = []
        self._observed: set[tuple[int, int]] = set()

    def add_observation(
        self,
        i: int,
        j: int,
        borehole: np.ndarray,
        ore_value: float,
    ) -> None:
        """Record a new drilled borehole observation.

        Parameters
        ----------
        i, j      : grid coordinates of the drilled cell
        borehole  : (V, D) float32 standardised borehole profile
        ore_value : raw peak ore yield at this location

        Raises
        ------
        ValueError if (i, j) has already been observed.
        """
        if (i, j) in self._observed:
            raise ValueError(
                f"Location ({i}, {j}) has already been drilled. "
                "Duplicate observations are not allowed."
            )
        self._positions.append((i, j))
        self._boreholes.append(borehole)
        self._ore_values.append(float(ore_value))
        self._observed.add((i, j))

    def is_observed(self, i: int, j: int) -> bool:
        """Return True if cell (i, j) has already been drilled."""
        return (i, j) in self._observed

    def get_observed_mask(self) -> np.ndarray:
        """Return a (n_x, n_y) bool array; True where a cell has been drilled."""
        mask = np.zeros((self.n_x, self.n_y), dtype=bool)
        for i, j in self._positions:
            mask[i, j] = True
        return mask

    def to_model_inputs(self) -> dict:
        """Build model-ready input arrays from current observations.

        Returns
        -------
        dict with keys:
            boreholes    : (K, V, D) float32 — standardised borehole profiles
            ore_vals     : (K,)      float32 — raw ore values (caller normalises)
            positions    : (K, 2)    float32 — normalised [0, 1] grid coordinates;
                           positions[k] = [i / (n_x - 1), j / (n_y - 1)]
            padding_mask : (K,)      bool    — all False (no padding at inference)

        Raises
        ------
        ValueError if no observations have been recorded yet.
        """
        K = len(self._positions)
        if K == 0:
            raise ValueError(
                "to_model_inputs() called with no observations. "
                "Drill at least one borehole first."
            )

        boreholes = np.stack(self._boreholes, axis=0)  # (K, V, D)
        ore_vals = np.array(self._ore_values, dtype=np.float32)  # (K,)

        positions = np.zeros((K, 2), dtype=np.float32)
        for k, (i, j) in enumerate(self._positions):
            positions[k, 0] = i / (self.n_x - 1)
            positions[k, 1] = j / (self.n_y - 1)

        padding_mask = np.zeros(K, dtype=bool)  # no padding at single-sample inference

        return {
            "boreholes": boreholes,
            "ore_vals": ore_vals,
            "positions": positions,
            "padding_mask": padding_mask,
        }

    def get_sparse_ore_map(self) -> np.ndarray:
        """Return a (n_x, n_y) float32 array with raw ore values at drilled cells.

        Undrilled cells are zero. Useful as the first panel of per-step plots.
        """
        sparse = np.zeros((self.n_x, self.n_y), dtype=np.float32)
        for (i, j), ore in zip(self._positions, self._ore_values):
            sparse[i, j] = ore
        return sparse

    @property
    def observed_ore_values(self) -> list[float]:
        """Raw ore values at all drilled locations, in drill order."""
        return list(self._ore_values)

    @property
    def n_drills(self) -> int:
        """Number of boreholes drilled so far."""
        return len(self._positions)
