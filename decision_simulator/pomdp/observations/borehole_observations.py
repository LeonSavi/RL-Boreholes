from __future__ import annotations

import numpy as np
import torch

from encoder.autoencoder import standardise
from decision_simulator.resources import DecisionSimulationResources
from decision_simulator.typing import DrillObservation


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
    """
    return np.stack(
        [true_map["variables"][v][i, j, :] for v in variables],
        axis=0,
    ).astype(np.float32)


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
        drilled.add(loc)
    unvisited = [c for c in all_candidate_borehole_coords if c not in drilled]
    return observations, unvisited
