"""Backward-compatibility shim for train_encoder.

train_encoder.py was renamed to 5b_train_encoder.py. Python cannot import
modules whose names start with a digit, so this shim re-exports all symbols
under the old module name. Do not add logic here; edit 5b_train_encoder.py.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "_train_encoder_impl",
    Path(__file__).parent / "5b_train_encoder.py",
)
_mod = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]

# Public API used by 4_pull_maps.py, 5a_train_jepa.py, simulator/smoke_test.py
boreholes_from_map            = _mod.boreholes_from_map
compute_standardisation_stats = _mod.compute_standardisation_stats
standardise                   = _mod.standardise
stream_batches                = _mod.stream_batches
stream_batches_from_dir       = _mod.stream_batches_from_dir
batches_per_map               = _mod.batches_per_map

# Private symbols imported by 5a_train_jepa.py
_Prefetcher   = _mod._Prefetcher
EarlyStopper  = _mod.EarlyStopper
_underlying   = _mod._underlying
PROFILE_STEPS = _mod.PROFILE_STEPS
_dump_profile = _mod._dump_profile
_TrainLog     = _mod._TrainLog
