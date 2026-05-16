#!/usr/bin/env python
"""Numbered entry-point for autoencoder training.

The actual implementation lives in `train_encoder.py` so other modules
(`4_pull_maps.py`, `5a_train_jepa.py`, `analysis_simulation.py`,
`simulator/smoke_test.py`) can `import` from it — Python module names
can't start with a digit.

Run with:
    python 5b_train_encoder.py [--flags ...]
"""
from train_encoder import main

if __name__ == "__main__":
    main()
