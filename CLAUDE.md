# Project notes for Claude

## Terminology

**evolution map** — the multi-panel grid saved by `plot_evolution_map`
(`decision_simulator/utils/plotting.py`). Each row is one selected drill step,
rendered as 5 columns: drill locations | true ore | predicted ore | uncertainty | absolute error.
Steps are selected from `SELECTED_STEPS = {1, 2, 3, 5, 8, 10}` in `pomdp.py`.

Do NOT confuse with `plot_policy_evolution` (a line chart of total predicted ore vs steps).
That function exists in `plotting.py` but is intentionally unused and must not be
re-introduced into `pomdp.py` or called when the user mentions "evolution map",
"evolution graph", or anything similar.
