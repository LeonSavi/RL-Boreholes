# planners/

This folder is reserved for future non-myopic planning algorithms.

Planned implementations:

- **MCTS** (Monte Carlo Tree Search) — tree-based lookahead over drilling sequences
- **POMCP** (Partially Observable Monte Carlo Planning) — MCTS extended to POMDPs via particle belief trees
- **Rollout planners** — policy rollout with a fixed simulation horizon
- **Non-myopic greedy** — multi-step lookahead greedy that accounts for information value of future drills

No implementations exist yet. Current policies (greedy, particle belief) are myopic — they select the single best next location without lookahead.
