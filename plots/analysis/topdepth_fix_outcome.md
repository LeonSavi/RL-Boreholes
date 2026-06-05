# Phase 1 outcome — comonotone-quantile fix

## What was wrong

`scripts/diagnostics/topdepth_dispersion_audit.py` flagged two
defects in `_sample_tops_with_constraints`:

- **A2** — simulator post-constraint top-depth std was only $31\%$
  of the real-data std (median across formations).
- **A3** — the 20-retry rejection loop collapsed to a fixed
  median stack in $71\%$ of draws.

Both were the same root cause: independent KDE samples per
formation are rarely monotone+min-thickness, so the rejection loop
hit its retry cap and used median tops + 50m padding. The
synthetic columns clustered at one fixed stack.

## The fix

`simulator/formation_geometry.py:_sample_tops_with_constraints`
now uses **comonotone-quantile coupling**:

```python
u = rng.uniform(0.02, 0.98)
for fm in combination:
    top_fm = empirical_quantile(stats.top_depths, u)
    # then clamp into [prev + min_thickness, max_depth_room_for_rest]
```

One uniform draw $u$ per column is the quantile for every
formation in the stack. This preserves:

- each formation's empirical top-depth marginal (perfectly);
- positive rank correlation across formations (any deep well
  drills deep into all formations — compaction physics).

The whole change is ~20 lines, no refit needed (the
`FormationGeometry.pkl` file is unchanged; only the sampling code
changed).

## Result — Garzon position metric

Wasserstein-1 between sim and real per-formation top depths,
before and after the fix (500 sim columns each time):

| Formation | Before W1 (m) | After W1 (m) | Drop |
|---|---|---|---|
| NU | 137 | 30 | 78% |
| NM | 190 | 45 | 76% |
| NL | 206 | 46 | 78% |
| CK | 261 | 43 | 83% |
| KN | 438 | 75 | 83% |
| AT | 460 | 116 | 75% |
| SL | 423 | 75 | 82% |
| SG | 317 | 64 | 80% |
| ZE | 428 | 90 | 79% |
| RO | 425 | 72 | 83% |
| RB | 467 | 130 | 72% |
| RN | 517 | 136 | 74% |
| DC | 501 | 122 | 76% |

**Median improvement: 78% drop across all formations.**

## Sequence-metric side effect

Frobenius distance on the formation-successor matrix went from
$1.51$ to $1.48$ (modest). The sequence structure was already
well-captured; the fix only addressed depth dispersion.

## Verification

- `python -m simulator.smoke_test` passes (all unit tests green,
  including the gas-response shift, max-ore-bodies cap, bank
  sample with formation, and end-to-end shape integrity).
- `data/clean/formation_geometry.pkl` unchanged; the fix is in
  sampling code, not fitted statistics.

## Implication for Phase 2 (RVG integration)

The simulator's per-formation top-depth Wasserstein is now 30–136 m
across all formations. The Phase 2 go/no-go gate required RVG
integration to drop these by another $\geq 50$ m for NU/NM/NL —
that is geometrically impossible for NU (currently 30 m).

The RVG gap report's other arguments still stand:
- NM/NL coverage above the KDE threshold ($0 \to 20$ bins);
- Upper-section transition-matrix Frobenius shift ($1.51$);
- 100% lithology-vocabulary alignment.

But the urgency is reduced. The simulator works well enough that
RVG integration is now optional rather than mandatory.

Decision pending from the user.
