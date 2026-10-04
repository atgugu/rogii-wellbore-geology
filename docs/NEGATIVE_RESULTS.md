# What was tried and did not work

The most reusable part of the project. Each line is a hypothesis, the evidence against it, and the mechanism as
far as it was identified. All numbers are recorded during the competition unless a script is named. "Board" means
the public leaderboard, against an unchanged re-run of the same pipeline (run-to-run σ about 0.03 ft).

## Operators fitted against an offline residual

| idea | offline | board | mechanism |
|---|---|---|---|
| Cross-well marker surface: a ridge plane through the 10 nearest *distinct* wells, gradient-capped | −0.46 | +0.33 (clipped ±10 ft) to +1.44 (unclipped) | The gain was −0.46 under leave-one-well-out and 0.00 under leave-spatial-block-out; I judged the first to be the relevant regime and shipped it. The board result was worse than either, and a bound on the correction mattered far more there than offline. |
| Steering dip from the trajectory (α = 0.4 / 0.75 / 1.1) | −0.07 / −0.10 / −0.13 | +0.08 / +0.09 / +0.21 | Monotone in strength: a systematically wrong correction. Being within-well did not make it transfer. |
| Projection augmentation | favourable | +0.16 to +0.30 | Fitted on the harness residual (below). |

Common cause: the offline harness measured a pipeline about 1.45 ft worse than the one deployed (7.73 vs about
6.28). A correction trained to remove that residual removes something the deployed pipeline had already removed and
injects the difference. Shrinking the coefficient scales the damage; it does not fix the sign. **Rule adopted: an
operator fitted on a proxy residual needs a monotone strength ladder on held-out data before it carries any
weight.**

## Better models that lose

| idea | result | mechanism |
|---|---|---|
| Whole-well bidirectional sequence model for the dip | 7.95 vs 7.73 for the classical base over 771 wells; better on 46% of wells | Promising interim numbers were one fold with one seed, or leaked through a third-party alias. Ensembling 1→4 models gained 0.33 and diminished fast. |
| Flexible DP over the GR log | 15.83 vs 13.91 (carry-last), 140 wells | It finds paths that fit GR better than the truth on 72% of wells. The optimiser is right; the objective's minimum is not at the truth. |
| Total-variation dip DP (the correct prior for human-drawn polylines) | 0.25 ft on synthetic data, 30.3 on real | Verified correct on synthetic data. A large search space contains many low-cost impostors. |
| GR re-ranking of five real candidate tracks | 61–63% pairwise, no blend gain | The value of choosing and the accuracy needed both grow with the spread of the candidates; they cancel. |
| A cat/xgb model as a third ensemble component | optimal weight ≈ 0.08, gain ≈ 0.03 | Its features are aggregates of the signals the stack already used; diversity comes from different *information*, not a different model class on the same features. |
| Delaunay triangulation instead of IDW for the surface | −0.07 offline | Dropped with the surface operator above. |

## Calibration and tuning that did not transfer

| idea | result | mechanism |
|---|---|---|
| Blend weight chosen out of fold | best offline at a *negative* weight (5.45); hidden worst (6.99 vs 6.42 at 0.60) | The learned track was fitted on all training wells, so every out-of-fold number involving it was an upper bound. |
| Single-knob probes of the final pipeline (hedge strength, delta-corrector strength, seeds, …) | 22 of 22 probes at or above the family mean | Downstream components were calibrated at the incumbent configuration (see the README). |
| 256 selector seeds | +0.07 to +0.10, depending on the control | Interacts with the delta corrector, which had been fitted at the incumbent seed count. |
| Seed count beyond 128 | no gain | The selector is a softmax-weighted mean, not an argmax; a datum-mode flip was never observed (0 in 21,120 runs). |
| Outlier-well fallbacks | none beat the base | Catastrophic wells are detectable (4.1× lift in the top decile) but every fallback (flat-U, linear-U, heavy smoothing) is worse than the base even there. |
| Blending the strongest public notebooks | no gain | Pulled and hashed 8 of the best: 40–45 of 47 cells identical; independently built engines correlated at about 0.89. |
| Tuning a blend partner on its own RMSE | worse blend | Tune on the blend; stronger regularisation decorrelates the partner. (The best standalone partner gave the worse blend: 9.1057 vs 9.1001.) |

## What survived

- A distance-gated neighbour correction of `U` from same-family wells (about −0.07 on the board).
- Smoothing in `U` rather than in `TVT`.
- Bounding any cross-well correction.
- Configuration choices to the existing pipeline were the only changes whose sign the board did not reverse, and
  even those reached a local optimum on every coordinate that could be measured.
