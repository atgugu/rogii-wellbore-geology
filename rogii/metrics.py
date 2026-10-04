"""Competition metric + the stress-slice report every experiment must emit.

Pooled row RMSE is the decision metric. Mean-per-well is diagnostic only
(they differ ~30% on this data). Long wells dominate: Var(MSE) ~ sum(n_w^2)/N^2.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def pooled_rmse(sq_sum: float, n: int) -> float:
    return float(np.sqrt(sq_sum / max(n, 1)))


def well_frame(per_well: dict[str, tuple[float, int]]) -> pd.DataFrame:
    """per_well: {well_id: (sum_sq_err, n_eval)} -> tidy frame."""
    df = pd.DataFrame(
        [(w, s, n, np.sqrt(s / n)) for w, (s, n) in per_well.items()],
        columns=["well", "sse", "n", "rmse"],
    )
    return df.sort_values("rmse", ascending=False).reset_index(drop=True)


def report(per_well: dict[str, tuple[float, int]]) -> dict:
    """Full stress report from per-well (sse, n)."""
    df = well_frame(per_well)
    n_total = int(df.n.sum())
    sse_total = float(df.sse.sum())
    pooled = pooled_rmse(sse_total, n_total)

    # worst-decile pooled (by per-well rmse ranking)
    k10 = max(1, len(df) // 10)
    worst10 = df.head(k10)
    k5 = max(1, len(df) // 20)
    worst5 = df.head(k5)

    long_cut = df.n.quantile(0.8)
    long_df = df[df.n >= long_cut]

    return {
        "pooled_rmse": round(pooled, 4),
        "n_rows": n_total,
        "n_wells": len(df),
        "well_rmse_median": round(float(df.rmse.median()), 4),
        "well_rmse_p90": round(float(df.rmse.quantile(0.90)), 4),
        "well_rmse_p95": round(float(df.rmse.quantile(0.95)), 4),
        "worst_decile_pooled": round(pooled_rmse(worst10.sse.sum(), int(worst10.n.sum())), 4),
        "long_well_pooled": round(pooled_rmse(long_df.sse.sum(), int(long_df.n.sum())), 4),
        "sse_share_worst5pct": round(float(worst5.sse.sum() / sse_total), 4),
        "sse_share_worst10pct": round(float(worst10.sse.sum() / sse_total), 4),
        "mean_per_well_rmse": round(float(df.rmse.mean()), 4),
    }


def score_predictions(wells: dict, preds: dict[str, np.ndarray]) -> tuple[dict, pd.DataFrame]:
    """preds: {well_id: array over the well's eval rows}. Returns (report, well_frame)."""
    per_well: dict[str, tuple[float, int]] = {}
    for wid, w in wells.items():
        if wid not in preds:
            raise KeyError(f"missing predictions for {wid}")
        m = w.eval_mask
        y = w.tvt[m]
        p = np.asarray(preds[wid], float)
        if p.shape != y.shape:
            raise ValueError(f"{wid}: pred shape {p.shape} != eval shape {y.shape}")
        if not np.isfinite(p).all():
            raise ValueError(f"{wid}: non-finite predictions")
        e = p - y
        per_well[wid] = (float(e @ e), int(m.sum()))
    return report(per_well), well_frame(per_well)
