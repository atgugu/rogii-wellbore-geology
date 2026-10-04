"""Truth-aided oracle ladder.

Community reference on the same 773-well eval zones (712037 / 726465):
carry-last ~15.9 | best constant ~9.04 | best line ~6.70 | quadratic ~5.34
| smooth low-frequency ~2.90-3.05. Reproducing these validates mask + scorer.
"""
from __future__ import annotations

import numpy as np

from .io import Well


def _polyfit_oracle(md: np.ndarray, y: np.ndarray, deg: int) -> np.ndarray:
    s = (md - md.mean()) / max(md.std(), 1e-9)
    coef = np.polyfit(s, y, deg)
    return np.polyval(coef, s)


def _smooth_oracle(y: np.ndarray, window_ft: int = 1201) -> np.ndarray:
    """Low-frequency component of truth: centered moving average with
    reflected edges (window in rows == ft at 1 ft sampling).
    1201 ft calibrated so the pooled rung lands ~3.0 = community reference."""
    n = len(y)
    w = min(window_ft, n if n % 2 == 1 else n - 1)
    if w < 5:
        return y.copy()
    if w % 2 == 0:
        w -= 1
    pad = w // 2
    yp = np.pad(y, pad, mode="reflect")
    kernel = np.ones(w) / w
    return np.convolve(yp, kernel, mode="valid")


def oracle_predictions(well: Well, kind: str) -> np.ndarray:
    m = well.eval_mask
    y = well.tvt[m]
    md = well.md[m]
    if kind == "carry_last":
        return np.full(y.shape, well.last_known_tvt)
    if kind == "constant":
        return np.full(y.shape, y.mean())
    if kind == "line":
        return _polyfit_oracle(md, y, 1)
    if kind == "quadratic":
        return _polyfit_oracle(md, y, 2)
    if kind == "smooth":
        return _smooth_oracle(y)
    raise ValueError(kind)


LADDER = ["carry_last", "constant", "line", "quadratic", "smooth"]
