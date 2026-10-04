"""Data loading with a parquet cache.

Well files are ~600 KB CSVs x 773; first load builds two parquet bundles
(horizontal + typewell) so subsequent loads take ~1 s.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

HW_SUFFIX = "__horizontal_well.csv"
TW_SUFFIX = "__typewell.csv"
FORMATION_COLS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]


@dataclass
class Well:
    """One lateral + its typewell. Arrays are row-aligned to the source CSV."""

    well_id: str
    md: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    gr: np.ndarray
    tvt_input: np.ndarray          # NaN on the eval zone
    tvt: np.ndarray | None         # None on real test wells
    formations: dict[str, np.ndarray] = field(default_factory=dict)
    tw_tvt: np.ndarray = None
    tw_gr: np.ndarray = None
    tw_geology: np.ndarray = None

    # ---- mask semantics (single source of truth) ----
    @property
    def eval_mask(self) -> np.ndarray:
        """Rows to predict: TVT_input is NaN. Matches sample_submission ids."""
        return np.isnan(self.tvt_input)

    @property
    def ps_index(self) -> int:
        """Prediction start = first eval row (prefix verified contiguous)."""
        m = self.eval_mask
        return int(np.argmax(m)) if m.any() else len(self.md)

    @property
    def last_known_tvt(self) -> float:
        i = self.ps_index
        return float(self.tvt_input[i - 1]) if i > 0 else float("nan")

    @property
    def n_eval(self) -> int:
        return int(self.eval_mask.sum())

    def eval_ids(self) -> list[str]:
        return [f"{self.well_id}_{i}" for i in np.flatnonzero(self.eval_mask)]

    # ---- derived ----
    @property
    def u_input(self) -> np.ndarray:
        """U = TVT + Z on the known prefix (NaN on eval zone)."""
        return self.tvt_input + self.z

    def tw_gr_at(self, tvt: np.ndarray) -> np.ndarray:
        """Typewell GR sampled at stratigraphic positions (edge-filled)."""
        return np.interp(tvt, self.tw_tvt, self.tw_gr)


def _bundle_paths(data_dir: str, split: str) -> tuple[str, str]:
    cache = os.path.join(data_dir, "_cache")
    os.makedirs(cache, exist_ok=True)
    return (os.path.join(cache, f"{split}_hw.parquet"),
            os.path.join(cache, f"{split}_tw.parquet"))


def _build_bundle(data_dir: str, split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    src = os.path.join(data_dir, split)
    hw_frames, tw_frames = [], []
    for fn in sorted(os.listdir(src)):
        if fn.endswith(HW_SUFFIX):
            wid = fn[: -len(HW_SUFFIX)]
            df = pd.read_csv(os.path.join(src, fn))
            df.insert(0, "WELLNAME", wid)
            hw_frames.append(df)
        elif fn.endswith(TW_SUFFIX):
            wid = fn[: -len(TW_SUFFIX)]
            df = pd.read_csv(os.path.join(src, fn))
            df.insert(0, "WELLNAME", wid)
            tw_frames.append(df)
    hw = pd.concat(hw_frames, ignore_index=True)
    tw = pd.concat(tw_frames, ignore_index=True)
    return hw, tw


def load_split(data_dir: str, split: str = "train", use_cache: bool = True
               ) -> dict[str, Well]:
    """Load every well of a split as {well_id: Well}."""
    hw_p, tw_p = _bundle_paths(data_dir, split)
    if use_cache and os.path.exists(hw_p) and os.path.exists(tw_p):
        hw, tw = pd.read_parquet(hw_p), pd.read_parquet(tw_p)
    else:
        hw, tw = _build_bundle(data_dir, split)
        if use_cache:
            hw.to_parquet(hw_p), tw.to_parquet(tw_p)

    wells: dict[str, Well] = {}
    tw_groups = dict(iter(tw.groupby("WELLNAME", sort=False)))
    for wid, g in hw.groupby("WELLNAME", sort=False):
        t = tw_groups[wid].sort_values("TVT")
        has_tvt = "TVT" in g.columns and g["TVT"].notna().any()
        formations = {c: g[c].to_numpy() for c in FORMATION_COLS if c in g.columns}
        geol = t["Geology"].to_numpy() if "Geology" in t.columns else None
        wells[wid] = Well(
            well_id=wid,
            md=g["MD"].to_numpy(float),
            x=g["X"].to_numpy(float),
            y=g["Y"].to_numpy(float),
            z=g["Z"].to_numpy(float),
            gr=g["GR"].to_numpy(float),
            tvt_input=g["TVT_input"].to_numpy(float),
            tvt=g["TVT"].to_numpy(float) if has_tvt else None,
            formations=formations,
            tw_tvt=t["TVT"].to_numpy(float),
            tw_gr=t["GR"].to_numpy(float),
            tw_geology=geol,
        )
    return wells


def tw_fingerprint(w: Well) -> str:
    """Family id for byte-identical / near-identical typewells."""
    arr = np.round(w.tw_gr[:: max(1, len(w.tw_gr) // 256)], 2).tobytes()
    return hashlib.sha1(arr).hexdigest()[:12]
