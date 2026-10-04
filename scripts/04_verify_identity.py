"""Verify the exact identity that makes this a dip-sequence problem.

On every training well, for each of the six marker columns,

    TVT(md) + Z(md) - marker(md)  =  constant per well

i.e. U = TVT + Z is the marker surface shifted by a per-well offset. This script
reports the within-well standard deviation of that residual (the data are recorded
at 0.05 ft resolution) and checks that the markers are conformal within a well,
i.e. that the spacing between any two of them is constant along the lateral.
Rows where a marker column is missing are skipped.

Run:  python scripts/04_verify_identity.py  ->  results/identity.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rogii.io import FORMATION_COLS, load_split
from rogii.paths import DATA_DIR, RESULTS_DIR


def main():
    wells = load_split(DATA_DIR, "train")
    per_marker = {c: [] for c in FORMATION_COLS}
    conformal = []
    for wid in sorted(wells):
        w = wells[wid]
        u = w.tvt + w.z
        for c in FORMATION_COLS:
            r = u - w.formations[c]
            if np.isfinite(r).sum() > 10:
                per_marker[c].append(float(np.nanstd(r)))
        # conformality: marker-to-marker spacing is constant along the well
        spacing = [np.nanstd(w.formations[a] - w.formations[b])
                   for a, b in zip(FORMATION_COLS[:-1], FORMATION_COLS[1:])]
        spacing = [x for x in spacing if np.isfinite(x)]
        if spacing:
            conformal.append(float(max(spacing)))

    out = {
        "n_wells": len(wells),
        "within_well_std_of_TVT_plus_Z_minus_marker_ft": {
            c: {"n_wells": len(v), "median": float(np.median(v)), "p99": float(np.percentile(v, 99)),
                "max": float(np.max(v))} for c, v in per_marker.items()},
        "max_within_well_std_of_marker_spacing_ft": {
            "median": float(np.median(conformal)),
            "max": float(np.max(conformal))},
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "identity.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
