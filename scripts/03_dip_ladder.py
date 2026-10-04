"""Oracle ladder in the DIP parameterisation: how many dip segments buy what.

The generating identity is exact on train (std 0.0069 ft = the 0.05 ft quantum):

    U(md) = TVT(md) + Z(md) = marker_elevation(X(md), Y(md)) + C_well

and the marker columns are human-drawn piecewise-linear polylines. So, anchored
on the last known prefix row, the whole task is a dip sequence:

    TVT_i = TVT_anchor - (Z_i - Z_anchor) + sum_{j<=i} dip_j * dMD_j

The trajectory term is free. This script asks the only question that matters for
model design: if an oracle handed us the best k piecewise-constant dips per
well, what would we score? That bounds every model class by its capacity to
resolve dip change points, and says how many we actually need.

Rungs are ANCHORED -- the level is pinned to the well's own known prefix, never
fitted -- so these are honest ceilings for a predictor, not free-floating fits.

Run:  python scripts/03_dip_ladder.py  ->  results/dip_ladder.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rogii.io import load_split
from rogii.paths import DATA_DIR, RESULTS_DIR

BASE_D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANCHOR = 200


def main():
    wells = load_split(DATA_DIR, "train")
    ids = sorted(wells)
    SEG = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]
    sse = {k: 0.0 for k in SEG}
    for extra in ("carry_tvt", "flat_U", "const_tvt_oracle", "lin_tvt_oracle",
                  "quad_tvt_oracle"):
        sse[extra] = 0.0
    n = 0
    lens = []

    for wid in ids:
        w = wells[wid]
        m = w.eval_mask
        ie = np.flatnonzero(m)
        ip = np.flatnonzero(~m)
        if len(ie) < 50 or len(ip) < ANCHOR:
            continue
        md, z, y = w.md[ie], w.z[ie], w.tvt[ie]
        if not (np.isfinite(md).all() and np.isfinite(z).all() and np.isfinite(y).all()):
            continue
        # anchor on the LAST known row, not a window mean: a 200-row mean of TVT
        # lags the well by ~100 ft and inflates every rung (carry 17.10 vs 15.91).
        # 15.9099 reproduces the community's convergent flat baseline exactly.
        a = ip[-1]
        md0, z0, t0 = float(w.md[a]), float(w.z[a]), float(w.tvt_input[a])
        u0 = t0 + z0
        U = y + z
        dmd = md - md0
        L = max(float(dmd[-1]), 1.0)
        lens.append(L)

        # --- trivial rungs, all anchored on the well's own prefix
        sse["carry_tvt"] += float(((t0 - y) ** 2).sum())
        sse["flat_U"] += float(((u0 - z - y) ** 2).sum())
        # oracle constant / linear / quadratic in MD, on TVT (level free = oracle)
        for name, deg in (("const_tvt_oracle", 0), ("lin_tvt_oracle", 1),
                          ("quad_tvt_oracle", 2)):
            A = np.vander(dmd / L, deg + 1)
            c, *_ = np.linalg.lstsq(A, y, rcond=None)
            sse[name] += float(((A @ c - y) ** 2).sum())

        # --- anchored piecewise-constant dip: U(s) = u0 + sum_j dip_j * (arc in seg j)
        # column j = how much MD has been travelled inside segment j by row i
        for k in SEG:
            edges = np.linspace(0.0, L, k + 1)
            A = np.clip(dmd[:, None] - edges[None, :-1], 0.0,
                        np.diff(edges)[None, :])
            r = U - u0
            c, *_ = np.linalg.lstsq(A, r, rcond=None)
            sse[k] += float(((A @ c + u0 - z - y) ** 2).sum())
        n += len(ie)

    print(f"{len(lens)} wells, {n} eval rows, mean eval length "
          f"{np.mean(lens):.0f} ft (median {np.median(lens):.0f})\n")
    print("anchored on the well's own prefix; oracle chooses only the dips")
    print(f"{'rung':34s} {'pooled RMSE':>12s}")
    print("-" * 48)
    for name in ("carry_tvt", "flat_U"):
        print(f"{name:34s} {np.sqrt(sse[name]/n):12.4f}")
    print("-" * 48)
    for k in SEG:
        seg_ft = np.mean(lens) / k
        print(f"{('oracle %d dip segment%s' % (k, '' if k == 1 else 's')):34s} "
              f"{np.sqrt(sse[k]/n):12.4f}   (~{seg_ft:.0f} ft/segment)")
    print("-" * 48)
    for name in ("const_tvt_oracle", "lin_tvt_oracle", "quad_tvt_oracle"):
        print(f"{name:34s} {np.sqrt(sse[name]/n):12.4f}")
    print("-" * 48)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = {
        "n_wells": len(lens), "n_eval_rows": n,
        "mean_eval_length_ft": float(np.mean(lens)),
        "carry_last_tvt": float(np.sqrt(sse["carry_tvt"] / n)),
        "flat_U": float(np.sqrt(sse["flat_U"] / n)),
        "oracle_dip_segments": {str(k): float(np.sqrt(sse[k] / n)) for k in SEG},
        "oracle_tvt_const": float(np.sqrt(sse["const_tvt_oracle"] / n)),
        "oracle_tvt_line": float(np.sqrt(sse["lin_tvt_oracle"] / n)),
        "oracle_tvt_quadratic": float(np.sqrt(sse["quad_tvt_oracle"] / n)),
    }
    with open(os.path.join(RESULTS_DIR, "dip_ladder.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
