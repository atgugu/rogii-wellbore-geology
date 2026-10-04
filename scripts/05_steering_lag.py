"""Does the TRAJECTORY carry the dip, and does it LEAD or LAG the surface?

Physical claim: the well is geosteered. A geologist watches the formation and
tells the driller to turn, so the trajectory Z(md) is causally DOWNSTREAM of the
structural surface U(md) -- with a reaction delay. If so, Z at md+lag carries
information about U at md, and look-ahead over the future trajectory (which is
given in full at inference, and is host-settled legal) is a direct dip signal.

This matters because the other candidate dip sources are weak: a GR-only dynamic
program overfits (see scripts/07_gr_identifiability.py), the prefix's terminal dip
is noisier than the quantity it would correct, and neighbouring wells constrain the
dip only coarsely.

Note the trivial baselines are both special cases of one family:

    U_i = U_anchor + lam * (Z_i - Z_anchor)
    <=> TVT_i = TVT_anchor - (1 - lam) * (Z_i - Z_anchor)

lam = 0 is flat-U (107.5 pooled); lam = 1 is carry-last-TVT (15.9099). So
carry-TVT already asserts that the surface tracks the trajectory one-for-one.
The question is whether the optimal lam differs from 1, whether it varies per
well in a predictable way, and whether a LEAD improves it.

Run:  python scripts/05_steering_lag.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rogii.io import load_split
from rogii.paths import DATA_DIR, RESULTS_DIR

BASE_D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def smooth(x, k):
    if k < 3:
        return x.copy()
    c = np.convolve(x, np.ones(k) / k, "same")
    e = k // 2
    o = x.copy()
    if len(x) > 2 * e + 1:
        o[e:len(x) - e] = c[e:len(x) - e]
    return o


def main():
    wells = load_split(DATA_DIR, "train")
    ids = sorted(wells)

    # ---- 1. cross-correlation of dU and dZ vs LAG, inside the eval zone
    LAGS = [-1500, -1000, -600, -300, -150, 0, 150, 300, 600, 1000, 1500]
    K = 301                              # smoothing before differencing (ft)
    num = {L: 0.0 for L in LAGS}
    d2u = {L: 0.0 for L in LAGS}
    d2z = {L: 0.0 for L in LAGS}
    # ---- 2. optimal lambda in U = U_anchor + lam*(Z - Z_anchor)
    sxy = sxx = 0.0
    lam_w, wt = [], []
    sse = {}
    n = 0

    for wid in ids:
        w = wells[wid]
        ie = np.flatnonzero(w.eval_mask)
        ip = np.flatnonzero(~w.eval_mask)
        if len(ie) < 800 or len(ip) < 50:
            continue
        z, y = w.z[ie], w.tvt[ie]
        if not (np.isfinite(z).all() and np.isfinite(y).all()):
            continue
        a = ip[-1]
        z0, t0 = float(w.z[a]), float(w.tvt_input[a])
        U = y + z
        u0 = t0 + z0
        dz = z - z0
        du = U - u0

        # global and per-well lambda (no intercept: both are anchored)
        sxy += float(dz @ du)
        sxx += float(dz @ dz)
        if float(dz @ dz) > 1e-6:
            lam_w.append(float(dz @ du) / float(dz @ dz))
            wt.append(len(ie))

        for lam in (0.0, 0.5, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0):
            e = (u0 + lam * dz) - z - y
            sse[lam] = sse.get(lam, 0.0) + float(e @ e)
        n += len(ie)

        # lagged cross-correlation of the two rate signals
        gu = np.gradient(smooth(U, K), w.md[ie])
        gz = np.gradient(smooth(z, K), w.md[ie])
        gu -= gu.mean()
        gz -= gz.mean()
        for L in LAGS:
            s = int(round(L))            # MD is sampled at ~1 ft
            if s >= 0:
                A, B = gu[:len(gu) - s or None], gz[s:]
            else:
                A, B = gu[-s:], gz[:len(gz) + s or None]
            m = min(len(A), len(B))
            if m < 200:
                continue
            A, B = A[:m], B[:m]
            num[L] += float(A @ B)
            d2u[L] += float(A @ A)
            d2z[L] += float(B @ B)

    lam_hat = sxy / sxx
    lam_w = np.array(lam_w)
    wt = np.array(wt, dtype=float)
    print(f"{len(lam_w)} wells, {n} eval rows\n")
    print("=== U = U_anchor + lam*(Z - Z_anchor):  how tightly does the surface "
          "track the trajectory?")
    print(f"  pooled-optimal global lambda = {lam_hat:.4f}")
    print(f"  per-well lambda: mean {np.average(lam_w, weights=wt):.3f}  "
          f"median {np.median(lam_w):.3f}  sd {lam_w.std():.3f}  "
          f"p10 {np.percentile(lam_w,10):.3f}  p90 {np.percentile(lam_w,90):.3f}")
    print(f"\n  {'lam':>6s} {'pooled RMSE':>12s}")
    for lam in sorted(sse):
        tag = ("  <- flat-U" if lam == 0.0 else
               "  <- carry-last-TVT" if lam == 1.0 else "")
        print(f"  {lam:6.2f} {np.sqrt(sse[lam]/n):12.4f}{tag}")
    ora = 0.0
    # oracle per-well lambda, to bound what a perfect lambda predictor buys
    for wid, lw in zip([i for i in ids], []):
        pass
    print(f"\n  oracle per-well lambda would need a predictor of a quantity with "
          f"sd {lam_w.std():.3f} about a mean of {np.average(lam_w, weights=wt):.3f}")

    print("\n=== lagged correlation of dU/dMD with dZ/dMD "
          f"(smoothed {K} ft, inside the eval zone)")
    print("  positive lag = trajectory rate is read AHEAD of the surface rate,")
    print("  i.e. the FUTURE trajectory informs the CURRENT dip (steering delay)")
    print(f"\n  {'lag (ft)':>9s} {'corr':>8s}")
    best = (0.0, None)
    for L in LAGS:
        if d2u[L] > 0 and d2z[L] > 0:
            c = num[L] / np.sqrt(d2u[L] * d2z[L])
            if abs(c) > abs(best[0]):
                best = (c, L)
            print(f"  {L:9d} {c:8.4f}")
    print(f"\n  peak |corr| = {best[0]:.4f} at lag {best[1]} ft")


if __name__ == "__main__":
    main()
