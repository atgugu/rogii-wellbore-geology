#!/usr/bin/env python3
"""Is gamma ray informative at LOW model complexity, even if it is not at high?

An earlier experiment (a dynamic program over the dip path, not included here)
fitted the GR log better than ground truth on most wells: with enough
flexibility the optimiser finds paths whose GR misfit is lower than the truth's.
That shows the likelihood can be overfitted. It does not show GR is uninformative
when the path is constrained to a few dip segments -- and a few dip segments is
where the oracle ladder (scripts/03_dip_ladder.py) says the value is.

THE TEST (no DP, no fitting, so it cannot overfit):
  * take the eval zone of each well
  * build the K-segment least-squares approximation of the TRUE dU/dMD -- the
    best any K-segment method could do, i.e. the oracle at that complexity
  * build M impostors of the SAME complexity by resampling the segment slopes
    from the fleet's slope distribution, anchored identically
  * score every candidate by the same GR likelihood (heel-calibrated typewell
    match), and record where the truth ranks

If the truth's rank is uniform, GR carries no information at that complexity and
GR is not worth pursuing at that complexity.  If the truth ranks near the top,
a total-variation prior on the dip plus GR is a candidate dip estimator.

Run:  python scripts/07_gr_identifiability.py --n 120
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rogii.io import load_split
from rogii.paths import DATA_DIR, RESULTS_DIR

BASE_D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def seg_fit(y, md, K, rng=None, slopes=None):
    """Piecewise-linear fit of y(md) with K segments, breakpoints on a grid.
    If `slopes` is given, use those slopes instead of the least-squares ones
    (an impostor of the same complexity, anchored at the same start)."""
    n = len(y)
    if K <= 1:
        cuts = [0, n]
    else:
        cuts = [int(round(t)) for t in np.linspace(0, n, K + 1)]
    out = np.empty(n)
    val = y[0]
    used = []
    for s in range(len(cuts) - 1):
        a, b = cuts[s], cuts[s + 1]
        if b <= a:
            continue
        dm = md[a:b] - md[a]
        if slopes is None:
            # least squares slope through the segment, continuous at the joint
            g = float(np.dot(dm - dm.mean(), y[a:b] - y[a:b].mean()) /
                      max(np.dot(dm - dm.mean(), dm - dm.mean()), 1e-9))
        else:
            g = float(slopes[s])
        used.append(g)
        out[a:b] = val + g * dm
        val = out[b - 1]
    return out, used


def gr_cost(tvt, gr, tw_tvt, tw_gr, a, b, s, cover_pen=True):
    """Heel-calibrated typewell GR match, Cauchy loss (the DP's emission).

    `s` MUST be fixed per well and shared by every candidate. Re-estimating it
    per candidate makes the loss scale-invariant, so a candidate with wildly
    larger residuals gets a proportionally larger s and a LOWER cost -- which
    ranks bad paths above good ones. That bug made the truth look worse than
    random on the first run of this script.

    Candidates that leave the typewell's TVT range are also penalised at the
    fleet-median cost rather than silently scoring only their covered rows,
    which would otherwise reward paths that exit the typewell entirely."""
    pred = np.interp(tvt, tw_tvt, tw_gr, left=np.nan, right=np.nan)
    r = (gr - (a * pred + b))
    ok = np.isfinite(r)
    if ok.sum() < 50:
        return np.inf
    c = np.log1p((r[ok] / (2.0 * s)) ** 2)
    if cover_pen:
        # uncovered rows cost the 90th percentile of the covered ones
        miss = len(r) - ok.sum()
        if miss:
            c = np.concatenate([c, np.full(miss, np.quantile(c, 0.90))])
    return float(np.mean(c))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--imp", type=int, default=200)
    args = ap.parse_args()

    wells = load_split(DATA_DIR, "train")
    ids = sorted(wells)[: args.n]
    rng = np.random.default_rng(0)

    # fleet slope distribution, from the true dU/dMD of the sampled wells
    pool = []
    for wid in ids:
        w = wells[wid]
        m = w.eval_mask
        if m.sum() < 500 or w.tvt is None:
            continue
        u = w.tvt[m] + w.z[m]
        md = w.md[m]
        pool.append(np.diff(u) / np.maximum(np.diff(md), 1e-6))
    pool = np.concatenate(pool)
    pool = pool[np.isfinite(pool)]
    lo, hi = np.quantile(pool, [0.02, 0.98])
    pool = pool[(pool > lo) & (pool < hi)]
    print(f"fleet dU/dMD: sd {pool.std():.5f}  q02 {lo:.5f}  q98 {hi:.5f}", flush=True)

    for K in (1, 2, 4):
        ranks, n_ok = [], 0
        for wid in ids:
            w = wells[wid]
            m = w.eval_mask
            if m.sum() < 500 or w.tvt is None or w.tw_tvt is None:
                continue
            md, z, gr = w.md[m], w.z[m], w.gr[m]
            u_true = w.tvt[m] + z
            if not np.isfinite(gr).any():
                continue
            # heel calibration of (a, b) on the visible prefix
            p = ~m
            if p.sum() < 100:
                continue
            tvp = w.tvt_input[p]
            grp = w.gr[p]
            twp = np.interp(tvp, w.tw_tvt, w.tw_gr, left=np.nan, right=np.nan)
            ok = np.isfinite(twp) & np.isfinite(grp)
            if ok.sum() < 80:
                continue
            A = np.stack([twp[ok], np.ones(ok.sum())], 1)
            ab, *_ = np.linalg.lstsq(A, grp[ok], rcond=None)
            a, b = float(ab[0]), float(ab[1])
            # FIXED noise scale, from the PREFIX residuals of the calibration --
            # legal at inference and identical for every candidate.
            rp = grp[ok] - (a * twp[ok] + b)
            sig = float(np.median(np.abs(rp)) * 1.4826) + 1e-6

            u_fit, slopes = seg_fit(u_true, md, K)
            c_true = gr_cost(u_fit - z, gr, w.tw_tvt, w.tw_gr, a, b, sig)
            if not np.isfinite(c_true):
                continue
            worse = 0
            for _ in range(args.imp):
                s = rng.choice(pool, size=max(K, 1), replace=True)
                u_imp, _ = seg_fit(u_true, md, K, slopes=s)
                c = gr_cost(u_imp - z, gr, w.tw_tvt, w.tw_gr, a, b, sig)
                if np.isfinite(c) and c > c_true:
                    worse += 1
            ranks.append(worse / args.imp)
            n_ok += 1
        r = np.array(ranks)
        print(f"K={K}: {n_ok} wells | truth beats {r.mean()*100:5.1f}% of same-complexity "
              f"impostors (50% = no information) | top-decile on {100*(r>0.9).mean():4.1f}% "
              f"of wells | median {np.median(r)*100:5.1f}%", flush=True)


if __name__ == "__main__":
    main()
