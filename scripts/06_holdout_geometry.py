"""Will the hidden wells have train neighbours?

Cross-well features (interpolating the marker surface or its dip from nearby
training wells) behave very differently under a random split of wells and under
a spatial hold-out. Which regime applies turns on ONE question: did the
organisers split wells at random, or hold out a spatial region / whole pads?

Three tests, none of which need the hidden data:

1. PADS. Horizontal wells are drilled in groups from a shared surface pad, so
   their heels cluster tightly while their laterals fan out. If the split is
   random by well, a hidden well's pad-mates are overwhelmingly likely to be in
   train, which guarantees a neighbour a few hundred feet away. Measure the pad
   structure: how many pads, how many wells each, and how close pad-mates run.

2. VOIDS. If a whole region were held out, the train wells would show an
   anomalous hole -- a contiguous empty area surrounded by dense drilling.
   Occupancy of a grid over the drilled footprint tests this.

3. SIMULATED HOLDOUT. Drop a random 20% of wells (the hidden-set fraction) and
   measure the nearest-remaining-well distance for the dropped ones. Compare
   against the same statistic under a pad-wise and a block-wise drop. This says
   directly what a cross-well feature would see under each hypothesis.

Run:  python scripts/06_holdout_geometry.py
"""
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rogii.io import load_split
from rogii.paths import DATA_DIR, RESULTS_DIR

BASE_D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUB = 20


def main():
    wells = load_split(DATA_DIR, "train")
    ids = sorted(wells)
    X = {i: wells[i].x[::SUB] for i in ids}
    Y = {i: wells[i].y[::SUB] for i in ids}
    heel = np.array([[wells[i].x[0], wells[i].y[0]] for i in ids])
    print(f"{len(ids)} wells\n")

    # ---- 1. pads: cluster the heels
    print("=== 1. PAD STRUCTURE (clustering well heels) ===")
    for r in (200.0, 500.0, 1000.0):
        t = cKDTree(heel)
        seen, pads = set(), []
        for j in range(len(ids)):
            if j in seen:
                continue
            grp = set(t.query_ball_point(heel[j], r))
            frontier = grp - seen
            while frontier:                       # single-linkage
                nxt = set()
                for q in frontier:
                    nxt |= set(t.query_ball_point(heel[q], r))
                frontier = nxt - grp
                grp |= nxt
            grp -= seen
            if grp:
                seen |= grp
                pads.append(sorted(grp))
        sz = np.array([len(p) for p in pads])
        share = (sz[sz >= 2].sum() / len(ids)) if len(sz) else 0.0
        print(f"  link radius {r:6.0f} ft -> {len(pads):4d} pads | "
              f"median size {np.median(sz):.0f} mean {sz.mean():.2f} max {sz.max()} | "
              f"{share:.1%} of wells share a pad with >=1 other")

    # ---- 2. voids over the drilled footprint
    print("\n=== 2. SPATIAL VOIDS (occupancy over the drilled footprint) ===")
    ax = np.concatenate([X[i] for i in ids])
    ay = np.concatenate([Y[i] for i in ids])
    for cell in (2000.0, 4000.0):
        gx = ((ax - ax.min()) // cell).astype(int)
        gy = ((ay - ay.min()) // cell).astype(int)
        H = np.zeros((gy.max() + 1, gx.max() + 1), dtype=int)
        np.add.at(H, (gy, gx), 1)
        occ = H > 0
        # interior empties = empty cells with >=6 of 8 neighbours occupied
        pad = np.pad(occ, 1)
        nb = sum(pad[1 + dy:1 + dy + occ.shape[0], 1 + dx:1 + dx + occ.shape[1]]
                 for dy in (-1, 0, 1) for dx in (-1, 0, 1)) - occ
        holes = int(((~occ) & (nb >= 6)).sum())
        print(f"  cell {cell:5.0f} ft: grid {occ.shape}  occupied {occ.sum()} "
              f"({occ.mean():.1%})  enclosed empty cells {holes}")

    # ---- 3. simulated holdouts
    print("\n=== 3. SIMULATED 20% HOLDOUT: distance from a dropped well to the "
          "nearest REMAINING well ===")
    rng = np.random.default_rng(0)
    n_hold = int(round(0.20 * len(ids)))

    def report(name, hold_idx):
        keep = [ids[j] for j in range(len(ids)) if j not in set(hold_idx)]
        px = np.concatenate([X[i] for i in keep])
        py = np.concatenate([Y[i] for i in keep])
        tree = cKDTree(np.column_stack([px, py]))
        med = []
        for j in hold_idx:
            i = ids[j]
            d, _ = tree.query(np.column_stack([X[i], Y[i]]), k=1, workers=-1)
            med.append(float(np.median(d)))
        med = np.array(med)
        print(f"  {name:28s} n={len(med):3d}  median {np.median(med):7.0f} ft  "
              f"p90 {np.percentile(med,90):8.0f}  frac<1500ft {np.mean(med<1500):.1%}")
        return med

    report("random by well", rng.choice(len(ids), n_hold, replace=False))
    # pad-wise: drop whole heel-clusters until 20% held out
    t = cKDTree(heel)
    seen, pads = set(), []
    for j in range(len(ids)):
        if j in seen:
            continue
        grp = set(t.query_ball_point(heel[j], 500.0)) - seen
        if grp:
            seen |= grp
            pads.append(sorted(grp))
    order = rng.permutation(len(pads))
    sel = []
    for p in order:
        if len(sel) >= n_hold:
            break
        sel += pads[p]
    report("whole pads", sel[:n_hold + 8])
    # block-wise: one spatial K-means block
    from scipy.cluster.vq import kmeans2
    C = np.array([[X[i].mean(), Y[i].mean()] for i in ids])
    _, lab = kmeans2((C - C.mean(0)) / C.std(0), 5, minit="++", seed=0)
    b = np.bincount(lab).argmax()
    report("one spatial block", [j for j in range(len(ids)) if lab[j] == b])


if __name__ == "__main__":
    main()
