"""Regenerate the figures in figures/ from results/ and one example well.

Run:  python scripts/08_make_figures.py
"""
import csv
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rogii.io import load_split
from rogii.paths import DATA_DIR, RESULTS_DIR, ROOT

FIG = os.path.join(ROOT, "figures")
INK, MUTED, GRID = "#1f2328", "#6b7280", "#e5e7eb"
BLUE, ORANGE = "#2a78d6", "#eb6834"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "figure.dpi": 100, "savefig.dpi": 200,
    "savefig.facecolor": "white", "figure.facecolor": "white",
})


def fig_dip_ladder():
    d = json.load(open(os.path.join(RESULTS_DIR, "dip_ladder.json")))
    k = [int(x) for x in d["oracle_dip_segments"]]
    v = [d["oracle_dip_segments"][str(x)] for x in k]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(k, v, "-o", color=BLUE, lw=2, ms=5, zorder=3)
    refs = [("carry the last known TVT", d["carry_last_tvt"]),
            ("best constant TVT (oracle)", d["oracle_tvt_const"]),
            ("best line in TVT (oracle)", d["oracle_tvt_line"]),
            ("best quadratic in TVT (oracle)", d["oracle_tvt_quadratic"])]
    for name, y in refs:
        ax.axhline(y, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
        below = name.startswith("best quadratic")
        ax.text(66, y * (0.97 if below else 1.03), f"{name}  {y:.2f}", ha="right",
                va="top" if below else "bottom", fontsize=8.5, color=MUTED)
    for x, y in ((1, v[0]), (2, v[1]), (4, v[3]), (16, v[7])):
        ax.annotate(f"{y:.2f}", (x, y), (x * 1.06, y * 0.86), ha="left", va="top",
                    fontsize=9, color=INK)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(k); ax.set_xticklabels([str(x) for x in k])
    ax.set_yticks([0.2, 0.5, 1, 2, 5, 10, 20])
    ax.set_yticklabels(["0.2", "0.5", "1", "2", "5", "10", "20"])
    ax.minorticks_off()
    ax.set_xlabel("piecewise-constant dip segments per well (oracle-chosen)")
    ax.set_ylabel("pooled RMSE, ft (log scale)")
    ax.set_title("How much of the task is a handful of dip changes?",
                 loc="left", fontsize=12, color=INK, fontweight="bold", pad=24)
    ax.text(0, 1.03, f"773 train wells, {d['n_eval_rows']:,} eval rows; "
            "level pinned to each well's own prefix", transform=ax.transAxes,
            fontsize=8.5, color=MUTED, va="bottom")
    ax.set_ylim(0.12, 24)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "dip_ladder.png"))
    plt.close(fig)


def example_well(wells):
    """First well (sorted by id) of typical length whose carry-last RMSE sits
    near the fleet median, so the picture is representative, not a showpiece."""
    for wid in sorted(wells):
        w = wells[wid]
        m = w.eval_mask
        n = int(m.sum())
        if 4700 <= n <= 5000 and np.isfinite(w.formations["EGFDU"]).all():
            a = np.flatnonzero(~m)[-1]
            r = np.sqrt(np.mean((w.tvt_input[a] - w.tvt[m]) ** 2))
            if 10.0 <= r <= 11.5:
                return w
    raise RuntimeError("no example well")


def fig_identity():
    wells = load_split(DATA_DIR, "train")
    w = example_well(wells)
    m = w.eval_mask
    ip, ie = np.flatnonzero(~m), np.flatnonzero(m)
    a = ip[-1]
    lo = max(0, a - 400)           # show the last 400 ft of prefix onward
    md = w.md / 1000.0
    u = w.tvt + w.z
    marker = w.formations["EGFDU"]
    # oracle k-segment dips, anchored on the last known row (as in scripts/03)
    dmd = w.md[ie] - w.md[a]
    L = dmd[-1]
    u0 = w.tvt_input[a] + w.z[a]

    def oracle(k):
        edges = np.linspace(0, L, k + 1)
        A = np.clip(dmd[:, None] - edges[None, :-1], 0, np.diff(edges)[None, :])
        c, *_ = np.linalg.lstsq(A, u[ie] - u0, rcond=None)
        return A @ c + u0 - w.z[ie]

    fig, axs = plt.subplots(2, 1, figsize=(7.2, 5.6), sharex=True,
                            gridspec_kw={"height_ratios": [1, 1.25], "hspace": 0.12})
    ax = axs[0]
    ax.plot(md[lo:], marker[lo:], color=ORANGE, lw=1.8, label="EGFDU marker beneath the well path")
    ax.plot(md[lo:], w.z[lo:], color=INK, lw=1.4, label="well trajectory, Z")
    ax.set_ylabel("elevation, ft")
    ax.legend(frameon=False, loc="best", fontsize=8.5, labelcolor=INK)
    ax.set_title("One lateral: the trajectory wiggles, the target is the surface",
                 loc="left", fontsize=12, color=INK, fontweight="bold", pad=24)
    ax.text(0, 1.03, "TVT + Z − marker is constant along the well "
            f"(σ = {np.std(u - marker):.4f} ft here; ≤ 0.009 ft on all 773 wells)",
            transform=ax.transAxes, fontsize=8.5, color=MUTED, va="bottom")
    ax = axs[1]
    ax.plot(md[lo:a + 1], w.tvt[lo:a + 1], color=INK, lw=1.6, label="TVT, known prefix")
    ax.plot(md[ie], w.tvt[ie], color=INK, lw=1.0, alpha=0.55, label="TVT, to be predicted")
    ax.plot(md[ie], np.full(len(ie), w.tvt[a]), color=MUTED, lw=1.2, ls=(0, (4, 3)),
            label="carry the last TVT")
    ax.plot(md[ie], oracle(2) , color=BLUE, lw=2, label="oracle, 2 dip segments")
    ye = w.tvt[ie]
    ax.set_ylim(min(ye.min(), w.tvt[a]) - 6, max(ye.max(), w.tvt[a]) + 6)
    ax.axvline(md[a], color=MUTED, lw=0.8)
    ax.text(md[a], ax.get_ylim()[0], " prediction starts", fontsize=8.5,
            color=MUTED, va="bottom")
    ax.set_xlabel("measured depth, 1000 ft"); ax.set_ylabel("TVT, ft")
    ax.legend(frameon=False, loc="upper left", fontsize=8.5, labelcolor=INK, ncol=2)
    fig.subplots_adjust(left=0.12, right=0.97, top=0.92, bottom=0.09)
    fig.savefig(os.path.join(FIG, "identity_example.png"))
    plt.close(fig)


def fig_reversal():
    rows = list(csv.DictReader(open(os.path.join(RESULTS_DIR, "board_vs_offline.csv"))))
    labels = [f"{r['operator']}\n{r['setting']}" for r in rows]
    off = [float(r["offline_delta_rmse"]) for r in rows]
    brd = [float(r["board_delta_rmse"]) for r in rows]
    y = np.arange(len(rows))[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    for yi, o, b in zip(y, off, brd):
        ax.plot([o, b], [yi, yi], color=GRID, lw=3, zorder=1, solid_capstyle="round")
    ax.scatter(off, y, s=60, color=BLUE, zorder=3, label="offline estimate (cross-validated)")
    ax.scatter(brd, y, s=60, color=ORANGE, zorder=3, label="measured on the leaderboard")
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("change in pooled RMSE, ft   (← better · worse →)")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(0.0, 0.42), fontsize=8.5, labelcolor=INK)
    ax.set_title("Operators fitted offline changed sign on the board",
                 loc="left", fontsize=12, color=INK, fontweight="bold", pad=24)
    ax.text(0, 1.03, "Board change is against an unchanged re-run of the same pipeline",
            transform=ax.transAxes, fontsize=8.5, color=MUTED, va="bottom")
    ax.set_xlim(-0.55, 0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "offline_vs_board.png"))
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    fig_dip_ladder()
    fig_identity()
    fig_reversal()
