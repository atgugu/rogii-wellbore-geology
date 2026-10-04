"""Full-corpus data audit: verifies the mask semantics the whole harness
rests on, and records population statistics.

Run:  python scripts/01_audit_data.py
Out:  results/audit.json
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rogii.io import load_split, tw_fingerprint
from rogii.paths import DATA_DIR, RESULTS_DIR


t0 = time.time()
wells = load_split(DATA_DIR, "train")
print(f"loaded {len(wells)} wells in {time.time()-t0:.1f}s")

rows = []
violations = {"prefix_gap": [], "tvt_nan": [], "eval_empty": [], "md_step": [],
              "tw_not_monotonic": [], "tw_gr_nan": []}
for wid, w in wells.items():
    nn = ~np.isnan(w.tvt_input)
    idx = np.flatnonzero(nn)
    prefix_contig = (len(idx) == 0) or (idx[-1] - idx[0] + 1 == len(idx) and idx[0] == 0)
    if not prefix_contig:
        violations["prefix_gap"].append(wid)
    if w.tvt is None or np.isnan(w.tvt).any():
        violations["tvt_nan"].append(wid)
    if w.n_eval == 0:
        violations["eval_empty"].append(wid)
    step = np.diff(w.md)
    if not np.allclose(step, 1.0):
        violations["md_step"].append(wid)
    if not np.all(np.diff(w.tw_tvt) >= 0):
        violations["tw_not_monotonic"].append(wid)
    if np.isnan(w.tw_gr).any():
        violations["tw_gr_nan"].append(wid)

    m = w.eval_mask
    gr_eval = w.gr[m]
    tvt_eval = w.tvt[m]
    rows.append({
        "well": wid,
        "n_rows": len(w.md),
        "n_eval": int(m.sum()),
        "prefix_frac": round(1 - m.mean(), 4),
        "gr_nan_frac_eval": round(float(np.isnan(gr_eval).mean()), 4),
        "tvt_span_eval": round(float(tvt_eval.max() - tvt_eval.min()), 2),
        "tvt_out_of_tw_range": round(float(
            ((tvt_eval < w.tw_tvt[0]) | (tvt_eval > w.tw_tvt[-1])).mean()), 4),
        "tw_family": tw_fingerprint(w),
        "x0": float(w.x.mean()), "y0": float(w.y.mean()),
    })

n_eval_total = sum(r["n_eval"] for r in rows)
fams = {}
for r in rows:
    fams.setdefault(r["tw_family"], 0)
    fams[r["tw_family"]] += 1

audit = {
    "n_wells": len(wells),
    "n_eval_rows_total": n_eval_total,
    "violations": {k: v for k, v in violations.items()},
    "violation_counts": {k: len(v) for k, v in violations.items()},
    "prefix_frac_quartiles": list(np.round(np.percentile(
        [r["prefix_frac"] for r in rows], [0, 25, 50, 75, 100]), 4)),
    "n_eval_quartiles": list(np.round(np.percentile(
        [r["n_eval"] for r in rows], [0, 25, 50, 75, 100]), 1)),
    "gr_nan_eval_quartiles": list(np.round(np.percentile(
        [r["gr_nan_frac_eval"] for r in rows], [0, 25, 50, 75, 100]), 4)),
    "tvt_out_of_tw_range_mean": round(float(np.mean(
        [r["tvt_out_of_tw_range"] for r in rows])), 4),
    "n_typewell_families": len(fams),
    "largest_families": sorted(fams.values(), reverse=True)[:10],
}

os.makedirs(RESULTS_DIR, exist_ok=True)
with open(os.path.join(RESULTS_DIR, "audit.json"), "w") as f:
    json.dump({"summary": audit, "wells": rows}, f)
print(json.dumps(audit, indent=2))
