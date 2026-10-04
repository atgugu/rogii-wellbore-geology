"""Reproduce the community oracle ladder on the exact eval zones.

Reference (712037 / 726465, full 773): carry-last ~15.9 | constant ~9.04
| line ~6.70 | quadratic ~5.34 | smooth ~2.90-3.05.
Pass = pooled numbers land near these -> mask semantics + scorer verified.

Run:  python scripts/02_oracle_ladder.py  ->  results/oracle_ladder.json
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rogii.io import load_split
from rogii.paths import DATA_DIR, RESULTS_DIR
from rogii.metrics import score_predictions
from rogii.oracle import LADDER, oracle_predictions

os.makedirs(RESULTS_DIR, exist_ok=True)

wells = load_split(DATA_DIR, "train")

reference = {"carry_last": 15.91, "constant": 9.04, "line": 6.70,
             "quadratic": 5.34, "smooth": 3.0}
results = {}
for kind in LADDER:
    t0 = time.time()
    preds = {wid: oracle_predictions(w, kind) for wid, w in wells.items()}
    rep, wf = score_predictions(wells, preds)
    results[kind] = rep
    print(f"{kind:>10}: pooled {rep['pooled_rmse']:7.3f}  "
          f"(ref ~{reference[kind]:5.2f})  worst-dec {rep['worst_decile_pooled']:7.3f}  "
          f"[{time.time()-t0:.1f}s]")

with open(os.path.join(RESULTS_DIR, "oracle_ladder.json"), "w") as f:
    json.dump(results, f, indent=2)
