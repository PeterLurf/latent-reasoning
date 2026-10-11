"""NORM-CAP cap-level sweep (Theory 2026-10-11, Literature's SANE prior 2608.22354): merge <run>/eval_normcap_x<f>.json (halt173 --normcap
--normcap_c <run>/eval_s185_normcap.json --normcap_factor f --kprimes 8,16,32) into <run>/eval_normcap_sweep.json: per factor, pooled n 3..8 at K' 8 / 16 / 32
and the n 9..12 mean at K' 16. Refuses unless all five factor files exist.
    python scratch/merge_normcap_sweep.py <run>"""
import json
import os
import sys

import numpy as np

FACTORS = (0.5, 0.75, 1.0, 1.25, 1.5)
EXTRA = (2.0, 3.0)                                                         # the wider sweep (2026-10-11, s184 monotone to 1.5x): merged when present
run = sys.argv[1]; FS = list(FACTORS) + [f for f in EXTRA if os.path.exists(os.path.join(run, "eval_normcap_x%s.json" % f))]; out = {"factors": FS, "by_factor": {}}
for f in FS:
    p = os.path.join(run, "eval_normcap_x%s.json" % f); assert os.path.exists(p), "missing " + p
    d = json.load(open(p)); nc = d["normcap"]; A = nc["acc_by_n_kprime"]
    out["by_factor"][str(f)] = {"C": nc["C"], "pooled_n3_8": {k: float(np.mean([A[k][str(n)] for n in range(3, 9)])) for k in ("8", "16", "32")},
                                "n9_12_mean_k16": float(np.mean([A["16"][str(n)] for n in range(9, 13)])), "acc_by_n_kprime": A}
out["C_base"] = out["by_factor"]["1.0"]["C"]; out["def"] = "halt173 --normcap with C = factor x the run's train-item C (eval_s185_normcap.json), K' {8,16,32}; pooled = mean acc over n 3..8"
json.dump(out, open(os.path.join(run, "eval_normcap_sweep.json"), "w", newline="\n"), indent=1)
print({f: (round(v["pooled_n3_8"]["8"], 3), round(v["pooled_n3_8"]["16"], 3), round(v["pooled_n3_8"]["32"], 3), round(v["n9_12_mean_k16"], 3)) for f, v in out["by_factor"].items()})
