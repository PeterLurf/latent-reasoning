"""Note 230-1 (1): the dose-scale ratios from a §224 FH run's training log (descriptive): per 100-step window after lambda is set,
lambda * L_FP / answer_ce and lambda * L_FP / (answer_ce + rec_ce + far_ce) (window means, the fields fh_lambda / fh_lfp1 / answer_ce / rec_ce / far_ce),
summarised (first window, median, last window) + the first logged window's L_FP (steps 1..100; the dose-rule runs log no step-0 value).
    python -B scratch/s230_dose_ratios.py s224_fhft_s1 s224_fhft_s2 -> scratch/gate_s230/dose_ratios_s224.json"""
import json
import sys

import numpy as np

out = {}
for run in sys.argv[1:]:
    L = [json.loads(l) for l in open("boxes/runs_fsq/%s/log.jsonl" % run)]
    rows = [r for r in L if r.get("fh_lfp1") is not None and r.get("answer_ce")]
    lam_set = next((r for r in L if "fh_lambda_set" in r), None)
    on = [r for r in rows if r.get("fh_lambda")]
    a_ = [r["fh_lambda"] * r["fh_lfp1"] / r["answer_ce"] for r in on]
    b_ = [r["fh_lambda"] * r["fh_lfp1"] / (r["answer_ce"] + (r.get("rec_ce") or 0) + (r.get("far_ce") or 0)) for r in on]
    s = lambda v: {"first": v[0], "median": float(np.median(v)), "last": v[-1], "n_windows": len(v)} if v else None  # noqa: E731
    out[run] = {"lambda_set": lam_set, "first_window_lfp1": rows[0]["fh_lfp1"] if rows else None, "first_window_step": rows[0]["step"] if rows else None,
                "fp_over_ans": s(a_), "fp_over_rest": s(b_)}
json.dump(out, open("scratch/gate_s230/dose_ratios_s224.json", "w", newline="\n"), indent=1); print(json.dumps(out, indent=1)[:1500])
