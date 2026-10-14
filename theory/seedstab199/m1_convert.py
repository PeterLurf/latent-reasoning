"""§199 M1 input conversion (committed so the step is visible): the CPU subagent's EMP_alpha_pass.json table [model][n] ->
theory/data/s199_m1_alpha.json {model: {n: {alpha_pass, p_mean_logsv}}} for theory/score_s199_m1.py. Copies values only; no filtering.
    python theory/seedstab199/m1_convert.py <path to EMP_alpha_pass.json>"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
t = json.load(open(sys.argv[1]))["table"]
out = {m: {n: {"alpha_pass": v["alpha_pass"], "p_mean_logsv": v.get("mean_log_sv_pass")} for n, v in t[m].items()} for m in t}
json.dump(out, open(os.path.join(ROOT, "theory", "data", "s199_m1_alpha.json"), "w", newline="\n"), indent=1)
print("wrote", len(out), "models")
