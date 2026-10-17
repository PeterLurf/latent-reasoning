#!/usr/bin/env bash
# note 213-27 / 213-28 AGE-PROJ on D3-T ($0): vbank_diag part ageproj on DEV (default cuda:0 = the 4060), fp32, eval_bs 32 -> s213_d3t_s1/eval_ageproj.json;
# gate: intact within +-0.005 of eval_vb acc_n2 and acc_k['10'] at every n 1..14 (cross-card tolerance; else re-run on the 3070). Gate block written into the file.
cd "$(dirname "$0")/.."
DEV=${DEV:-cuda:0}; R=boxes/runs_fsq/s213_d3t_s1
python -m boxes.vbank_diag --run $R --part ageproj --device $DEV --eval_bs 32 > logs/s213_ageproj_s213_d3t_s1.log 2>&1; rc=$?
echo "$(date '+%F %T') ageproj rc $rc $(tail -n 1 logs/s213_ageproj_s213_d3t_s1.log | cut -c1-140)"
python - "$R" <<'PY'
import json, os, sys
R = sys.argv[1]; f = R + "/eval_ageproj.json"; d = json.load(open(f)); v = json.load(open(R + "/eval_vb.json")); i = d["arms"]["intact"]
m2 = max(abs(i["acc_n2"][n] - v["acc_n2"][n]) for n in i["acc_n2"]); m10 = max(abs(i["acc_k10"][n] - v["acc_k"]["10"][n]) for n in i["acc_k10"])
d["gate"] = {"max_abs_diff_n2": m2, "max_abs_diff_k10": m10, "card": d.get("card"), "ref_card": v.get("card"), "tol": 0.005, "pass": m2 <= 0.005 and m10 <= 0.005}
json.dump(d, open(f + ".tmp", "w", newline="\n"), indent=1); os.replace(f + ".tmp", f); print("GATE ageproj intact vs eval_vb:", d["gate"])
PY
