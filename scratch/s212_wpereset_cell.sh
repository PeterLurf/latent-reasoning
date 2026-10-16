#!/bin/bash
# wpereset on one §211 cell (descriptive, $0): $1 = cell dir name, $2 = device; optional $3 = pid to wait for (Windows pid via PowerShell). Card must equal the cell's eval_vb card for the intact gate.
cd /d/Latent
C=$1; D=$2; R=boxes/runs_fsq/$C; L=logs/s212_wpereset_$C.log
if [ -n "$3" ]; then while powershell -NoProfile -Command "if (Get-Process -Id $3 -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }"; do sleep 30; done; fi
python -u -m boxes.vbank_diag --run $R --part wpereset --device $D --eval_bs 32 > $L 2>&1
echo "$(date '+%F %T') wpereset $C rc $? $(tail -n 1 $L)" >> logs/s211_stage0.out
python - "$R" >> $L 2>&1 <<'PY'
import json, sys
R = sys.argv[1]; w = json.load(open(R + "/eval_wpereset.json")); v = json.load(open(R + "/eval_vb.json")); rs = json.load(open(R + "/eval_rs.json"))
g1 = w["intact"]["acc_n2"] == v["acc_n2"]; g2 = w["intact"]["acc_k24"] == v["acc_k"]["24"]
g3 = all(w["intact"]["rs_n9_13"][k][f] == rs["cells"][k][f] for k in w["intact"]["rs_n9_13"] for f in ("n_items", "intact", "restart", "exact_rate"))
print("GATE", R, "card", w["card"], "vs eval_vb", v["card"], "rs", rs["card"], "acc_n2", g1, "acc_k24", g2, "rs cells", g3)
PY
echo "$(date '+%F %T') wpereset $C $(grep GATE $L)" >> logs/s211_stage0.out
