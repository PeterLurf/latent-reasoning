#!/bin/bash
# N1-WPE-RESET (Theory 10-16 ~14:00, $0 descriptive): vbank_diag --part wpereset on N1 (s211_vb_carry_s1), 2070 fp32 eval_bs 32, then the intact-arm gate.
cd /d/Latent
R=boxes/runs_fsq/s211_vb_carry_s1
python -u -m boxes.vbank_diag --run $R --part wpereset --device cuda:2 --eval_bs 32 > logs/s212_wpereset.log 2>&1
echo "$(date '+%F %T') wpereset rc $? $(tail -n 1 logs/s212_wpereset.log)" >> logs/s211_stage0.out
python - >> logs/s212_wpereset.log 2>&1 <<'PY'
import json
R = "boxes/runs_fsq/s211_vb_carry_s1"; w = json.load(open(R + "/eval_wpereset.json")); v = json.load(open(R + "/eval_vb.json")); rs = json.load(open(R + "/eval_rs.json"))
g1 = w["intact"]["acc_n2"] == v["acc_n2"]; g2 = w["intact"]["acc_k24"] == v["acc_k"]["24"]
g3 = all(w["intact"]["rs_n9_13"][k][f] == rs["cells"][k][f] for k in w["intact"]["rs_n9_13"] for f in ("n_items", "intact", "restart", "exact_rate"))
print("GATE intact==eval_vb acc_n2", g1, "acc_k24", g2, "rs cells", g3)
PY
echo "$(date '+%F %T') wpereset $(grep GATE logs/s212_wpereset.log)" >> logs/s211_stage0.out
