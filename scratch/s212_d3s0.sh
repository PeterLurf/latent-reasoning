#!/bin/bash
# note 212-15 D3 Stage 0 rows (Audit OK 10-16 14:4x): $1 = card label, $2 = device, $3.. = "cell:part" jobs run in order on that card.
cd /d/Latent
D=$2; shift 2
for job in "$@"; do
  C=${job%%:*}; P=${job##*:}; R=boxes/runs_fsq/$C; L=logs/s212_${P}_$C.log
  python -u -m boxes.vbank_diag --run $R --part $P --device $D --eval_bs 32 > $L 2>&1
  echo "$(date '+%F %T') d3s0 $C $P rc $? $(tail -n 1 $L)" >> logs/s211_stage0.out
  python - "$R" "$P" >> $L 2>&1 <<'PY'
import json, sys
R, P = sys.argv[1], sys.argv[2]; v = json.load(open(R + "/eval_vb.json"))
if P == "stabhist":
    w = json.load(open(R + "/eval_stabhist.json"))["intact"]
    print("GATE", R, P, "acc_n2", all(w["acc_n2"][k] == v["acc_n2"][k] for k in w["acc_n2"]), "acc_k24", all(w["acc_k24"][k] == v["acc_k"]["24"][k] for k in w["acc_k24"]))
elif P == "d3sel":
    w = json.load(open(R + "/eval_d3sel.json")); rs = json.load(open(R + "/eval_rs.json"))["cells"]
    g1 = w["acc_n2"]["intact"] == v["acc_n2"]
    g2 = all(w["cells"][k][f] == rs[k][f] for k in w["cells"] for f in ("n_items", "intact", "restart"))
    g3 = all(w["gapped_control"][k][f] == rs[k][f] for k in w["gapped_control"] for f in ("n_items", "restart"))
    print("GATE", R, P, "acc_n2", g1, "rs cells", g2, "ctrl restart", g3)
else:
    print("GATE", R, P, "n/a (no intact reference)")
PY
  echo "$(date '+%F %T') d3s0 $C $P $(grep GATE $L | tail -n 1)" >> logs/s211_stage0.out
done
echo "$(date '+%F %T') d3s0 card $D done" >> logs/s211_stage0.out
