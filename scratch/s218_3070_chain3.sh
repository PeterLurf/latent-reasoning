#!/bin/bash
# §218 on the 3070 (cuda:1) after chain2 (MSYS pid 48534): G-L4b hook half, then G-L6 plain path.
#   G-L6: (a) vbank.main --mode slva constructs + 1 step (CPU, bs 2); (b) vbank_eval --parts acc on a scratch copy of s213_d3t_s1 (args.json + model.pt),
#   the 3070, eval_bs 32 (eval_vb's card / bs) -> acc_n2 / acc_k equal to the committed eval_vb.json exactly. The copy is deleted after.
cd /d/Latent
while kill -0 48534 2>/dev/null; do sleep 60; done
python -B scratch/gate_s218_lvb.py l4b_hook --device cuda:1 > scratch/gate_s218/g_l4b_hook.log 2>&1; rc=$?
echo "$(date '+%F %T') l4b_hook rc $rc" >> logs/s218_gates.out
O=scratch/_tmp/g_l6_step; rm -rf "${O:?}"; OMP_NUM_THREADS=2 python -B -m boxes.vbank --mode slva --steps 2 --bs 2 --device cpu --log_every 1 --ckpt_every 0 --stop_at 1 --out $O > scratch/gate_s218/g_l6_step.log 2>&1; rcS=$?
C=scratch/_tmp/g_l6_d3t; mkdir -p $C; cp boxes/runs_fsq/s213_d3t_s1/args.json boxes/runs_fsq/s213_d3t_s1/model.pt $C/
python -B -m boxes.vbank_eval --run $C --device cuda:1 --eval_bs 32 --parts acc > scratch/gate_s218/g_l6_acc.log 2>&1; rcA=$?
python - <<'PY'
import json
a = json.load(open("scratch/_tmp/g_l6_d3t/eval_vb.json")); b = json.load(open("boxes/runs_fsq/s213_d3t_s1/eval_vb.json"))
eq = {k: a.get(k) == b.get(k) for k in ("acc_n2", "acc_k")}
st = open("scratch/gate_s218/g_l6_step.log").read()
res = {"acc_equal": eq, "card": a.get("card"), "ref_card": b.get("card"), "eval_bs": a.get("eval_bs"), "step_ok": "STOPPED_AT 1" in st,
       "pass": all(eq.values()) and "STOPPED_AT 1" in st, "note": "vbank / slva untouched by §218 (git diff empty); D3-T acc re-run on eval_vb's card and bs"}
json.dump(res, open("scratch/gate_s218/g_l6.json", "w", newline="\n"), indent=1); print("g_l6", res)
PY
rm -f scratch/_tmp/g_l6_d3t/model.pt scratch/_tmp/g_l6_step/model.pt
echo "$(date '+%F %T') l6 rcStep $rcS rcAcc $rcA" >> logs/s218_gates.out
