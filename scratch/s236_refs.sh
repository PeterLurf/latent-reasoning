#!/usr/bin/env bash
# §236 REG references ($0, local 4060 = cuda:0, fp32, eval_bs 100; Audit G-D 10-27): s228_pdir2c_s1_reg (acc) + its batch-size check (acc at eval_bs 50 into
# boxes/runs_fsq/s228_pdir2c_s1_reg_bs50, per-item preds compared -> scratch/gate_s236/ref_bs_invariance.json), then aw_plaw4_s1_reg / aw_plaw4_s2_reg
# (ws, acc, cue_only, swap_kd). Unread: only file presence / item-identity counts are checked here.
#   nohup bash scratch/s236_refs.sh > logs/s236_refs.out 2>&1 &
cd "$(dirname "$0")/.." || exit 2
set -o pipefail
run() { echo "$(date '+%F %T') start $*"; python -B scratch/s236_reg_eval.py "$@"; rc=$?; echo "$(date '+%F %T') rc $rc ($*)"; return $rc; }
run --run s228_pdir2c_s1 --parts acc --eval_bs 100 --device cuda:0 || exit 3
run --run s228_pdir2c_s1 --parts acc --eval_bs 50 --device cuda:0 --out boxes/runs_fsq/s228_pdir2c_s1_reg_bs50 || exit 3
python -B - <<'EOF'
import json
a = json.load(open("boxes/runs_fsq/s228_pdir2c_s1_reg/eval_acc.json")); b = json.load(open("boxes/runs_fsq/s228_pdir2c_s1_reg_bs50/eval_acc.json")); o = {"by_d": {}}
for d in a["items"]:
    pa, pb = a["items"][d]["pred"], b["items"][d]["pred"]; sa, sb = a["items"][d]["sig"], b["items"][d]["sig"]
    o["by_d"][d] = {"n": len(pa), "same_items": sa == sb, "n_pred_differ": sum(x != y for x, y in zip(pa, pb))}
o["n_pred_differ_total"] = sum(v["n_pred_differ"] for v in o["by_d"].values()); o["all_same_items"] = all(v["same_items"] for v in o["by_d"].values())
o["eval_bs"] = [a["eval_bs"], b["eval_bs"]]; o["pass"] = o["all_same_items"] and o["n_pred_differ_total"] == 0
o["def"] = "Audit G-D 10-27: s228_pdir2c_s1_reg acc at eval_bs 100 vs 50 on the same TEST2-REG items (fp32, 4060): per-item preds equal, else the count of differing items"
json.dump(o, open("scratch/gate_s236/ref_bs_invariance.json", "w", newline="\n"), indent=1); print("BS-INVARIANCE pass", o["pass"], "differ", o["n_pred_differ_total"], flush=True)
EOF
run --run aw_plaw4_s1 --parts ws,acc,cue_only,swap_kd --eval_bs 100 --device cuda:0 || exit 3
run --run aw_plaw4_s2 --parts ws,acc,cue_only,swap_kd --eval_bs 100 --device cuda:0 || exit 3
echo "$(date '+%F %T') REFS DONE"
