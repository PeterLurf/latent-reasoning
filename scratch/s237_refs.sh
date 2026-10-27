#!/usr/bin/env bash
# §237 step-0 REG references (Note 237-1 (1); $0, local 4060 = cuda:0, fp32, eval_bs 100): the init models evaluated on §236's REG sets before any §237 training.
# aw_plaw2_s1_reg at d_max 4 (trained at 2; the P-4 parts need d 3-4): acc, ws, lp5h, swap_kd, cue_only, lp5, cue_wrong; aw_pcd2_s1_reg: acc. Unread.
#   nohup bash scratch/s237_refs.sh > logs/s237_refs.out 2>&1 &
cd "$(dirname "$0")/.." || exit 2
run() { echo "$(date '+%F %T') start $*"; python -B scratch/s236_reg_eval.py "$@"; rc=$?; echo "$(date '+%F %T') rc $rc ($*)"; return $rc; }
run --run aw_plaw2_s1 --parts acc,ws,lp5h,swap_kd,cue_only,lp5,cue_wrong --eval_bs 100 --set_d_max 4 --device cuda:0 || exit 3
run --run aw_pcd2_s1 --parts acc --eval_bs 100 --device cuda:0 || exit 3
for d in aw_plaw2_s1_reg aw_pcd2_s1_reg; do rm -f boxes/runs_fsq/$d/model.pt; done
echo "$(date '+%F %T') REFS DONE"
