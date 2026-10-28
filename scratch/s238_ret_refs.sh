#!/usr/bin/env bash
# §238 retention refs (Note 238-1 (2) / 238-4; $0, local 4060 = cuda:0, fp32, eval_bs 100): the DF-trained LAT models evaluated on §237's REG sets (acc, ws). Unread.
#   nohup bash scratch/s238_ret_refs.sh > logs/s238_ret_refs.out 2>&1 &
cd "$(dirname "$0")/.." || exit 2
run() { echo "$(date '+%F %T') start $*"; python -B scratch/s236_reg_eval.py --law 0 "$@"; rc=$?; echo "$(date '+%F %T') rc $rc ($*)"; return $rc; }
run --run df_plaw4_s1 --parts acc,ws --eval_bs 100 --device cuda:0 || exit 3
run --run df_plaw4_s2 --parts acc,ws --eval_bs 100 --device cuda:0 || exit 3
for d in df_plaw4_s1_reg df_plaw4_s2_reg; do rm -f boxes/runs_fsq/$d/model.pt; done
echo "$(date '+%F %T') REFS DONE"
