#!/usr/bin/env bash
# §239 LONG-RUN evals (LR_v1 §2; Audit G-D 27b32515): one --kprime_list 8,16,32,64 call per model, d 2-5, eval_bs 250 at every K', fp32, the 4060 (cuda:0).
# df_plaw4_s1 / s2 (law 2 -> TEST2-DF), rcb_plaw4_s1 / s2 (law 0 -> TEST2-REG); writes <run>/eval_long.json only. Unread.
#   nohup bash scratch/s239_run.sh > logs/s239_run.out 2>&1 &
cd "$(dirname "$0")/.." || exit 2
for r in df_plaw4_s1 df_plaw4_s2 rcb_plaw4_s1 rcb_plaw4_s2; do
  echo "$(date '+%F %T') start $r"; python -B -m boxes.inject_aw_eval --run boxes/runs_fsq/$r --kprime_list 8,16,32,64 --long_ds 2,3,4,5 --eval_bs 250 --device cuda:0; rc=$?
  echo "$(date '+%F %T') rc $rc $r"; [ $rc = 0 ] || exit 3
done
echo "$(date '+%F %T') LONG DONE"
