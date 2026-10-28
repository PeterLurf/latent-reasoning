#!/usr/bin/env bash
# §240 step-0 refs (LH_v1 §3; $0, local 4060, fp32): eval_long_k128.json (K' 8,16,32,64,128; d 1-5; TEST2-DF via the run's law) on df_plaw4_s1 / df_plaw4_s2,
# one card and eval_bs 250 for every K' (239-1 (2)). Writes <run>/eval_long_k128.json only. Unread.
#   nohup bash scratch/s240_long0.sh > logs/s240_long0.out 2>&1 &
cd "$(dirname "$0")/.." || exit 2
for r in df_plaw4_s1 df_plaw4_s2; do
  echo "$(date '+%F %T') start $r"; python -B -m boxes.inject_aw_eval --run boxes/runs_fsq/$r --kprime_list 8,16,32,64,128 --long_ds 1,2,3,4,5 --long_name eval_long_k128.json --eval_bs 250 --device cuda:0; rc=$?
  echo "$(date '+%F %T') rc $rc $r"; [ $rc = 0 ] || exit 3
done
echo "$(date '+%F %T') LONG0 DONE"
