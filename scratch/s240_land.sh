#!/usr/bin/env bash
# §240 local post-landing evals (LH_v1 §3; $0, local 4060 = cuda:0, fp32), per landed arm, serialised on the card by a mkdir lock:
#   every arm: eval_long_k128.json (K' 8,16,32,64,128; d 1-5; TEST2-DF via the run's law; eval_bs 250, as the step-0 refs s240_long0.sh);
#   lh_fh_*  : REG refs <run>_reg (acc, ws, swap_kd, lp5h; eval_bs 100, as the §236/§238 refs; model.pt hard link removed after).
# Writes the output files only. Unread (Theory scores with score_lh).
#   nohup bash scratch/s240_land.sh lh_fh_s1 >> logs/s240_land.out 2>&1 &
cd "$(dirname "$0")/.." || exit 2
L=scratch/_tmp/s240_land.lock
until mkdir $L 2>/dev/null; do sleep 30; done
trap 'rmdir $L' EXIT
for r in "$@"; do
  echo "$(date '+%F %T') start long $r"; python -B -m boxes.inject_aw_eval --run boxes/runs_fsq/$r --kprime_list 8,16,32,64,128 --long_ds 1,2,3,4,5 --long_name eval_long_k128.json --eval_bs 250 --device cuda:0; rc=$?
  echo "$(date '+%F %T') rc $rc long $r"; [ $rc = 0 ] || exit 3
  case $r in lh_fh_*)
    echo "$(date '+%F %T') start reg $r"; python -B scratch/s236_reg_eval.py --law 0 --run $r --parts acc,ws,swap_kd,lp5h --eval_bs 100 --device cuda:0; rc=$?
    echo "$(date '+%F %T') rc $rc reg $r"; [ $rc = 0 ] || exit 3
    rm -f boxes/runs_fsq/${r}_reg/model.pt;;
  esac
  echo "$(date '+%F %T') LAND DONE $r"
done
