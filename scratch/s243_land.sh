#!/usr/bin/env bash
# §243 local post-landing evals (DC_v1 §3; $0, the 4060 = cuda:0, fp32), per landed arm, serialised on the card by the shared lock:
#   every arm: eval_long_dc.json (K' 8,10,16,64,128; d 1..12; eval_bs 125, as scratch/s243_long.sh for the inits);
#   dc_s1 / dc_s2: REG refs <run>_reg (acc, ws, swap_kd, lp5h; eval_bs 100, as the §236 / §238 / §240 refs; model.pt hard link removed after).
# Writes the output files only. Unread (Theory scores with score_dc).   nohup bash scratch/s243_land.sh dc_s1 >> logs/s243_land.out 2>&1 &
cd "$(dirname "$0")/.." || exit 2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # allocator only (no numeric change)
L=scratch/_tmp/s240_land.lock
until mkdir $L 2>/dev/null; do sleep 30; done
trap 'rmdir $L' EXIT
for r in "$@"; do
  echo "$(date '+%F %T') start long $r"; python -B -m boxes.inject_aw_eval --run boxes/runs_fsq/$r --kprime_list 8,10,16,64,128 --long_ds 1,2,3,4,5,6,7,8,9,10,11,12 --long_name eval_long_dc.json --eval_bs 125 --device cuda:0; rc=$?
  echo "$(date '+%F %T') rc $rc long $r"; [ $rc = 0 ] || exit 3
  case $r in dc_s*)
    echo "$(date '+%F %T') start reg $r"; python -B scratch/s236_reg_eval.py --law 0 --run $r --parts acc,ws,swap_kd,lp5h --eval_bs 100 --device cuda:0; rc=$?
    echo "$(date '+%F %T') rc $rc reg $r"; [ $rc = 0 ] || exit 3
    rm -f boxes/runs_fsq/${r}_reg/model.pt;;
  esac
  echo "$(date '+%F %T') LAND DONE $r"
done
