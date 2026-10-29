#!/usr/bin/env bash
# §243 DEPTH-CURRIC local run_long (DC_v1 §3 fix 4; $0): eval_long_dc.json (K' 8,10,16,64,128; d 1..12; TEST2-DF via the run's law) on the given runs, the 4060
# (cuda:0), fp32, eval_bs 125 for every run (bs 250 OOMed on the 8 GB 4060 at d 12) (the inits lh_fh_s1 / s2 and, after landing, dc_s1 / dc_s2 / dc_ctrl_s1 / dc_ctrl_s2). Writes eval_long_dc.json only. Unread.
# Shares the local card lock.   nohup bash scratch/s243_long.sh lh_fh_s1 lh_fh_s2 >> logs/s243_long.out 2>&1 &
cd "$(dirname "$0")/.." || exit 2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # allocator only (no numeric change)
L=scratch/_tmp/s240_land.lock
until mkdir $L 2>/dev/null; do sleep 30; done
trap 'rmdir $L' EXIT
for r in "$@"; do
  echo "$(date '+%F %T') start long $r"; python -B -m boxes.inject_aw_eval --run boxes/runs_fsq/$r --kprime_list 8,10,16,64,128 --long_ds 1,2,3,4,5,6,7,8,9,10,11,12 --long_name eval_long_dc.json --eval_bs 125 --device cuda:0; rc=$?
  echo "$(date '+%F %T') rc $rc long $r"; [ $rc = 0 ] || exit 3
  echo "$(date '+%F %T') LONG DONE $r"
done
