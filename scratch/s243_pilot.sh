#!/usr/bin/env bash
# §243 DEPTH-CURRIC LP0 PILOT (DC_v1 §2 fix 6; $0 local): dc_s1 (--d_max 6) then dc_ctrl_s1 (--d_max 4) from lh_fh_s1, the REAL 8k schedule (hold_start 500),
# 1,000 steps each (--stop_at 1000 -> exit 9, no eval), micro 16 (micro 8 OOMed at d 6 K 10 on 8 GB), the 4060 (cuda:0), same card for both. Only the gn_plain windows and the per-d norm guard are
# extracted afterwards (scratch/s243_pilot_extract.py -> scratch/gate_s243/pilot.json); no accuracy field is read.
#   nohup bash scratch/s243_pilot.sh > logs/s243_pilot.out 2>&1 &
cd "$(dirname "$0")/.." || exit 2
I=boxes/runs_fsq/lh_fh_s1/model.pt; SH=$(sha256sum $I | cut -c1-16)
COM="--arm inject --think_k 10 --aux_step 1 --aux_fade 48000,48000 --aux_reweight --fast_add --cache_lower --steps 8000 --cue step --write_src answer --init_trunk boxes/runs_fsq/s233_askname_s1/model.pt --init_sha c00f7ec0bd7977fe --write_mix_start 0 --write_mix_end 0 --mul_s_p 2 --init_model $I --init_model_sha $SH --hold_inc 1.6"
for arm in dc ctrl; do
  [ $arm = dc ] && DM=6 || DM=4
  d=scratch/_tmp/s243/pilot_$arm; rm -rf $d; mkdir -p $d
  echo "$(date '+%F %T') start $arm (d_max $DM)"; python -B -m boxes.inject_aw $COM --d_max $DM --micro 16 --stop_at 1000 --ckpt_every 0 --log_every 100 --device cuda:0 --out $d > $d/stdout.log 2>&1; rc=$?
  echo "$(date '+%F %T') rc $rc $arm"
done
python -B scratch/s243_pilot_extract.py && echo "$(date '+%F %T') PILOT DONE"
