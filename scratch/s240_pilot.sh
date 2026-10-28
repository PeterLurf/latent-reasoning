#!/usr/bin/env bash
# §240 norm-guard PILOT (Audit G-D 184f60c7: "pilot first"; $0 local): CTRL (--hold_log) then FH (--hold_inc 1.6) from df_plaw4_s1, the REAL 8k schedule (no sched_scale;
# penalty from step 500), 1,000 steps each (--stop_at 1000 -> exit 9, no eval), micro 8, the 4060 (cuda:0), same card for both. Only the norm / ratio / grad-norm fields
# are extracted afterwards (scratch/s240_pilot_extract.py -> scratch/gate_s240/pilot.json); no accuracy field is read.
#   nohup bash scratch/s240_pilot.sh > logs/s240_pilot.out 2>&1 &
cd "$(dirname "$0")/.." || exit 2
I=boxes/runs_fsq/df_plaw4_s1/model.pt; SH=$(sha256sum $I | cut -c1-16)
B=$(grep "^boxes/runs_fsq/lh_ctrl_s1 " scratch/gen_q_b360.txt | cut -d' ' -f2- | sed 's/--args_ref [^ ]*//; s/--auto_micro//; s/--eval_parts [^ ]*//; s/--ckpt_every [0-9]*//; s/--hold_log//')
for arm in ctrl fh; do
  [ $arm = ctrl ] && EX="--hold_log" || EX="--hold_inc 1.6"
  d=scratch/_tmp/s240/pilot_$arm; rm -rf $d; mkdir -p $d
  echo "$(date '+%F %T') start $arm"; python -B -m $B $EX --micro 8 --stop_at 1000 --ckpt_every 0 --log_every 100 --device cuda:0 --out $d > $d/stdout.log 2>&1; rc=$?
  echo "$(date '+%F %T') rc $rc $arm"
done
python -B scratch/s240_pilot_extract.py && echo "$(date '+%F %T') PILOT DONE"
