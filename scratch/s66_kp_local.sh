#!/usr/bin/env bash
# THEORY §66 CL K' curve (boxes/PILOT.md, fixed 15:38): K' = ceil(N/2) and 2N on the kept CL checkpoints, local GPU,
# bf16 autocast as the panel's CL inference.  bash scratch/s66_kp_local.sh <device> [seeds...]   -> logs/s66_kp.log
cd "$(dirname "$0")/.."
dev=$1; shift; seeds=${*:-1 2 3 4 5}
for s in $seeds; do for m in half double; do
  f=bench/results/boxes__boxes_cl_s${s}_kp$m.json; [ -f $f ] && continue
  [ -f boxes/runs/cl_s$s/model.pt ] || { echo "$(date +%T) cl_s$s no model.pt" >> logs/s66_kp.log; continue; }
  python -u -m bench.run --task boxes --model "boxes:boxes/runs/cl_s$s?kp=$m" --tag boxes_cl_s${s}_kp$m --device $dev > logs/bench_cl_s${s}_kp$m.log 2>&1
  grep -q WROTE logs/bench_cl_s${s}_kp$m.log && echo "$(date +%T) cl_s$s kp=$m done" >> logs/s66_kp.log || echo "$(date +%T) cl_s$s kp=$m FAILED" >> logs/s66_kp.log
done; done
echo "$(date +%T) KP_DONE $seeds" >> logs/s66_kp.log
