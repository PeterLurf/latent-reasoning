#!/usr/bin/env bash
# §94 Step 0 probes (note 94-1; same settings as 2e559168) on the 13 §94 models; CPU, 3 threads. Launched at Idle
# priority (cmd start /LOW): bash scratch/s94_probes_new.sh. Outputs theory/results/s94_step0/<run>.json; skips done ones.
cd "$(dirname "$0")/.."
for r in boxes/runs/f94_stddec3x_s{6,7,8,9,10} boxes/runs/f94_stddecinv3x_s{6,7,8,9,10} boxes/runs/f94_fadestd3x_s{6,7,8}; do
  o=theory/results/s94_step0/$(basename $r).json
  [ -s $o ] && continue
  python -m boxes.probe94 --run $r --threads 3 --out $o > logs/s94_probe_$(basename $r).log 2>&1; rc=$?
  echo "$(date +%T) s94 probe $(basename $r) rc $rc $([ -s $o ] && echo ok || echo no-output)" >> logs/s94_probes.log
done
echo "$(date +%T) S94_PROBES_DONE" >> logs/s94_probes.log
