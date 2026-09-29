#!/usr/bin/env bash
# Local progress dashboard, refreshed every 5 min (Ctrl+C to stop):   bash scratch/local_progress.sh [seconds]
# Shows: GPU load, every local run's step/stage/acc, queue events, stall-pilot results, P4 runs still unevaluated.
cd "$(dirname "$0")/.."
every=${1:-300}
RUNS="nostep_s5"
while :; do
  clear 2>/dev/null
  echo "=== $(date '+%H:%M:%S')  (refresh every ${every}s)"
  nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv,noheader | sed 's/^/GPU  /'
  echo "--- runs"
  python scratch/run_status.py $RUNS | grep -v not-started | sed 's/^STAT /  /'
  echo "--- queues (last event)"
  for q in 4060 3070 2070b 3070b 4060b; do [ -f logs/local_$q.out ] && echo "  $q: $(tail -n 1 logs/local_$q.out)"; done
  echo "--- stall pilot"
  python -c "
import json,glob
for f in sorted(glob.glob('results/stall_pilot*.json')):
    d=json.load(open(f)); cfg={}
    for v in d.values(): c=cfg.setdefault(v['config'],[0,0]); c[0]+=v['stalled']; c[1]+=1
    print('  '+f.split('/')[-1]+': '+'  '.join(f'{k} {s}/{n} stalled' for k,(s,n) in cfg.items()))"
  [ -f logs/stall_pilot.log ] && tail -n 1 logs/stall_pilot.log | sed 's/^/  /' || echo "  waiting for the 4060 queue"
  echo "--- guard events (last 3)"
  cat logs/guard_{nostep_k8_s3,nostep_k8_s4,nostep_s4,nostep_s5}.log 2>/dev/null | sort | tail -n 3 | sed 's/^/  /'
  sleep "$every"
done
