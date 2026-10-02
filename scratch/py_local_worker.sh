#!/usr/bin/env bash
# Local rolling worker for any python module, one job per card: bash scratch/py_local_worker.sh <device> <queue>;
# lines "<out dir> <module> <args...>"; claims by lock dirs c1/runs/locks/<name>; skips runs with eval.json.
cd "$(dirname "$0")/.."
mkdir -p c1/runs/locks
grep -v '^#' "$2" | grep -v '^$' | while read -r out mod rest; do n=$(basename $out)
  [ -s $out/eval.json ] && continue
  mkdir c1/runs/locks/$n 2>/dev/null || continue
  python -u -m $mod $rest --device $1 --out $out > logs/c1_$n.log 2>&1
  echo "$(date +%T) $n rc $? ($1)" >> logs/c1_local.log
done
echo "$(date +%T) WORKER_DONE $1 $2" >> logs/c1_local.log
