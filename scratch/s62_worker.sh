#!/usr/bin/env bash
# THEORY §62 worker: bash scratch/s62_worker.sh <device>. Pilot L round 1 exactly (distill_reanchor.py defaults: 3000 steps,
# bs 256, lr 1e-4, widths 2,4,6, K'=12) with --teacher gt and the arm's noise flags; then theory/eval_arch.py. Lock dirs.
cd "$(dirname "$0")/.."
mkdir -p theory/runs/s62_locks
grep -v '^#' scratch/s62_queue.txt | while read -r r seed head flags; do
  mkdir theory/runs/s62_locks/$r 2>/dev/null || continue
  python -u theory/distill_reanchor.py --init_from runs/p30_s$seed/model.pt --teacher gt --head $head --seed $seed $flags \
    --out theory/runs/$r --device $1 > logs/s62_$r.log 2>&1
  rc=$?
  [ $rc -eq 0 ] && python -u theory/eval_arch.py theory/runs/$r --device $1 >> logs/s62_$r.log 2>&1
  echo "$(date +%T) $r train rc $rc eval rc $? ($1)" >> logs/s62.log
done
echo "$(date +%T) WORKER_DONE $1" >> logs/s62.log
