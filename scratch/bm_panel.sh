#!/usr/bin/env bash
# THEORY §60 seed panel (frozen settings, bm/runs/PANEL_FREEZE.md): BM-oc (lambda 1) and BM-no x seeds 1-4, claimed by lock dirs.
# bash scratch/bm_panel.sh <device>
cd "$(dirname "$0")/.."
mkdir -p bm/runs/panel_locks
for s in 1 2 3 4; do for arm in oc no; do
  r=${arm}_s$s; lam=1; [ $arm = no ] && lam=0
  mkdir bm/runs/panel_locks/$r 2>/dev/null || continue
  python -u bm/run.py train --pool bm/data_pool --out bm/runs/$r --lam $lam --seed $s --steps 10000 --bs 128 --device $1 > logs/bm_$r.log 2>&1
  python -u bm/run.py eval --pool bm/data_pool --run bm/runs/$r --device $1 >> logs/bm_$r.log 2>&1
  echo "$(date +%T) $r rc $? ($1)" >> logs/bm_panel.log
done; done
echo "$(date +%T) WORKER_DONE $1" >> logs/bm_panel.log
