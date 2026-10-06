#!/usr/bin/env bash
# THEORY note 131-3: four held-out S5 runs on the local RTX 2070 (cuda:2 only), sequential. Plain fsq120 §120 FINAL S5, 24k, no
# --stop_plateau, 127-3 guard (--probe127), --no_cert, train_driver recorded; fp32 (Turing: amp None, eval_dtype "None").
# Memory check 2026-10-06 (scratch/memcheck_fsq120.py): L 32 fp32 --micro 4 -> peak 5.05 GiB of 8; eval 250 x 512 -> 0.96 GiB.
# Skips a seed whose eval.json exists. Stop file: scratch/c1_q_131x_2070.stop.   nohup bash scratch/c1_q_131x_2070.sh &
cd "$(dirname "$0")/.."
for s in 401 402 403 404; do
  [ -f scratch/c1_q_131x_2070.stop ] && break
  out=boxes/runs_fsq/s131x_s5_s${s}_2070
  [ -s $out/eval.json ] && continue
  OMP_NUM_THREADS=3 python -u -m boxes.fsq120 --family s5 --arm final --seed $s --micro 4 --no_cert --probe127 --device cuda:2 --out $out \
    > logs/c1_s131x_s5_s${s}_2070.log 2>&1
  echo "$(date +%T) s131x_s5_s${s}_2070 rc $? eval $([ -s $out/eval.json ] && echo yes || echo NO) (cuda:2)" >> logs/c1_local.log
done
echo "$(date +%T) c1_q_131x_2070 done" >> logs/c1_local.log
