#!/usr/bin/env bash
# PROTOCOL8 amendment 3 (secondary SC) on local cards: bash scratch/p8na_worker.sh <device>. Starts after this card's
# THEORY §62 worker is done; claims jobs by lock dirs. Latent arms: prosqa/eval_na_p8.py (p8_test + p8_na_test, depth >= 7,
# K' 10 / 16). CoT: prosqa/eval_p8.py --mode cot --n_per_m 150 on p8_na_test. Outputs prosqa/res_p8c_na/<arm>_s<seed>.json.
cd "$(dirname "$0")/.."
until grep -q "WORKER_DONE $1" logs/s62.log 2>/dev/null; do sleep 60; done
mkdir -p prosqa/res_p8c_na/locks
for job in cot_s1 cot_s2 cot_s3 cot_s4 coco_s1 at_s1 rt_s1 coco_s2 at_s2 rt_s2 coco_s3 at_s3 rt_s3 coco_s4 at_s4 rt_s4; do
  mkdir prosqa/res_p8c_na/locks/$job 2>/dev/null || continue
  case $job in
    cot_*) python -u prosqa/eval_p8.py --ckpt prosqa/ckpt/p8c_${job}_checkpoint_30 --data prosqa/p8data/p8_na_test.json --mode cot --n_per_m 150 --out prosqa/res_p8c_na/$job.json --device $1 > logs/p8na_$job.log 2>&1 ;;
    *) python -u prosqa/eval_na_p8.py --ckpt prosqa/ckpt/p8c_${job}_checkpoint_48 --out prosqa/res_p8c_na/$job.json --device $1 > logs/p8na_$job.log 2>&1 ;;
  esac
  echo "$(date +%T) $job rc $? ($1)" >> logs/p8na.log
done
echo "$(date +%T) WORKER_DONE $1" >> logs/p8na.log
