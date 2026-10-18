#!/usr/bin/env bash
# A0-1 (note 191-4; §218 §2.8): eval-only per-item re-evaluation of the two kept §191 models on the 2070 (cuda:2), fp32, eval_bs 16, full p8_na_test (n 4,053),
# each with its own queue arguments (gen_q_b271 / b272) -> <run>/eval_a01.json.   nohup bash scratch/a01_runs.sh >> logs/a01_runs.out 2>&1 &
cd "$(dirname "$0")/.."
E="--eval_only --a01 --fp32 --eval_bs 16 --device cuda:2"
python -m boxes.prosqa168s2 --arm stepaux --think_k 10 --aux_step 1.0 --aux_fade 12000,12000 --fast_add --chunk_delta 64 --ckpt_passes --cache_lower --relabel --test500_every 1000 --seed 1 --steps 12000 --bs 64 $E --out boxes/runs_fsq/s191_stepaux_relabel_s1 > logs/a01_stepaux.log 2>&1; rc=$?
echo "$(date '+%F %T') s191_stepaux_relabel_s1 a01 rc $rc $(grep '^A01' logs/a01_stepaux.log | cut -c1-200)"
python -m boxes.prosqa168s2 --arm textcot --relabel --test500_every 1000 --seed 1 --steps 12000 --bs 64 $E --out boxes/runs_fsq/s191_textcot_relabel_s1 > logs/a01_textcot.log 2>&1; rc=$?
echo "$(date '+%F %T') s191_textcot_relabel_s1 a01 rc $rc $(grep '^A01' logs/a01_textcot.log | cut -c1-200)"
echo "$(date '+%F %T') A0-1 done"
