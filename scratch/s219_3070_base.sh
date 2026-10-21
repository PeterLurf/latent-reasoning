#!/bin/bash
# §219 TEXT-SELF on the 3070 (cuda:1), uninterrupted (Addendum 2 item 3): G-T18 smoke, then s219_base_s1 = the fp32 re-eval of s170_cot_s1 (§2.9 seal order).
cd /d/Latent; export LATENT_MEM_CAP_GB=6.0
python -B -m boxes.text_eval --run boxes/runs_fsq/s219_base_s1 --part smoke --cot_eval_bs 32 --device cuda:1 --out_dir scratch/_tmp/s219/gt18 > logs/s219_gt18_smoke.log 2>&1; rc=$?
echo "$(date '+%F %T') s219 gt18_smoke rc $rc" >> logs/s219_runs.out
[ $rc -ne 0 ] && exit 1
python -B -m boxes.text_eval --run boxes/runs_fsq/s219_base_s1 --part text --cot_eval_bs 32 --device cuda:1 > logs/s219_base_eval.log 2>&1; rc=$?
echo "$(date '+%F %T') s219 base eval_text rc $rc $(ls boxes/runs_fsq/s219_base_s1/eval_text.json 2>/dev/null)" >> logs/s219_runs.out
