#!/usr/bin/env bash
# §216 G-S18 baselines (SD2_v1 §7; run only after s216_r2_s1 launched - admitted 05:13 10-18): part deep (eval_deep.json, §4.2) on the 3070, fp32, bs 16 (G-S16),
# FT-SELF (required), then D3-T s1 and FT-CTRL (optional), one job at a time. Output in each run dir (the registered baseline locations).
#   nohup bash scratch/s216_baselines.sh >> logs/s216_baselines.out 2>&1 &
cd "$(dirname "$0")/.."
for r in s215_ftself_s1 s213_d3t_s1 s215_ftctrl_s1; do
  python -m boxes.vbank_diag --run boxes/runs_fsq/$r --part deep --device cuda:1 --eval_bs 16 > logs/s216_deep_$r.log 2>&1; rc=$?
  echo "$(date '+%F %T') $r deep rc $rc $(tail -n 1 logs/s216_deep_$r.log | cut -c1-140)"
done
echo "$(date '+%F %T') §216 G-S18 baselines done"
