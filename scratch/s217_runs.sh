#!/usr/bin/env bash
# §217 (registered 45d7d33c; Audit GO 13:4x 10-18 on a5f5fa6b): eval_causal_deep.json on the 4060 (cuda:0), fp32, eval_bs 16, one model at a time.
#   nohup bash scratch/s217_runs.sh >> logs/s217_runs.out 2>&1 &
cd "$(dirname "$0")/.."
for r in s215_ftself_s1 s216_r2_s1 s213_d3t_s1 s215_ftctrl_s1 s216_d3t_s2 s216_r1_s2; do
  python -m boxes.vbank_causal --run boxes/runs_fsq/$r --device cuda:0 --eval_bs 16 > logs/s217_causal_$r.log 2>&1; rc=$?
  echo "$(date '+%F %T') $r causal_deep rc $rc $(tail -n 1 logs/s217_causal_$r.log | cut -c1-140)"
done
echo "$(date '+%F %T') §217 runs done"
