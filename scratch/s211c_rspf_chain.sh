#!/usr/bin/env bash
# 211-15 C: RS then PF on the §190 / §193 INJECT models (scratch/inject_rspf.py; bf16 at bs 250) - two bf16 cards, two models each.
#   nohup bash scratch/s211c_rspf_chain.sh >> logs/s211_stage0.out 2>&1 &
cd "$(dirname "$0")/.."
card() {
  local d=$1; shift
  for r in "$@"; do for p in rs pf; do
    python scratch/inject_rspf.py $r --part $p --device $d > logs/s211c_${p}_$r.log 2>&1; echo "$(date '+%F %T') c-$p $r rc $? $(tail -n 1 logs/s211c_${p}_$r.log | cut -c1-100)"
  done; done
}
card cuda:0 s190_rmsnopass_s1 s193_rmswin_s1 &
card cuda:1 s190_rmsnopass_s2 s193_rmswin_s2 &
wait; echo "$(date '+%F %T') 211-15 C rs/pf chain done"
