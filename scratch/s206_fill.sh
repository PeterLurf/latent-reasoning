#!/usr/bin/env bash
# §206 launcher (Audit APPROVED adef2e6a + 02:2x): J1 (b300; line 1 = W's eval-only ref) first, then J0 (b299), each with its own .extra; fill_generic_px skips
# the reserved 570.172.08 (s205_fill's), so the two fillers never compete for an offer. Pair guard: scratch/cost_guard.py s206 (marks 200 / 1k / 4k, J0 at J1's
# figure until it has data, destroys both cells on ALERT).   nohup bash scratch/s206_fill.sh >> logs/fill_s206.out 2>&1 &
cd "$(dirname "$0")/.."
for n in 300 299; do
  EXTRA_120="$(tr '\n' ' ' < scratch/gen_q_b$n.extra)" DPH_MAX="$(cat scratch/gen_q_b$n.dph)" bash scratch/fill_generic_px.sh s206_$n $n
done
echo "$(date '+%F %T') s206_fill: both cells STARTED"
