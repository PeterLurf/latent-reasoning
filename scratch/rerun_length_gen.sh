#!/usr/bin/env bash
# Re-run latent.length_gen (fixed for fork runs, PROTOCOL7 amendment 9) for every evaluated run matching a prefix
# that has a model but no length_gen result:   bash scratch/rerun_length_gen.sh <run prefix> <device>
cd "$(dirname "$0")/.."
for r in $(cat logs/queue_*.out 2>/dev/null | grep ' evaluated ' | awk '{print $3}' | grep "^$1" | sort -u); do
  [ -f runs/$r/model.pt ] && [ ! -s results/length_gen_$r.json ] || continue
  echo "$(date +%T) length_gen $r"
  python -m latent.length_gen runs/$r --out results/length_gen_$r.json --device $2 >> logs/eval_$r.log 2>&1 && echo "$(date +%T) ok $r" || echo "$(date +%T) FAIL $r"
done
echo "$(date +%T) RERUN_DONE"
