#!/usr/bin/env bash
# THEORY §67 amendment 67-2: d1 at M = 12, LN heads + distinct codes, 8,000 updates, lr 1e-4 (pilot configuration 8).
#   bash scratch/c1_672.sh <device> <seeds...>
cd "$(dirname "$0")/.."
dev=$1; shift
for s in "$@"; do
  o=c1/runs/a672_d1_M12_s$s; [ -f $o/eval.json ] && continue
  python -u -m c1.train --arm d1 --M 12 --head_norm ln --code_init distinct --updates 8000 --bs 512 --lr 1e-4 --seed $s --device $dev --out $o > logs/c1_a672_s$s.log 2>&1
  echo "$(date +%T) a672_s$s $(python -c "import json;e=json.load(open('$o/eval.json'));print(e['certificate']['certified'], e['certificate']['bad_pairs'], e['acc'])" 2>&1 | tail -1)" >> logs/c1_a672.log
done
echo "$(date +%T) DONE $dev" >> logs/c1_a672.log
