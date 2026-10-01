#!/usr/bin/env bash
# THEORY §63 (F4) on a Vast host: bash scratch/f4_vast_job.sh <tag> <device> <queue file>; queue lines "<model dir> <name>".
# Per model: arms S (default) and N (--no_step) x translator seeds 1, 2 (3000 steps, bs 128) -> theory/runs/f4/<name>_<A>_t<s>;
# training runs 4 at a time on the GPU; then the swap per arm (t1 <-> t2). Ends with QUEUE_DONE in logs/queue_<tag>.out.
cd "$(dirname "$0")/.."
tag=$1; dev=$2; q=$3; log=logs/queue_$tag.out; mkdir -p logs theory/runs/f4
grep -v '^#' "$q" | grep -v '^$' > /tmp/f4_models.txt
: > /tmp/f4_cmds.txt
while read -r md name; do for A in S N; do for s in 1 2; do
  fl=""; [ $A = N ] && fl="--no_step"
  echo "python -u theory/f4_translator.py --model $md --seed $s $fl --device $dev --out theory/runs/f4/${name}_${A}_t$s > logs/f4_${name}_${A}_t$s.log 2>&1; echo \"\$(date +%T) trained ${name}_${A}_t$s rc \$?\" >> $log" >> /tmp/f4_cmds.txt
done; done; done < /tmp/f4_models.txt
xargs -P 4 -I{} bash -c '{}' < /tmp/f4_cmds.txt
while read -r md name; do for A in S N; do
  python -u theory/f4_translator.py --swap theory/runs/f4/${name}_${A}_t1 theory/runs/f4/${name}_${A}_t2 --device $dev >> logs/f4_${name}_${A}_swap.log 2>&1
  rc=$?; for s in 1 2; do [ -s theory/runs/f4/${name}_${A}_t$s/eval_f4.json ] || rc=7; done
  echo "$(date +%T) evaluated ${name}_${A} rc $rc" >> $log
done; done < /tmp/f4_models.txt
echo "$(date +%T) QUEUE_DONE" >> $log
