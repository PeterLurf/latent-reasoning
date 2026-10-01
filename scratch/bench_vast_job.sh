#!/usr/bin/env bash
# bench batch on a Vast host: bash scratch/bench_vast_job.sh <tag> <device> <queue file>; lines "<name> <task> <spec> [extra]".
# Runs 4 jobs at a time on the GPU; a job counts as done only if its log says WROTE. Ends with QUEUE_DONE.
cd "$(dirname "$0")/.."
tag=$1; dev=$2; q=$3; log=logs/queue_$tag.out; mkdir -p logs bench/results prosqa/ckpt
pip install -q tiktoken > logs/pip_bench.log 2>&1
W=prosqa/ckpt/gpt2_pretrained.bin
if grep -q "f1s:" "$q"; then [ -s $W ] || curl -sL -o $W https://huggingface.co/openai-community/gpt2/resolve/main/pytorch_model.bin; fi
grep -v '^#' "$q" | grep -v '^$' | while read -r n t sp rest; do
  echo "python -u -m bench.run --task $t --model \"$sp\" --device $dev $rest > logs/bench_$n.log 2>&1; grep -q WROTE logs/bench_$n.log && rc=0 || rc=7; echo \"\$(date +%T) evaluated $n rc \$rc\" >> $log"
done > /tmp/bench_cmds.txt
xargs -P 4 -I{} bash -c '{}' < /tmp/bench_cmds.txt
echo "$(date +%T) QUEUE_DONE" >> $log
