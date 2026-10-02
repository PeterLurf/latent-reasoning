#!/usr/bin/env bash
# THEORY §67b on a Vast host: bash scratch/c1_vast_job.sh <tag> <device> <queue file> [P]
# Lines "<out dir> <c1.train args...>"; all lines run concurrently (P, default 4). A run is done iff <out>/eval.json exists.
cd "$(dirname "$0")/.."
tag=$1; dev=$2; q=$3; P=${4:-4}; log=logs/queue_$tag.out; mkdir -p logs prosqa/ckpt c1/runs
pip install -q tiktoken > logs/pip_c1.log 2>&1
W=prosqa/ckpt/gpt2_pretrained.bin
[ -s $W ] || curl -sL -o $W https://huggingface.co/openai-community/gpt2/resolve/main/pytorch_model.bin
sha256sum $W | grep -q '^7c5d3f4b8b76583b' || { echo "$(date +%T) GPT2_SHA_MISMATCH rc 9" >> $log; echo "$(date +%T) QUEUE_DONE" >> $log; exit 1; }
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader >> $log
grep -v '^#' "$q" | grep -v '^$' | while read -r out rest; do n=$(basename $out)
  echo "echo \"\$(date +%T) start $out\" >> $log; python -u -m c1.train $rest --device $dev --out $out > logs/c1_$n.log 2>&1; rc=\$?; [ -s $out/eval.json ] || rc=7; echo \"\$(date +%T) evaluated $out rc \$rc\" >> $log"
done > /tmp/c1_cmds.txt
xargs -d '\n' -P $P -I{} bash -c '{}' < /tmp/c1_cmds.txt
echo "$(date +%T) QUEUE_DONE" >> $log
