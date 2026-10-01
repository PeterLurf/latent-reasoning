#!/usr/bin/env bash
# THEORY §61 (F1-S) on a Vast host: bash scratch/f1s_vast_job.sh <tag> <device> <queue file>; queue lines "<run> <seed>".
cd "$(dirname "$0")/.."
tag=$1; dev=$2; q=$3; log=logs/queue_$tag.out; mkdir -p logs prosqa/runs_ail prosqa/ckpt
pip install -q tiktoken > logs/pip_ail.log 2>&1
W=prosqa/ckpt/gpt2_pretrained.bin
[ -s $W ] || curl -sL -o $W https://huggingface.co/openai-community/gpt2/resolve/main/pytorch_model.bin
sha256sum $W | grep -q '^7c5d3f4b8b76583b' || { echo "$(date +%T) GPT2_SHA_MISMATCH rc 9" >> $log; echo "$(date +%T) QUEUE_DONE" >> $log; exit 1; }
grep -v '^#' "$q" | grep -v '^$' | while read -r r seed; do
  echo "$(date +%T) start $r" >> $log
  python -u prosqa/ail_f1s.py --out prosqa/runs_ail/$r --seed $seed --device $dev > logs/$r.out 2>&1
  echo "$(date +%T) trained $r rc $?" >> $log
  python -u prosqa/ail_f1s.py --eval prosqa/runs_ail/$r --device $dev >> logs/$r.out 2>&1
  echo "$(date +%T) evaluated $r rc $?" >> $log
done
echo "$(date +%T) QUEUE_DONE" >> $log
