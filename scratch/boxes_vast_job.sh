#!/usr/bin/env bash
# THEORY §66 on a Vast host: bash scratch/boxes_vast_job.sh <tag> <device> <queue file> [parallel, default 4].
# Lines "<run> <arm> <k> <seed> <steps> <bs> <mode>"; mode pilot = train only; full = train (+ edit eval for WC) + bench boxes.
# Writes boxes/runs/<run>/host.txt. A run is done only if eval_train.json exists (and, for full, the bench log says WROTE).
cd "$(dirname "$0")/.."
tag=$1; dev=$2; q=$3; P=${4:-4}; log=logs/queue_$tag.out; mkdir -p logs boxes/runs prosqa/ckpt bench/results
pip install -q tiktoken > logs/pip_boxes.log 2>&1
W=prosqa/ckpt/gpt2_pretrained.bin
for try in 1 2 3; do sha256sum $W 2>/dev/null | grep -q '^7c5d3f4b8b76583b' && break; rm -f $W; curl -sL -o $W https://huggingface.co/openai-community/gpt2/resolve/main/pytorch_model.bin; done   # retry a bad download
sha256sum $W | grep -q '^7c5d3f4b8b76583b' || { echo "$(date +%T) GPT2_SHA_MISMATCH rc 9" >> $log; echo "$(date +%T) QUEUE_DONE" >> $log; exit 1; }
grep -v '^#' "$q" | grep -v '^$' | while read -r r arm k seed steps bs mode; do
  echo "d=boxes/runs/$r; mkdir -p \$d; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader > \$d/host.txt; echo \"\$(date +%T) start $r\" >> $log; python -u boxes/train.py --arm $arm --k $k --seed $seed --steps $steps --bs $bs --device $dev --out \$d > logs/$r.out 2>&1; rc=0; [ -s \$d/eval_train.json ] || rc=7; if [ $mode = full ] && [ \$rc = 0 ]; then python -u -m bench.run --task boxes --model boxes:\$d --tag boxes_$r --device $dev > logs/bench_$r.log 2>&1; grep -q WROTE logs/bench_$r.log || rc=8; fi; echo \"\$(date +%T) evaluated $r rc \$rc\" >> $log"
done > /tmp/boxes_cmds.txt
xargs -d '\n' -P $P -I{} bash -c '{}' < /tmp/boxes_cmds.txt
echo "$(date +%T) QUEUE_DONE" >> $log
