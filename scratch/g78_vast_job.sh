#!/usr/bin/env bash
# THEORY §78 on a Vast host: bash scratch/g78_vast_job.sh <tag> <device> <queue file> [P]
# Queue lines "<arm> <seed> <mode>" (mode full | smoke); all lines run concurrently (P, default 2) on the one GPU.
# full: 6 epochs, bs 16 x grad_acc 2, bf16 (gsmlen/PILOT78.md), then predictions on test and aug_long.
# smoke: 50 steps, predictions on 20 test items. A run is "evaluated" iff its pred_test.json exists.
cd "$(dirname "$0")/.."
tag=$1; dev=$2; q=$3; P=${4:-2}; log=logs/queue_$tag.out; mkdir -p logs gsmlen/runs
pip install -q transformers==4.46.2 > logs/pip_g78.log 2>&1
for try in 1 2 3; do python -c "from transformers import GPT2LMHeadModel, GPT2TokenizerFast; GPT2LMHeadModel.from_pretrained('openai-community/gpt2'); GPT2TokenizerFast.from_pretrained('openai-community/gpt2')" >> logs/pip_g78.log 2>&1 && break; sleep 30; done
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader >> $log
grep -v '^#' "$q" | grep -v '^$' | while read -r arm seed mode; do
  out=gsmlen/runs/${arm}_s${seed}; tr="--epochs 6"; pr=""
  [ "$mode" = smoke ] && { out=gsmlen/runs/smoke_${arm}_s${seed}; tr="--max_steps 50"; pr="--max_valid 20"; }
  n=$(basename $out)
  cmd="echo \"\$(date +%T) start $out\" >> $log; python -u gsmlen/train78.py train --arm $arm --seed $seed $tr --bs 16 --grad_acc 2 --device $dev --out $out > logs/g78_$n.log 2>&1"
  cmd="$cmd && python -u gsmlen/train78.py predict --arm $arm --ckpt $out/model --split test $pr --device $dev --out $out/pred_test.json >> logs/g78_$n.log 2>&1"
  [ "$mode" = full ] && cmd="$cmd && python -u gsmlen/train78.py predict --arm $arm --ckpt $out/model --split aug_long --device $dev --out $out/pred_aug_long.json >> logs/g78_$n.log 2>&1"
  echo "$cmd; rc=\$?; [ -s $out/pred_test.json ] || rc=7; echo \"\$(date +%T) evaluated $out rc \$rc\" >> $log"
done > /tmp/g78_cmds.txt
xargs -d '\n' -P $P -I{} bash -c '{}' < /tmp/g78_cmds.txt
echo "$(date +%T) QUEUE_DONE" >> $log
