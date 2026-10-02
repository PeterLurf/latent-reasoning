#!/usr/bin/env bash
# §68 sizing (timing only): bf16 throughput on a Vast host. TC = official Coconut code with bf16: True (debug epoch,
# 5,000 items, bs 32, 1 GPU); WS = s68/ws.py at bs 32, 200 optimizer steps, bf16 autocast and fp32.
#   bash scratch/gsm_time_vast2.sh <tag> <device> <unused queue>
cd "$(dirname "$0")/.."
tag=$1; log=/workspace/Latent/logs/queue_$tag.out; mkdir -p logs
{ cd /workspace && git clone -q https://github.com/facebookresearch/coconut.git && cd coconut && git checkout -q 27273cb && \
  pip install -q transformers==4.46.2 datasets==3.1.0 wandb==0.18.7 && mkdir -p data && \
  B=da03/Internalize_CoT_Step_by_Step/e06a32ee5e4cd117171daeb4755d2a97ece62761/data/gsm8k && \
  wget -q https://media.githubusercontent.com/media/$B/train.txt -O data/gsm_train.txt && \
  wget -q https://raw.githubusercontent.com/$B/valid.txt -O data/gsm_valid.txt && \
  for s in train valid; do python preprocessing/gsm_icot.py $s; done; } > /workspace/Latent/logs/gsm_time2_setup.log 2>&1
cd /workspace/coconut; sha256sum data/gsm_train.json >> $log
sed -e "s#YOUR_PATH_TO_SAVE_THE_MODEL#/workspace/ck_cotbf#" -e "s#^debug: False#debug: True#" -e "s#^num_epochs: .*#num_epochs: 1#" \
    -e "s#^bf16: False#bf16: True#" args/gsm_cot.yaml > args/time_cotbf.yaml
t0=$(date +%s); WANDB_MODE=disabled torchrun --nnodes 1 --nproc_per_node 1 run.py args/time_cotbf.yaml > /workspace/Latent/logs/gsm_time2_cotbf.log 2>&1
echo "$(date +%T) cot_bf16 rc $? wall_s $(( $(date +%s) - t0 ))" >> $log
cd /workspace/Latent
for p in bf16 fp32; do
  f=""; [ $p = bf16 ] && f="--bf16"
  t0=$(date +%s); python -u -m s68.ws train --train /workspace/coconut/data/gsm_train.json --valid /workspace/coconut/data/gsm_valid.json \
     --out /workspace/ws_$p --bs 32 --grad_acc 1 --max_steps 200 --max_valid 100 $f > logs/gsm_time2_ws_$p.log 2>&1
  echo "$(date +%T) ws_$p rc $? wall_s $(( $(date +%s) - t0 ))" >> $log
  cp /workspace/ws_$p/train_log.jsonl logs/gsm_time2_ws_${p}_trainlog.jsonl 2>/dev/null; cp /workspace/ws_$p/budget.json logs/gsm_time2_ws_${p}_budget.json 2>/dev/null
done
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader >> $log
echo "$(date +%T) QUEUE_DONE" >> $log
