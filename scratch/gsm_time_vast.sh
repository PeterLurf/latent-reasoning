#!/usr/bin/env bash
# GSM8K step-2 prep (timing only, no training result): official Coconut code on a Vast host.
#   bash scratch/gsm_time_vast.sh <tag> <device> <unused queue file>
# CoT stage and the final Coconut stage (6 latents), each one debug epoch (5,000 train examples, bs 32, fp32,
# 1 GPU); wall time per epoch -> per-epoch estimate for 385,620 examples. Log logs/queue_<tag>.out.
cd "$(dirname "$0")/.."
tag=$1; log=logs/queue_$tag.out; mkdir -p logs; W=/workspace/coconut
{ cd /workspace && git clone -q https://github.com/facebookresearch/coconut.git && cd coconut && git checkout -q 27273cb && \
  pip install -q transformers==4.46.2 datasets==3.1.0 wandb==0.18.7 > /dev/null 2>&1 && mkdir -p data && \
  B=da03/Internalize_CoT_Step_by_Step/e06a32ee5e4cd117171daeb4755d2a97ece62761/data/gsm8k && \
  wget -q https://media.githubusercontent.com/media/$B/train.txt -O data/gsm_train.txt && \
  wget -q https://raw.githubusercontent.com/$B/valid.txt -O data/gsm_valid.txt && \
  for s in train valid; do python preprocessing/gsm_icot.py $s; done; } > /workspace/Latent/logs/gsm_time_setup.log 2>&1
cd $W; sha256sum data/gsm_train.json >> /workspace/Latent/$log
mk() { sed -e "s#YOUR_PATH_TO_SAVE_THE_MODEL#/workspace/ck_$1#" -e "s#^debug: False#debug: True#" -e "s#^num_epochs: .*#num_epochs: $3#" \
         -e "s#^resume: .*#resume: $2#" -e "s#YOUR_PATH_TO_COT_CHECKPOINT#None#" args/$4 > args/time_$1.yaml; }
mk cot 0 1 gsm_cot.yaml; mk coco 9 10 gsm_coconut.yaml
for c in cot coco; do
  t0=$(date +%s); WANDB_MODE=disabled torchrun --nnodes 1 --nproc_per_node 1 run.py args/time_$c.yaml > /workspace/Latent/logs/gsm_time_$c.log 2>&1; rc=$?
  echo "$(date +%T) $c rc $rc wall_s $(( $(date +%s) - t0 ))" >> /workspace/Latent/$log
done
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader >> /workspace/Latent/$log
echo "$(date +%T) QUEUE_DONE" >> /workspace/Latent/$log
