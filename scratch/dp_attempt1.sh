#!/usr/bin/env bash
# §71 amendment 71-2, diagnosed configuration 1: the repo's stated environment (README: triton > 3.1.0). Our image had
# torch 2.5.1 + triton 3.1.0, where the README's own S3 Configuration 2 reached only 0.18 sequence accuracy at 512 (token
# 0.96). Upgrade to torch 2.6.0 / triton 3.2.0, re-run the S3 check (100 epochs); log versions.
cd /workspace/DeltaProduct/state_tracking
pip install -q "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0" --index-url https://download.pytorch.org/whl/cu124 > /workspace/Latent/logs/dp_attempt1_pip.log 2>&1
pip list 2>/dev/null | grep -i -E '^(torch|triton|transformers|flash-linear-attention) ' > /workspace/Latent/logs/dp_attempt1_env.txt
WANDB_MODE=disabled PYTHONPATH=$PWD python src/main.py train --group=S3 --k=128 --k_test=512 --n_layers=1 --epochs=100 \
  --allow_neg_eigval=True --num_householder=2 --batch_size=2048 --seed=666 --lr=1e-3 --n_heads=8 --use_scheduler=True \
  --logging=False > /workspace/Latent/logs/dp_attempt1_s3b.log 2>&1
echo ATTEMPT1_S3_DONE >> /workspace/Latent/logs/dp_attempt1_s3b.log
