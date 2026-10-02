#!/usr/bin/env bash
# DeltaProduct reproduction check: stop the S5 job, then run the README's S3 Configuration 2 (expected to train).
pkill -f s71_dp_job.sh; sleep 1; pkill -f "python src/main.py"; sleep 3
cd /workspace/DeltaProduct/state_tracking
grep -o "val/best_sequence_accuracy': [0-9.]*" /workspace/Latent/logs/s71_dp_s1.log | tail -1 > /workspace/Latent/logs/dpdiag_env.txt
pip list 2>/dev/null | grep -i -E '^(torch|triton|transformers|flash-linear-attention) ' >> /workspace/Latent/logs/dpdiag_env.txt
PYTHONPATH=$PWD python src/generate_data.py --group=S3 --k=128 --samples=100000 --seed=1 > /tmp/s3gen.log 2>&1
PYTHONPATH=$PWD python src/generate_data.py --group=S3 --k=512 --samples=10000 --seed=2 >> /tmp/s3gen.log 2>&1
WANDB_MODE=disabled PYTHONPATH=$PWD python src/main.py train --group=S3 --k=128 --k_test=512 --n_layers=1 --epochs=100 \
  --allow_neg_eigval=True --num_householder=2 --batch_size=2048 --seed=666 --lr=1e-3 --n_heads=8 --use_scheduler=True \
  --logging=False > /workspace/Latent/logs/dpdiag_s3_100.log 2>&1
echo DIAG_DONE >> /workspace/Latent/logs/dpdiag_s3_100.log
