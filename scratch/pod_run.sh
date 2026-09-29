#!/usr/bin/env bash
# One cloud training run under the stage-1 guard:  bash scratch/pod_run.sh <name> <extra train args...>
# Base recipe = PROTOCOL4 seed-0 main recipe; records host GPU/driver/torch into runs/<name>.host.txt.
set -u
name=$1; shift
cd /workspace/Latent && mkdir -p runs logs
{ nvidia-smi -L; nvidia-smi --query-gpu=driver_version --format=csv,noheader;
  python -c "import torch;print('torch',torch.__version__)"; date -u; } > "runs/$name.host.txt"
exec python scratch/stage1_guard.py "runs/$name" 0 3 --wait -- -m latent.train --mode coconut --bs 512 \
  --steps 30000 --warmup 300 --eval_every 1000 --aux_steps 10000 --aux_anneal 5000 --children_weight 1 \
  --curriculum gated --init gpt2 --device cuda:0 --resume --out "runs/$name" "$@"
