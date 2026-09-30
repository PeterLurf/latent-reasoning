#!/usr/bin/env bash
cd "$(dirname "$0")/../prosqa"
python -u reanchor_p8.py --ckpt ckpt/p8_coconut_checkpoint_48 --data p8data/p8_test.json --out res_p8b/p8b1_base_ref.json --n_per_m 150 --Kp 10 --device ${1:-cuda:2} --random_pos > ../logs/p8b1_base_ref.log 2>&1
echo "$(date +%T) BASE_REF_DONE rc $?" >> ../logs/p8b1_base_ref.log
