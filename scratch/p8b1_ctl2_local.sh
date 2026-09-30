#!/usr/bin/env bash
# Local re-run of the P8b1 probe for ctl_s2 (the Vast host's probe runs ~10x slower than the others). Same command and seed as pod_p8b1.sh.
cd "$(dirname "$0")/../prosqa"
python -u reanchor_p8.py --ckpt ckpt/p8b1_ctl_s2_checkpoint_48.part --data p8data/p8_test.json --out res_p8b1/p8b1_ctl_s2_local.json --n_per_m 150 --Kp 10 --device cuda:0 --random_pos > ../logs/p8b1_ctl2_local.log 2>&1
echo "$(date +%T) CTL2_LOCAL_DONE rc $?" >> ../logs/p8b1_ctl2_local.log
