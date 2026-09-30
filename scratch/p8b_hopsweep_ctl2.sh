#!/usr/bin/env bash
# Hop sweep for ctl_s2 (the last model), after its verified drain and after the 2070's PROTOCOL10 worker is done.
cd "$(dirname "$0")/../prosqa"
until grep -q "p8b1_ctl_s2 53555209 drained" ../logs/auto_drain.log 2>/dev/null && grep -q "WORKER_DONE cuda:2" ../logs/p10.log 2>/dev/null; do sleep 120; done
python -u hopsweep_p8b.py --ckpt ckpt/p8b1_ctl_s2_checkpoint_48 --data p8data/p8_test.json --out res_p8b/hopsweep_ctl_s2.json --n_per_d 40 --skip_probe --device cuda:2 >> ../logs/p8b_hopsweep.log 2>&1
echo "$(date +%T) ctl_s2 rc $?" >> ../logs/p8b_hopsweep.log
echo "$(date +%T) HOPSWEEP_DONE" >> ../logs/p8b_hopsweep.log
