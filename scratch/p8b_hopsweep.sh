#!/usr/bin/env bash
# Exploratory PROTOCOL8b follow-up on the local 2070: hop sweep of the written oracle checkpoint on the 6 step-1 models.
# Started before P8b1 was scored, so (Audit guard) it runs on p8_test problems disjoint from the P8b1 probe
# (--skip_probe) and prints no accuracies; results are not read until the P8b1 verdict is committed.
# Log logs/p8b_hopsweep.log; results prosqa/res_p8b/hopsweep_<model>.json. ctl_s2 waits for its drained checkpoint.
cd "$(dirname "$0")/../prosqa"
for m in ckpt_s1 ckpt_s2 ckpt_s3 ctl_s1 ctl_s3 ctl_s2; do
  # ctl_s2: only after its verified drain (the checkpoint file exists while scp is still writing it)
  until [ -f ckpt/p8b1_${m}_checkpoint_48 ] && { [ $m != ctl_s2 ] || grep -q "p8b1_ctl_s2 53555209 drained" ../logs/auto_drain.log 2>/dev/null; }; do sleep 120; done
  python -u hopsweep_p8b.py --ckpt ckpt/p8b1_${m}_checkpoint_48 --data p8data/p8_test.json --out res_p8b/hopsweep_$m.json --n_per_d 40 --skip_probe --device ${1:-cuda:2} >> ../logs/p8b_hopsweep.log 2>&1
  echo "$(date +%T) $m rc $?" >> ../logs/p8b_hopsweep.log
done
echo "$(date +%T) HOPSWEEP_DONE" >> ../logs/p8b_hopsweep.log
