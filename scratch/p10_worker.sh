#!/usr/bin/env bash
# PROTOCOL10 worker: bash scratch/p10_worker.sh <device>. Claims models by mkdir lock; ctl_s2 only after its verified drain.
cd "$(dirname "$0")/../prosqa"
for m in p8b1_ckpt_s1 p8b1_ctl_s1 p8b1_ckpt_s3 p8b1_ctl_s3 p8b1_ckpt_s2 p8_coconut p8b1_ctl_s2; do
  mkdir res_p10/lock_$m 2>/dev/null || continue
  until [ -f ckpt/${m}_checkpoint_48 ] && { [ $m != p8b1_ctl_s2 ] || grep -q "p8b1_ctl_s2 53555209 drained" ../logs/auto_drain.log 2>/dev/null; }; do sleep 120; done
  python -u segchain_p10.py --ckpt ckpt/${m}_checkpoint_48 --data p8data/p8_test.json --out res_p10/$m.json --device $1 >> ../logs/p10_$m.log 2>&1
  echo "$(date +%T) $m rc $? ($1)" >> ../logs/p10.log
done
echo "$(date +%T) WORKER_DONE $1" >> ../logs/p10.log
