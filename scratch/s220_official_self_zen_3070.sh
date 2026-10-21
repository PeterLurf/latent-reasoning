#!/bin/bash
# Note 220-3 (Audit-approved): zen-gpt2 official-vs-official self-consistency on the 3070 (cuda:1, fp32; zen's card), AFTER zen's stage-0 run AND after
# the §219 T1 post-landing chain releases the 3070 (§219 evals have priority there: its DONE line or a 'held' ALERT). Shuffles 22030 / 22031 ->
# scratch/s220/zen_codi_gpt2/official_shuffle_<seed>.json, then gate_official_self.json.
cd /d/Latent; export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
L=logs/s220_stage0.out
until grep -q "s220 zen (3070) rc" $L && grep -qE "T1 post-landing chain DONE|held" logs/s219_post_landing.out 2>/dev/null; do sleep 60; done
for sd in 22030 22031; do
  python -B scratch/s220_gcs.py --model zen --device cuda:1 --official_shuffle $sd > logs/s220_zen_shuffle_$sd.log 2>&1; rc=$?
  echo "$(date '+%F %T') s220 zen official_shuffle $sd (3070) rc $rc" >> $L
done
python -B scratch/s220_official_self.py --model zen > logs/s220_zen_official_self.log 2>&1; rc=$?
echo "$(date '+%F %T') s220 zen official_self rc $rc: $(tail -n 1 logs/s220_zen_official_self.log)" >> $L
