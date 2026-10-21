#!/bin/bash
# Note 220-3 (Audit-approved): CODI20 official-vs-official self-consistency on the 2070 (cuda:2, fp32; the CODI20 card) after 218-17's k3swap there:
# two official-only runs, order shuffled with rng 22030 / 22031 -> scratch/s220/official_shuffle_<seed>.json, then gate_official_self.json (A1..A3).
cd /d/Latent; export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
L=logs/s220_stage0.out
until grep -q "218-17 k3swap (untrained, 2070) rc" logs/s218_ref_chain.out 2>/dev/null; do sleep 60; done
for sd in 22030 22031; do
  python -B scratch/s220_gcs.py --model codi20 --device cuda:2 --official_shuffle $sd > logs/s220_codi20_shuffle_$sd.log 2>&1; rc=$?
  echo "$(date '+%F %T') s220 codi20 official_shuffle $sd (2070) rc $rc" >> $L
done
python -B scratch/s220_official_self.py --model codi20 > logs/s220_codi20_official_self.log 2>&1; rc=$?
echo "$(date '+%F %T') s220 codi20 official_self rc $rc: $(tail -n 1 logs/s220_codi20_official_self.log)" >> $L
