#!/bin/bash
# 4060 (cuda:0, UUID GPU-be1bdfb5) queue i: §222 PROBE-S S1 certification (PS_v1 at 7174fc3b; Theory 10-21 answers (1)-(4)).
# (1) after queue h's done line ("4060 queue d done") and no python on the 4060: m_rule (untrained model, certification pool 22700 / dirs 22750).
# (2) then wait for §218 A1's NON-PROBE eval line (s218_a1_post_landing.sh, registered first on this card) and a free 4060: cert untrained, then cert ref.
cd /d/Latent; export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
L=logs/s222_chain.out
busy() { nvidia-smi --query-compute-apps=gpu_uuid,process_name --format=csv,noheader | grep "GPU-be1bdfb5" | grep -qi python; }
until grep -q "4060 queue d done" logs/s218_ref_chain.out && ! busy; do sleep 60; done
echo "$(date '+%F %T') m_rule start (4060)" >> $L
python -B scratch/s222_cert.py --mode m_rule --device cuda:0 --pool_n 25000 > logs/s222_m_rule.log 2>&1; rc=$?
echo "$(date '+%F %T') m_rule rc $rc $(ls scratch/gate_s222/m_rule.json 2>/dev/null)" >> $L
[ -s scratch/gate_s222/m_rule.json ] || { echo "$(date '+%F %T') m_rule.json missing - stop" >> $L; exit 1; }
until grep -q "A1 eval_lvb NON-PROBE parts" logs/s218_ref_chain.out && ! busy; do sleep 60; done; sleep 60; until ! busy; do sleep 60; done
for M in untrained ref; do
  echo "$(date '+%F %T') cert $M start (4060)" >> $L
  python -B scratch/s222_cert.py --mode cert --model $M --device cuda:0 --resume > logs/s222_cert_$M.log 2>&1; rc=$?
  echo "$(date '+%F %T') cert $M rc $rc $(ls scratch/gate_s222/cert_$M.json 2>/dev/null)" >> $L
done
echo "$(date '+%F %T') s222 queue i done" >> $L
