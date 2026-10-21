#!/bin/bash
# Note 218-25 revised feasibility S2 (PROBE2, antisymmetric head; descriptive) on the 2070 (cuda:2, fp32; card recorded), Theory 10-21 card split:
# after §221 CODI20 gfid finishes there. -> g_l5_known/feas2_S2_*.json.
cd /d/Latent; export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
L=logs/s218_ref_chain.out
until grep -q "s221 gfid codi20 (2070) rc" logs/s221_gfid.out 2>/dev/null; do sleep 60; done
python -B scratch/s218_gl5_known.py --device cuda:2 --arms feas2_s2 > logs/s218_feas2_s2.log 2>&1; rc=$?
echo "$(date '+%F %T') 218-25 feas2 S2 (untrained, 2070) rc $rc" >> $L
