#!/bin/bash
# §221 G-FID, CODI20 on its pinned card, the 2070 (cuda:2), after the Note 218-23 sweep's 2070 part (Note 221-1; Audit GO): s221_gfid.py --model codi20
# (determinism full bs-1 INTACT re-run + the pairs[0:50] subset) -> boxes/runs_fsq/gsm_codi_20ep_s1/gate_fid.json.
cd /d/Latent; export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
L=logs/s221_gfid.out
until grep -q "218-23 sweep (2070 part) done" logs/s218_ref_chain.out 2>/dev/null; do sleep 60; done
python -B scratch/s221_gfid.py --model codi20 --device cuda:2 > logs/s221_gfid_codi20.log 2>&1; rc=$?
echo "$(date '+%F %T') s221 gfid codi20 (2070) rc $rc $(ls boxes/runs_fsq/gsm_codi_20ep_s1/gate_fid.json 2>/dev/null)" >> $L
