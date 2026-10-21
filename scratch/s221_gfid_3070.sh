#!/bin/bash
# §221 G-FID, zen-gpt2 on its pinned card, the 3070 (cuda:1), after the 3070 queue r2b (Note 221-1; Audit G-D PASS dcb189ad):
# (1) the official zhenyi4/codi test.py at bs 1, fp32 (D1, D9) -> scratch/s220/zen_codi_gpt2/fid_official_bs1.json; (2) s221_gfid.py --model zen (the
# pairs[0:50] subset + assembly) -> gate_fid.json; (3) DESCRIPTIVE one-off official bf16 run (Audit) -> scratch/s220/zen_codi_gpt2/official_bf16/.
cd /d/Latent; export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
L=logs/s221_gfid.out
until grep -q "3070 queue (r2b) done" logs/s219_3070_queue.out 2>/dev/null; do sleep 60; done
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 third_party/venv_codi/Scripts/python.exe scratch/s221_official_zen.py --device cuda:1 --dtype fp32 > logs/s221_official_zen_fp32.log 2>&1; rc=$?
echo "$(date '+%F %T') s221 zen official fp32 (3070) rc $rc" >> $L
python -B scratch/s221_gfid.py --model zen --device cuda:1 > logs/s221_gfid_zen.log 2>&1; rc=$?
echo "$(date '+%F %T') s221 gfid zen (3070) rc $rc $(ls scratch/s220/zen_codi_gpt2/gate_fid.json 2>/dev/null)" >> $L
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 third_party/venv_codi/Scripts/python.exe scratch/s221_official_zen.py --device cuda:1 --dtype bf16 --out_dir scratch/s220/zen_codi_gpt2/official_bf16 > logs/s221_official_zen_bf16.log 2>&1; rc=$?
echo "$(date '+%F %T') s221 zen official bf16 DESCRIPTIVE (3070) rc $rc" >> $L
