#!/bin/bash
# §218 on the 4060 (cuda:0) after scratch/s217_runs.sh (MSYS pid 39936): G-L1c OVERFIT-256 (Note 218-8), then G-EVMEM (untrained LangSLVA, shapes only;
# -> boxes/runs_fsq/lvb_gates/gate_evmem.json + scratch/gate_s218/g_evmem/eval_lvb.json).
cd /d/Latent
while kill -0 39936 2>/dev/null; do sleep 60; done
python -B scratch/gate_s218_l1c.py --device cuda:0 > scratch/gate_s218/g_l1c.log 2>&1; rc=$?
echo "$(date '+%F %T') l1c rc $rc" >> logs/s218_gates.out
G=scratch/gate_s218/g_evmem; mkdir -p $G
python -B -m boxes.langvb_eval --untrained --cal_n 16 --out_dir $G --device cuda:0 --parts evmem > $G/run.log 2>&1; rc=$?
echo "$(date '+%F %T') evmem rc $rc" >> logs/s218_gates.out
