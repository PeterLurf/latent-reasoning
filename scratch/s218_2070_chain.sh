#!/bin/bash
# §218 on the 2070 (cuda:2, fp32): a small probe / fd smoke, then G-L5 (untrained LangSLVA: PROBE(0), PLANT lin / xor t 0..2, FD-MIRROR reported).
# cal_n 16 (the 64-item calibration does not fit 8 GB; G-L5 tests the probe pipeline, not the calibration).
cd /d/Latent
S=scratch/_tmp/g_l5_smoke; mkdir -p $S
python -B -m boxes.langvb_eval --untrained --cal_n 16 --out_dir $S --device cuda:2 --parts probe,fd --probe_ts 0,2 --plant_ts 0,1 --probe_steps 3 --pool_n 128 --fd_items 128 --per_D 8 --eval_bs 8 > $S/run.log 2>&1; rc=$?
echo "$(date '+%F %T') l5_smoke rc $rc" >> logs/s218_gates.out
[ $rc -ne 0 ] && exit 1
G=scratch/gate_s218/g_l5; mkdir -p $G
python -B -m boxes.langvb_eval --untrained --cal_n 16 --out_dir $G --device cuda:2 --parts probe,fd --probe_ts 0 --plant_ts 0,1,2 --eval_bs 16 > $G/run.log 2>&1; rc=$?
echo "$(date '+%F %T') l5 rc $rc" >> logs/s218_gates.out
