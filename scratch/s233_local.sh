#!/bin/bash
# §233 $0 local gates on the RTX 4060 (cuda:0), one at a time: P1-ASK+NAME, then CD-local (ST_v1 §2). Wrappers committed before launch.
cd /d/Latent; export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2; L=logs/s233_local.out; D=boxes/runs_fsq
echo "$(date '+%F %T') P1-ASK+NAME start" >> $L
python -B docs/research_steptold_2026-10-24/st_ask_name_local.py --d_max 2 --steps 8000 --auto_micro --mem_budget 7.0 --ckpt_every 2000 --eval_parts acc --device cuda:0 --out $D/s233_askname_s1 > logs/s233_askname_s1.log 2>&1
rc=$?; echo "$(date '+%F %T') P1-ASK+NAME rc $rc summary $( [ -s scratch/gate_s233/p1_ask_name.json ] && echo present || echo MISSING)" >> $L
echo "$(date '+%F %T') CD-local start" >> $L
python -B docs/research_steptold_2026-10-24/st_cd_local.py --d_max 2 --steps 8000 --auto_micro --mem_budget 7.0 --ckpt_every 2000 --eval_parts acc --device cuda:0 --out $D/s233_cdloc_s1 > logs/s233_cdloc_s1.log 2>&1
rc=$?; echo "$(date '+%F %T') CD-local rc $rc summary $( [ -s scratch/gate_s233/cd_local.json ] && echo present || echo MISSING)" >> $L
echo "$(date '+%F %T') s233 local done" >> $L
