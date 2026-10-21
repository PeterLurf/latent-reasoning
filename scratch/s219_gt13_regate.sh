#!/bin/bash
# §219 after applying the Audit-approved G-T13 trainer patch (cde2c604) to boxes/textself_ft.py: re-run the CPU gates against boxes/ (no S219_TRAINER)
# and a plain-path smoke (construct + 1 step, no self flags). CPU only, Idle-ish (nice), 2 threads (gate_s219 default); t4 deterministic (Note 219-3).
cd /d/Latent; export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2; unset S219_TRAINER
L=logs/s219_gt13_regate.out
g() { nice -n 19 python -B scratch/gate_s219.py "$@" > logs/s219_regate_$1.log 2>&1; local rc=$?; echo "$(date '+%F %T') regate $* rc $rc $(tail -n 1 logs/s219_regate_$1.log | cut -c1-120)" >> $L; }
for x in static t0 t3 t8 t7 t9t10; do g $x; done
S219_DET=1 g t4
g t5
g e2e
D=scratch/_tmp/s219/plain1; rm -rf $D
nice -n 19 python -B -m boxes.textself_ft --arm textcot --steps 12000 --lr 3e-05 --bs 8 --device cpu --p_self 0 --stop_at 1 --no_final_eval --out $D > logs/s219_regate_plain1.log 2>&1; rc=$?
echo "$(date '+%F %T') plain-path smoke rc $rc steps $(grep -c '"loss"' $D/log.jsonl 2>/dev/null) eval_files $(ls $D/eval*.json 2>/dev/null | wc -l)" >> $L
find scratch/_tmp/s219 -name "*.pt" -delete
echo "$(date '+%F %T') §219 G-T13 regate done" >> $L
