#!/bin/bash
# §227 TEXT-SERIAL ($0, local; registered f69fd658): one cell TEXT on the RTX 4060 (cuda:0, the registered kernel): --task ow --order_mode topo --direct
# --full_ft --lr 1e-4 --text_cot, cg_thr 0.45, cg_cap 6000, steps 48000 (the schedule), stop_at 18000, --auto_micro --mem_budget 7.0; full-state ckpt every
# 2,000 (a dir with ckpt_last.pt resumes with --resume); free decoding every 1,000 and at stop (decode_evals.jsonl) -> scratch/_tmp/s227_text/TEXT +
# cell_meta.json; then scratch/s227_pilot_summary.py -> scratch/gate_s227/pilot_text.json.   nohup bash scratch/s227_text.sh >> logs/s222_chain.out 2>&1 &
cd /d/Latent; export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
grep -q '"--text_cot"' boxes/codi_ps.py || { echo "$(date '+%F %T') s227 REFUSED: the --text_cot build is not committed"; exit 2; }
DV=cuda:0; O=scratch/_tmp/s227_text/TEXT; L=logs/s222_chain.out; RES=""
# 10-23: --resume skips auto_micro, so the probed micro (mem_check.json) is passed explicitly
if [ -e "$O/ckpt_last.pt" ]; then RES="--resume --micro $(python -c "import json; print(json.load(open('$O/mem_check.json'))['micro'])")"
else [ -e "$O" ] && mv "$O" "$O.void_$(date +%H%M%S)"; fi
mkdir -p $O; echo "$(date '+%F %T') s227 TEXT start ($DV, kernel registered${RES:+, resume})" >> $L
python -B boxes/codi_ps.py --direct --task ow --order_mode topo --full_ft --lr 1e-4 --text_cot --cg_thr 0.45 --cg_cap 6000 --steps 48000 --stop_at 18000 --device $DV --out $O --auto_micro --mem_budget 7.0 $RES >> logs/s227_text.log 2>&1; rc=$?
printf '{"card_device": "%s", "card": "RTX 4060", "kernel": "registered", "rc": %s}\n' "$DV" "$rc" > $O/cell_meta.json
echo "$(date '+%F %T') s227 TEXT rc $rc (9 = stopped at 18000 as registered)" >> $L
[ $rc = 9 ] || { echo "$(date '+%F %T') s227 NO SUMMARY: rc $rc (the run did not reach stop_at 18000); tell Theory / Audit" >> $L; exit 3; }   # 10-23: never summarise a crashed run
python -B scratch/s227_pilot_summary.py >> logs/s227_pilot_summary.log 2>&1; echo "$(date '+%F %T') s227 summary rc $? -> scratch/gate_s227/pilot_text.json" >> $L
