#!/bin/bash
# §226 ORDERED-WORDS stage 0 pilots ($0, local; registered 09368b14): P0-OW (forward, mod 11; binding), P0b-TOPO (--order_mode topo), P0c-CHAIN (--dag off),
# SEQUENTIALLY on the RTX 4060 (cuda:0, the registered kernel), each with the §223 P3a recipe exactly (DIRECT; lr 3e-4, cg_thr 0.45, cg_cap 6000, steps
# 48000 = the schedule, stop_at 12000; --auto_micro --mem_budget 7.0) + --task ow (+ the cell flag) -> scratch/_tmp/s226_p0/<cell> (full-state ckpt every
# 2,000; a cell with ckpt_last.pt resumes with --resume) + cell_meta.json; then scratch/s226_pilot_summary.py -> scratch/gate_s226/pilot_ow.json.
# Runs only on the committed (G-D'd) boxes/codi_ps.py with --task ow.   nohup bash scratch/s226_p0.sh >> logs/s222_chain.out 2>&1 &
cd /d/Latent; export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
grep -q '"--task"' boxes/codi_ps.py && [ -f boxes/ow_words.py ] || { echo "$(date '+%F %T') s226 P0 REFUSED: the §226 build is not committed"; exit 2; }
DV=cuda:0; T=scratch/_tmp/s226_p0; mkdir -p $T; L=logs/s222_chain.out
for cell in P0-OW P0b-TOPO P0c-CHAIN; do
  case $cell in P0-OW) F="--task ow";; P0b-TOPO) F="--task ow --order_mode topo";; P0c-CHAIN) F="--task ow --dag off";; esac
  O=$T/$cell; RES=""
  if [ -e "$O/ckpt_last.pt" ]; then RES="--resume"; elif [ -e "$O/train_done.json" ] || grep -q "s226 P0 $cell rc 9" $L; then echo "$(date '+%F %T') s226 P0 $cell already done - next" >> $L; continue
  else [ -e "$O" ] && mv "$O" "$O.void_$(date +%H%M%S)"; fi
  mkdir -p $O; echo "$(date '+%F %T') s226 P0 $cell start ($DV, kernel registered${RES:+, resume})" >> $L
  python -B boxes/codi_ps.py --direct --lr 3e-4 --cg_thr 0.45 --cg_cap 6000 --steps 48000 --stop_at 12000 $F --device $DV --out $O --auto_micro --mem_budget 7.0 $RES >> logs/s226_p0_$cell.log 2>&1; rc=$?
  printf '{"card_device": "%s", "card": "RTX 4060", "kernel": "registered", "rc": %s}\n' "$DV" "$rc" > $O/cell_meta.json
  echo "$(date '+%F %T') s226 P0 $cell rc $rc (9 = stopped at 12000 as registered)" >> $L
done
python -B scratch/s226_pilot_summary.py >> logs/s226_pilot_summary.log 2>&1; echo "$(date '+%F %T') s226 P0 summary rc $? -> scratch/gate_s226/pilot_ow.json" >> $L
