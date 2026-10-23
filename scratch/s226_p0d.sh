#!/bin/bash
# §226 P0d-FULLFT ($0, local; Note 226-1 + Audit's ruling; §226's last cell): P0b-TOPO's flags with ONLY --full_ft and --lr 1e-4 changed - DIRECT,
# --task ow --order_mode topo, cg_thr 0.45, cg_cap 6000, steps 48000 (the schedule), stop_at 12000 - on the RTX 4060 (cuda:0, the registered kernel),
# --auto_micro --mem_budget 7.0, full-state ckpt every 2,000 (a dir with ckpt_last.pt resumes with --resume) -> scratch/_tmp/s226_p0d/P0d-FULLFT +
# cell_meta.json; then scratch/s226_pilot_summary.py --p0d -> scratch/gate_s226/pilot_p0d.json.   nohup bash scratch/s226_p0d.sh >> logs/s222_chain.out 2>&1 &
cd /d/Latent; export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
grep -q '"--full_ft"' boxes/codi_ps.py || { echo "$(date '+%F %T') s226 P0d REFUSED: the --full_ft build is not committed"; exit 2; }
DV=cuda:0; O=scratch/_tmp/s226_p0d/P0d-FULLFT; L=logs/s222_chain.out; RES=""
if [ -e "$O/ckpt_last.pt" ]; then RES="--resume"; else [ -e "$O" ] && mv "$O" "$O.void_$(date +%H%M%S)"; fi
mkdir -p $O; echo "$(date '+%F %T') s226 P0d start ($DV, kernel registered${RES:+, resume})" >> $L
python -B boxes/codi_ps.py --direct --task ow --order_mode topo --full_ft --lr 1e-4 --cg_thr 0.45 --cg_cap 6000 --steps 48000 --stop_at 12000 --device $DV --out $O --auto_micro --mem_budget 7.0 $RES >> logs/s226_p0d.log 2>&1; rc=$?
printf '{"card_device": "%s", "card": "RTX 4060", "kernel": "registered", "rc": %s}\n' "$DV" "$rc" > $O/cell_meta.json
echo "$(date '+%F %T') s226 P0d rc $rc (9 = stopped at 12000 as registered)" >> $L
python -B scratch/s226_pilot_summary.py --p0d >> logs/s226_pilot_summary.log 2>&1; echo "$(date '+%F %T') s226 P0d summary rc $? -> scratch/gate_s226/pilot_p0d.json" >> $L
