#!/bin/bash
# §223 Note 223-4 pilot, optional M1-only cell + a DIRECT pilot on the SAME 4060 (Theory 10-21: M1's LP3 compared on one card). Frozen flags exactly
# (scratch/s223_args_frozen_{m1,direct}.json: --recode --lam 1 | --direct; every other trainer arg at its default, --steps 24000 kept), --stop_at 4600,
# --auto_micro --mem_budget 7.0, registered SDP kernel (Ampere/Ada) -> scratch/_tmp/s223_pilot/<cell>_4060 + cell_meta.json; then
# scratch/s223_pilot_summary.py adds both cells to scratch/gate_s223/pilot.json with their card.   nohup bash scratch/s223_pilot_4060.sh >> logs/s222_chain.out 2>&1 &
cd /d/Latent; export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
DV=cuda:0; T=scratch/_tmp/s223_pilot; mkdir -p "$T"; L=logs/s222_chain.out
for cell in m1 direct; do
  case $cell in direct) F="--direct";; m1) F="--recode --lam 1";; esac
  O=$T/${cell}_4060; [ -e "$O" ] && mv "$O" "$O.void_$(date +%H%M%S)"
  echo "$(date '+%F %T') s223 pilot ${cell}_4060 start ($DV, kernel registered)" >> $L
  python -B boxes/codi_ps.py $F --device $DV --out $O --stop_at 4600 --auto_micro --mem_budget 7.0 > logs/s223_pilot_${cell}_4060.log 2>&1; rc=$?
  printf '{"card_device": "%s", "card": "RTX 4060", "kernel": "registered", "rc": %s}\n' "$DV" "$rc" > $O/cell_meta.json
  echo "$(date '+%F %T') s223 pilot ${cell}_4060 rc $rc (9 = stopped at 4600 as registered)" >> $L
done
python -B scratch/s223_pilot_summary.py cuda:2 math >> logs/s223_pilot_summary.log 2>&1; echo "$(date '+%F %T') s223 pilot summary (4060 cells) rc $? -> scratch/gate_s223/pilot.json" >> $L
