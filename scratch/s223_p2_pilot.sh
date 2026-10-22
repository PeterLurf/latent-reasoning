#!/bin/bash
# §223 Note 223-8 pilot P2 ($0, local; Audit amendment 11a4b858; Theory 10-22): DIRECT, M1-only and PS (+ --m2_stage) SEQUENTIALLY on the RTX 4060
# (cuda:0, the registered kernel), each with the frozen flags except R1 lr 3e-4 and R2 cg_thr 0.45 (+ R3 --m2_stage on PS), --steps 24000 (the LR
# schedule), --stop_at 7600, --auto_micro --mem_budget 7.0 (the 8 GB card's micro; no training quantity changes) -> scratch/_tmp/s223_p2/<cell>
# + cell_meta.json, then scratch/s223_pilot_summary.py --p2 -> scratch/gate_s223/pilot2.json (LP1-LP3 read by Theory; PS after the ramp).
# Runs only on the committed (G-D'd) boxes/codi_ps.py with --m2_stage.   nohup bash scratch/s223_p2_pilot.sh >> logs/s222_chain.out 2>&1 &
cd /d/Latent; export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
grep -q "def soft_decoder" boxes/codi_ps.py || { echo "$(date '+%F %T') s223 P2 REFUSED: boxes/codi_ps.py has no --m2_stage (not committed)"; exit 2; }
DV=cuda:0; T=scratch/_tmp/s223_p2; mkdir -p "$T"; L=logs/s222_chain.out
for cell in direct m1 ps; do
  case $cell in direct) F="--direct";; m1) F="--recode --lam 1";; ps) F="--recode --lam 1 --bottleneck --m2_stage";; esac
  O=$T/$cell; [ -e "$O" ] && mv "$O" "$O.void_$(date +%H%M%S)"
  echo "$(date '+%F %T') s223 P2 $cell start ($DV, kernel registered)" >> $L
  python -B boxes/codi_ps.py $F --lr 3e-4 --cg_thr 0.45 --device $DV --out $O --stop_at 7600 --auto_micro --mem_budget 7.0 > logs/s223_p2_$cell.log 2>&1; rc=$?
  printf '{"card_device": "%s", "card": "RTX 4060", "kernel": "registered", "rc": %s}\n' "$DV" "$rc" > $O/cell_meta.json
  echo "$(date '+%F %T') s223 P2 $cell rc $rc (9 = stopped at 7600 as registered)" >> $L
done
python -B scratch/s223_pilot_summary.py cuda:0 registered --p2 >> logs/s223_pilot_summary.log 2>&1; echo "$(date '+%F %T') s223 P2 summary rc $? -> scratch/gate_s223/pilot2.json" >> $L
