#!/bin/bash
# §223 Note 223-4 LEARNING PILOT ($0, local; descriptive gate, no registered reading): DIRECT then PS, SEQUENTIALLY on the SAME card / kernel
# (Audit 10-21), each with its frozen flags exactly (scratch/s223_args_frozen_{direct,ps}.json: --direct | --recode --lam 1 --bottleneck; every other
# trainer arg at its default, --steps 24000 kept so the LR schedule is the registered one) + --out scratch/_tmp/s223_pilot/<cell> + --stop_at 4600
# (Note 223-4: stage 1's threshold advance lands at 2,000, stage 2's decision at 4,000 / 4,500) + --auto_micro --mem_budget 7.0 (META: the micro on
# an 8 GB card; no training quantity changes). On a Turing card (the 2070) the math SDP kernel is forced (scratch/s223_mathsdp_wrap.py); on the
# 4060 / 3070 the registered kernel runs. -> logs/s223_pilot_<cell>.log, then scratch/s223_pilot_summary.py -> scratch/gate_s223/pilot.json.
#   bash scratch/s223_pilot.sh <cuda dev> [math]
cd /d/Latent; export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
DV=${1:-cuda:2}; KER=${2:-}; T=scratch/_tmp/s223_pilot; mkdir -p "$T"; L=logs/s222_chain.out
PY="python -B"; [ "$KER" = math ] && PY="python -B scratch/s223_mathsdp_wrap.py"
for cell in direct ps; do
  case $cell in direct) F="--direct";; ps) F="--recode --lam 1 --bottleneck";; esac
  [ -e "$T/$cell" ] && mv "$T/$cell" "$T/$cell.void_$(date +%H%M%S)"
  echo "$(date '+%F %T') s223 pilot $cell start ($DV, kernel ${KER:-registered})" >> $L
  $PY boxes/codi_ps.py $F --device $DV --out $T/$cell --stop_at 4600 --auto_micro --mem_budget 7.0 > logs/s223_pilot_$cell.log 2>&1; rc=$?
  echo "$(date '+%F %T') s223 pilot $cell rc $rc (9 = stopped at 4600 as registered)" >> $L
done
python -B scratch/s223_pilot_summary.py "$DV" "${KER:-registered}" >> logs/s223_pilot_summary.log 2>&1; echo "$(date '+%F %T') s223 pilot summary rc $? -> scratch/gate_s223/pilot.json" >> $L
