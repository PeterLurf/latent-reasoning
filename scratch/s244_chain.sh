#!/usr/bin/env bash
# §244 ANSWER-ONLY launch chain (caps per Audit G-D: admit $0.70 / guard $1.10 per arm; CAPF 3.5; EVAL_H 0.4): the given cells ONE AT A TIME, each "<QS>|<WALLW>"
# (b366 has 4 queue lines: the host replay first). A non-zero filler rc stops the chain.
#   nohup bash scratch/s244_chain.sh "366:ao_s1:0.70:1.10:12000:0.70|0.05,0.1,0.1,2.5" >> logs/fill_s244.out 2>&1 &
#   nohup bash scratch/s244_chain.sh "367:ao_s2:0.70:1.10:12000:0.70|0.1,0.1,2.5" "368:aux_s2:0.70:1.10:12000:0.70|0.1,0.1,2.5" >> logs/fill_s244.out 2>&1 &
cd "$(dirname "$0")/.." || exit 2
for c in "$@"; do
  qs=${c%%|*}; w=${c##*|}
  while ps -ef | awk '$6 ~ /bash(\.exe)?$/ && $7 ~ /s2[0-9]+_fill[a-z0-9_]*\.sh$/' | grep -q .; do sleep 30; done
  echo "$(date '+%F %T') chain: launching $qs (WALLW $w)"
  WALLW=$w EVAL_H=0.4 CAPF=3.5 QS="$qs" bash scratch/s244_fill.sh; rc=$?
  echo "$(date '+%F %T') chain: filler rc $rc ($qs)"
  [ $rc = 0 ] || { echo "$(date '+%F %T') chain: STOPPED (filler rc $rc)"; exit 3; }
done
echo "$(date '+%F %T') chain: DONE $*"
