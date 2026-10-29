#!/usr/bin/env bash
# §245 AO-REP launch chain (caps per Audit G-D: admit $0.95 / guard $1.20 per arm; CAPF 2.5; EVAL_H 0.4): the given cells ONE AT A TIME, each "<QS>|<WALLW>"
#   nohup bash scratch/s245_chain.sh "370:ao_s3:0.95:1.20:18000:0.95|0.1,0.1,3.0" "371:ao_s4:0.95:1.20:18000:0.95|0.1,0.1,3.0" >> logs/fill_s245.out 2>&1 &
cd "$(dirname "$0")/.." || exit 2
for c in "$@"; do
  qs=${c%%|*}; w=${c##*|}
  while ps -ef | awk '$6 ~ /bash(\.exe)?$/ && $7 ~ /s2[0-9]+_fill[a-z0-9_]*\.sh$/' | grep -q .; do sleep 30; done
  echo "$(date '+%F %T') chain: launching $qs (WALLW $w)"
  WALLW=$w EVAL_H=0.4 CAPF=2.5 QS="$qs" bash scratch/s245_fill.sh; rc=$?
  echo "$(date '+%F %T') chain: filler rc $rc ($qs)"
  [ $rc = 0 ] || { echo "$(date '+%F %T') chain: STOPPED (filler rc $rc)"; exit 3; }
done
echo "$(date '+%F %T') chain: DONE $*"
