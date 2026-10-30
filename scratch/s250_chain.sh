#!/usr/bin/env bash
# §250 FRAME-DIV Stage 1 launch chain (caps: admit $1.00 / guard $1.20 per arm; CAPF 5.0; EVAL_H 1.0): the cells ONE AT A TIME, each "<QS>|<WALLW>"
#   nohup bash scratch/s250_chain.sh "372:fd_div_s1:1.00:1.20:8000:1.00|0.1,0.1,3.0" "373:fd_div_s2:1.00:1.20:8000:1.00|0.1,0.1,3.0" "374:fd_ctrl_s1:1.00:1.20:8000:1.00|0.1,0.1,3.0" "375:fd_ctrl_s2:1.00:1.20:8000:1.00|0.1,0.1,3.0" >> logs/fill_s250.out 2>&1 &
cd "$(dirname "$0")/.." || exit 2
for c in "$@"; do
  qs=${c%%|*}; w=${c##*|}
  while ps -ef | awk '$6 ~ /bash(\.exe)?$/ && $7 ~ /s2[0-9]+_fill[a-z0-9_]*\.sh$/' | grep -q .; do sleep 30; done
  echo "$(date '+%F %T') chain: launching $qs (WALLW $w)"
  WALLW=$w EVAL_H=1.0 CAPF=5.0 QS="$qs" bash scratch/s250_fill.sh; rc=$?
  echo "$(date '+%F %T') chain: filler rc $rc ($qs)"
  [ $rc = 0 ] || { echo "$(date '+%F %T') chain: STOPPED (filler rc $rc)"; exit 3; }
done
echo "$(date '+%F %T') chain: DONE $*"
