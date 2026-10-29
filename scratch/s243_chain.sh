#!/usr/bin/env bash
# §243 DEPTH-CURRIC launch chain (Note 243-1: DC admit $1.40 / guard $2.00, CTRL admit $1.15 / guard $1.80; CAPF 8.0; EVAL_H 1.0): the four fillers ONE AT A TIME
# (each s243_fill.sh runs in the foreground until its cell is admitted, then exits); a non-zero filler rc stops the chain (HOLD / refusal -> tell Audit).
#   nohup bash scratch/s243_chain.sh >> logs/fill_s243.out 2>&1 &
cd "$(dirname "$0")/.." || exit 2
for qs in "362:dc_s1:1.40:2.00:8000:1.40" "363:dc_s2:1.40:2.00:8000:1.40" "364:dc_ctrl_s1:1.15:1.80:8000:1.15" "365:dc_ctrl_s2:1.15:1.80:8000:1.15"; do
  while ps -ef | awk '$6 ~ /bash(\.exe)?$/ && $7 ~ /s2[0-9]+_fill[a-z0-9_]*\.sh$/' | grep -q .; do sleep 30; done
  echo "$(date '+%F %T') chain: launching $qs"
  WALLW=0.1,0.1,6.0 EVAL_H=1.0 CAPF=8.0 QS="$qs" bash scratch/s243_fill.sh; rc=$?
  echo "$(date '+%F %T') chain: filler rc $rc ($qs)"
  [ $rc = 0 ] || { echo "$(date '+%F %T') chain: STOPPED (filler rc $rc)"; exit 3; }
done
echo "$(date '+%F %T') chain: ALL FOUR ADMITTED"
