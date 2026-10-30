#!/usr/bin/env bash
# §251 ANSWER-EXTEND launch chain (caps: admit $1.05 / guard $1.12 per arm; CAPF 4.5 (env CAPF overrides; Note 251-2 rerun: 5.5); EVAL_H 0.5): the cells ONE AT A TIME, each "<QS>|<WALLW>"
#   nohup bash scratch/s251_chain.sh "376:aex_s1:1.05:1.12:16000:1.05|0.1,0.1,4.0" "377:aex_s2:1.05:1.12:16000:1.05|0.1,0.1,4.0" "378:aex0_s1:1.05:1.12:16000:1.05|0.1,0.1,4.0" "379:aex0_s2:1.05:1.12:16000:1.05|0.1,0.1,4.0" >> logs/fill_s251.out 2>&1 &
cd "$(dirname "$0")/.." || exit 2
for c in "$@"; do
  qs=${c%%|*}; w=${c##*|}
  while ps -ef | awk '$6 ~ /bash(\.exe)?$/ && $7 ~ /s2[0-9]+_fill[a-z0-9_]*\.sh$/' | grep -q .; do sleep 30; done
  echo "$(date '+%F %T') chain: launching $qs (WALLW $w)"
  WALLW=$w EVAL_H=0.5 CAPF=${CAPF:-4.5} QS="$qs" bash scratch/s251_fill.sh; rc=$?
  echo "$(date '+%F %T') chain: filler rc $rc ($qs)"
  [ $rc = 0 ] || { echo "$(date '+%F %T') chain: STOPPED (filler rc $rc)"; exit 3; }
done
echo "$(date '+%F %T') chain: DONE $*"
