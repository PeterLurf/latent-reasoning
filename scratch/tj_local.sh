#!/usr/bin/env bash
# Generic local theory-run worker: bash scratch/tj_local.sh <device> <queue file>. Lines "<out dir> <script> <args...>";
# claims by lock dirs (<queue>.locks/<basename out>); log logs/tj_local.log, per run logs/tj_<basename>.log.
cd "$(dirname "$0")/.."
L="$2.locks"; mkdir -p "$L"
grep -v '^#' "$2" | grep -v '^$' | while read -r out script rest; do
  n=$(basename $out); mkdir "$L/$n" 2>/dev/null || continue
  python -u $script $rest --device $1 --out $out > logs/tj_$n.log 2>&1
  rc=$?; ls $out/eval*.json >/dev/null 2>&1 || rc=7
  echo "$(date +%T) $n rc $rc ($1)" >> logs/tj_local.log
done
echo "$(date +%T) WORKER_DONE $1 $2" >> logs/tj_local.log
