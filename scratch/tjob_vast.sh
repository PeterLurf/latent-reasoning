#!/usr/bin/env bash
# Generic theory-run job on a Vast host (parallel on one GPU): bash scratch/tjob_vast.sh <tag> <device> <queue file>.
# Queue lines: "<out dir> <python script> <args...>" -> python -u <script> <args> --device <device> --out <out dir>.
# A run counts as done only if <out dir> holds a non-empty eval*.json. Ends with QUEUE_DONE in logs/queue_<tag>.out.
cd "$(dirname "$0")/.."
tag=$1; dev=$2; q=$3; log=logs/queue_$tag.out; mkdir -p logs
run() {
  out=$1; shift; n=$(basename $out); echo "$(date +%T) start $out" >> $log
  python -u "$@" --device $dev --out $out > logs/tj_$n.log 2>&1
  rc=$?; ls $out/eval*.json >/dev/null 2>&1 || rc=7
  echo "$(date +%T) evaluated $out rc $rc" >> $log
}
while read -r line; do case "$line" in \#*|"") continue;; esac; run $line & done < "$q"
wait
echo "$(date +%T) QUEUE_DONE" >> $log
