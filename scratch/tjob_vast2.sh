#!/usr/bin/env bash
# Generic theory-run job, AT MOST P CONCURRENT (default 2; 16 GB cards hold 2 of the §65 runs): 
#   bash scratch/tjob_vast2.sh <tag> <device> <queue file> [P]. Lines "<out dir> <python script> <args...>".
cd "$(dirname "$0")/.."
tag=$1; dev=$2; q=$3; P=${4:-2}; log=logs/queue_$tag.out; mkdir -p logs
grep -v '^#' "$q" | grep -v '^$' | while read -r out script rest; do n=$(basename $out)
  echo "echo \"\$(date +%T) start $out\" >> $log; python -u $script $rest --device $dev --out $out > logs/tj_$n.log 2>&1; rc=\$?; ls $out/eval*.json >/dev/null 2>&1 || rc=7; echo \"\$(date +%T) evaluated $out rc \$rc\" >> $log"
done > /tmp/tj_cmds.txt
xargs -d '\n' -P $P -I{} bash -c '{}' < /tmp/tj_cmds.txt
echo "$(date +%T) QUEUE_DONE" >> $log
