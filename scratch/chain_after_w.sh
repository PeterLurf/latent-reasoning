#!/usr/bin/env bash
# Start a queue worker script once another queue's log shows QUEUE_DONE:
#   bash scratch/chain_after_w.sh <prev log> <worker script> <tag> <device> <queue>
cd "$(dirname "$0")/.."
until grep -q QUEUE_DONE "$1" 2>/dev/null; do sleep 60; done
bash "$2" "$3" "$4" "$5" >> "logs/queue_$3.out" 2>&1
