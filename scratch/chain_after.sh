#!/usr/bin/env bash
# Start a queue worker once another queue's log shows QUEUE_DONE:  bash scratch/chain_after.sh <prev log> <tag> <device> <queue>
cd "$(dirname "$0")/.."
until grep -q QUEUE_DONE "$1" 2>/dev/null; do sleep 60; done
bash scratch/queue_worker2.sh "$2" "$3" "$4" >> "logs/queue_$2.out" 2>&1
