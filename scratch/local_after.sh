#!/usr/bin/env bash
# Overnight: wait until no python job uses <device> on this PC, then run the next local work on it, one job at a time
# (never two GPT-2 jobs per 8 GB card): the backfill queue (lock-based, shared by all cards).
#   bash scratch/local_after.sh <device>
cd "$(dirname "$0")/.."
dev=$1
busy() { powershell -NoProfile -Command "if ((Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match '--device $dev( |\$)' }).Count -gt 0) { exit 0 } else { exit 1 }"; }
while busy; do sleep 60; done
echo "$(date +%T) $dev free -> backfill" >> logs/local_after.log
bash scratch/bench_worker.sh $dev scratch/backfill_q1.txt
echo "$(date +%T) $dev backfill done" >> logs/local_after.log
