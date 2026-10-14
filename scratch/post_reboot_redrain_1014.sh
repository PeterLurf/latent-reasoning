#!/usr/bin/env bash
# Post-reboot 2026-10-14 13:45: restart the gen8 auto-drain watchers of the finished §200 A / B hosts (QUEUE_DONE at 20:13 / 18:52 host time).
cd "$(dirname "$0")/.."
nohup bash scratch/auto_drain_gen8.sh "218.150.198.117 51977 gen_b287 55890105 gen" >> logs/auto_drain_gen_b287.out 2>&1 &
nohup bash scratch/auto_drain_gen8.sh "209.226.246.174 53029 gen_b288 55895675 gen" >> logs/auto_drain_gen_b288.out 2>&1 &
