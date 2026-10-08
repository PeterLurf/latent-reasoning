#!/usr/bin/env bash
# Post-reboot 2026-10-08 13:20: restart the gen7 auto-drain watcher of every Vast host alive at 13:25 (generated from show instances).
cd "$(dirname "$0")/.."
nohup bash scratch/auto_drain_gen7.sh "213.224.31.105 38958 gen_b182 54847687 gen" >> logs/auto_drain_gen_b182.out 2>&1 &
nohup bash scratch/auto_drain_gen7.sh "151.237.25.16 20410 gen_b186 54859191 gen" >> logs/auto_drain_gen_b186.out 2>&1 &
nohup bash scratch/auto_drain_gen7.sh "58.8.187.198 39597 gen_b187 54860193 gen" >> logs/auto_drain_gen_b187.out 2>&1 &
nohup bash scratch/auto_drain_gen7.sh "116.102.78.26 20881 gen_b188 54861046 gen" >> logs/auto_drain_gen_b188.out 2>&1 &
nohup bash scratch/auto_drain_gen7.sh "178.223.71.173 16479 gen_b193 54862702 gen" >> logs/auto_drain_gen_b193.out 2>&1 &
nohup bash scratch/auto_drain_gen7.sh "31.37.212.237 50650 gen_b194 54875058 gen" >> logs/auto_drain_gen_b194.out 2>&1 &
nohup bash scratch/auto_drain_gen7.sh "77.48.24.239 43152 gen_b189 54876477 gen" >> logs/auto_drain_gen_b189.out 2>&1 &
nohup bash scratch/auto_drain_gen7.sh "89.221.67.136 16612 gen_b190 54877197 gen" >> logs/auto_drain_gen_b190.out 2>&1 &
