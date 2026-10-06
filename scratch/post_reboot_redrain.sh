#!/usr/bin/env bash
# Post-reboot 2026-10-05: restart the auto-drain watcher of every Vast host alive at ~23:00 (generated from show instances).
cd "$(dirname "$0")/.."
nohup bash scratch/auto_drain_gen5.sh "125.190.184.215 30692 gen_a2 54277416 gen" >> logs/auto_drain_gen_a2.out 2>&1 &
nohup bash scratch/auto_drain_gen5.sh "91.245.78.79 16476 gen_a4 54326779 gen" >> logs/auto_drain_gen_a4.out 2>&1 &
nohup bash scratch/auto_drain_gen5.sh "79.125.203.249 40584 gen_b27 54418137 gen" >> logs/auto_drain_gen_b27.out 2>&1 &
nohup bash scratch/auto_drain_gen5.sh "213.163.75.246 50601 gen_c2 54386763 gen" >> logs/auto_drain_gen_c2.out 2>&1 &
nohup bash scratch/auto_drain_gen5.sh "65.7.12.94 40560 gen_c3 54401709 gen" >> logs/auto_drain_gen_c3.out 2>&1 &
nohup bash scratch/auto_drain_gen5.sh "167.179.138.57 40124 gen_d1 54411699 gen" >> logs/auto_drain_gen_d1.out 2>&1 &
nohup bash scratch/auto_drain_gen5.sh "5.2.174.20 23128 gen_j1 54290090 gen" >> logs/auto_drain_gen_j1.out 2>&1 &
nohup bash scratch/auto_drain_gen5.sh "83.27.30.232 41253 gen_s4 54240473 gen" >> logs/auto_drain_gen_s4.out 2>&1 &
