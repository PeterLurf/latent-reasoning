#!/usr/bin/env bash
# THEORY §82 acquisition ladder (21da4fc): 32 free-table runs on one card, P 3 (Theory's commands).
cd "$(dirname "$0")/.."
d=${1:-cuda:2}; touch scratch/.hold76_$d
python theory/c1_tab_driver.py --groups Z59 Z60 D30 A4xZ5 --curricula ge3 --seeds 2111 2112 2113 2114 2115 --no_extras --device $d --P 3 --out theory/results/c1_tab82 > logs/s82_ge3.log 2>&1
python theory/c1_tab_driver.py --groups Z59 Z60 D30 A4xZ5 --curricula std --seeds 2111 2112 2113 --no_extras --device $d --P 3 --out theory/results/c1_tab82 > logs/s82_std.log 2>&1
echo "$(date +%T) s82 ladder done ($d): $(ls theory/results/c1_tab82/*.json 2>/dev/null | wc -l) files" >> logs/c1_local.log
rm -f scratch/.hold76_$d
