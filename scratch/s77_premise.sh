#!/usr/bin/env bash
# THEORY §77 premise (descriptive, pilot seeds 990-992): one-hot bilinear toy at readout width H on one card.
# Marks the card busy for feed76 via scratch/.w76_<dev>. bash scratch/s77_premise.sh <dev> <H> [group]
cd "$(dirname "$0")/.."
d=$1; H=$2; G=${3:-S4}
echo $$ > scratch/.w76_$d
python theory/c1_group.py --group $G --ratio 2 --tau0 4 --updates 16000 --lr 1e-2 --hid $H --seeds 990 991 992 --device $d --out theory/results/s77_premise_${G}_hid$H.json > logs/s77_premise_${G}_hid$H.log 2>&1
echo "$(date +%T) s77 premise $G hid $H rc $? ($d)" >> logs/c1_local.log
