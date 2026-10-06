#!/usr/bin/env bash
# Post-reboot post-hoc on the 2070 (cuda:2), one at a time: §126 fp32 contractions + ANNEAL growth/CERT.
cd "$(dirname "$0")/.."
run() { n=$1; shift; python -u -m boxes.fsq120 "$@" --device cuda:2 > logs/c1_${n}.log 2>&1; echo "$(date +%T) $n rc $? (cuda:2)" >> logs/c1_local.log; }
run s126_s5_rawh_s1_contraction_fp32 --family s5 --arm final --raw_h --seed 1 --contraction_only --fp32 --out boxes/runs_fsq/s126_s5_rawh_s1
run s126_s5_rawh_s4_contraction_fp32 --family s5 --arm final --raw_h --seed 4 --contraction_only --fp32 --out boxes/runs_fsq/s126_s5_rawh_s4
A="--family s5 --arm final --anneal --seed 2 --out boxes/runs_fsq/s126_s5_anneal_s2_fp32cast"
run s126_s5_anneal_s2_fp32cast_contraction_fp32 $A --contraction_only --fp32
run s126_s5_anneal_s2_fp32cast_growth $A --growth_only --ext_n 500
run s126_s5_anneal_s2_fp32cast_cert $A --cert_only
echo "$(date +%T) POSTHOC_2070_DONE" >> logs/c1_local.log
