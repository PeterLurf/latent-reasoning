#!/usr/bin/env bash
# §83 remainder, local (Theory, option (a)): primaries s2, s3 (5794e4c script), then extras s1-s3.
# Outputs theory/results/s83/wc0k4r_s<seed>.json and _ext.json.
# v2 (2026-10-03 16:30): rc captured right after python (v1 read $? after $(date), so every crash logged rc 0) and
# "no-output" appended when the .json is missing; --fp32 on Turing (the 2070, CC 7.5, has no bf16 SDPA kernel:
# v1's five runs crashed in 4 s with "cutlassF: no kernel found to launch").
cd "$(dirname "$0")/.."
d=${1:-cuda:2}; F=""
python -c "import torch,sys; sys.exit(0 if torch.cuda.get_device_capability('$d')[0] >= 8 else 1)" || F="--fp32 --bs 64"   # fp32 at bs 250 OOMs 8 GB (KV cache)
run() { out=$1; log=$2; shift 2
  python -u theory/s83_restart.py "$@" $F --device $d --out $out > $log 2>&1; rc=$?; [ -s $out ] || rc="$rc no-output"
  echo "$(date +%T) s83 $(basename $out .json) rc $rc ($d ${F:-bf16-as-trained})" >> logs/c1_local.log; }
for s in 2 3; do run theory/results/s83/wc0k4r_s$s.json logs/s83_local_primary_s$s.log --run boxes/runs_s5/wc0k4r_s$s --k 4 --kmax 16 --test_n 2000 --ext_n 0 --cal_n 20000 --sweep_n 0 --fixed 4 16; done
for s in 1 2 3; do run theory/results/s83/wc0k4r_s${s}_ext.json logs/s83_local_ext_s$s.log --run boxes/runs_s5/wc0k4r_s$s --k 4 --kmax 16 --test_n 2000 --ext_n 500 --cal_n 20000 --sweep_n 500 --fixed 4 8 16; done
echo "$(date +%T) S83_LOCAL_DONE $d" >> logs/c1_local.log
