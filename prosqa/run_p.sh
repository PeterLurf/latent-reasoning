#!/usr/bin/env bash
# PROTOCOL6 addendum P: evaluate checkpoints on the long and control sets. bash prosqa/run_p.sh <device> <tag>...
cd "$(dirname "$0")"
DEV=$1; shift
mkdir -p res_p
for t in "$@"; do
  case $t in g_*) pre="";; j_*) pre="_d";; esac
  for set in long ctl; do
    o=res_p/${t}_${set}.json
    [ -s $o ] || python long_eval.py --ckpt ckpt/${t}_checkpoint_50 --data data/prosqa${pre}_${set}_test.json --out $o --device $DEV || echo "FAIL $o"
  done
done
echo P_EVAL_DONE "$@"
