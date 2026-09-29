#!/usr/bin/env bash
# THEORY §24 Pilot G feasibility run requested by the theory session (2026-09-29 ~14:35), code at commit 0aecb47.
cd "$(dirname "$0")/.."
python scratch/stage1_guard.py theory/runs/fz_gat_s951 0 3 --wait -- theory/train_arch.py --mode coconut --bs 512 --steps 30000 --warmup 300 --eval_every 1000 --aux_steps 10000 --aux_anneal 5000 --children_weight 1 --curriculum gated --init gpt2 --no_step_emb --K 6 --lr 5e-4 --amp bf16 --save_every 1000 --device cuda:1 --resume --out theory/runs/fz_gat_s951 --seed 951 --arch setmem --anytime 1 >> logs/fz_gat_s951.out 2>> logs/fz_gat_s951.err
echo "$(date +%T) trained fz_gat_s951 rc $?"
[ -f theory/runs/fz_gat_s951/model.pt ] && python theory/eval_arch.py theory/runs/fz_gat_s951 --device cuda:1 2>> logs/eval_fz_gat_s951.err
echo "$(date +%T) evaluated fz_gat_s951"
