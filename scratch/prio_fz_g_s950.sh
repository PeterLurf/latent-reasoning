#!/usr/bin/env bash
# THEORY §24 Pilot G feasibility run requested by the theory session (2026-09-29 ~14:35), code at commit 0aecb47.
cd "$(dirname "$0")/.."
python scratch/stage1_guard.py theory/runs/fz_g_s950 0 3 --wait -- theory/train_arch.py --mode coconut --bs 512 --steps 30000 --warmup 300 --eval_every 1000 --aux_steps 10000 --aux_anneal 5000 --children_weight 1 --curriculum gated --init gpt2 --no_step_emb --K 6 --lr 5e-4 --amp bf16 --save_every 1000 --device cuda:0 --resume --out theory/runs/fz_g_s950 --seed 950 --arch setmem >> logs/fz_g_s950.out 2>> logs/fz_g_s950.err
echo "$(date +%T) trained fz_g_s950 rc $?"
[ -f theory/runs/fz_g_s950/model.pt ] && python theory/eval_arch.py theory/runs/fz_g_s950 --device cuda:0 2>> logs/eval_fz_g_s950.err
echo "$(date +%T) evaluated fz_g_s950"
