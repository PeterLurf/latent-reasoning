#!/usr/bin/env bash
# THEORY §58 (F6) queue worker: bash scratch/f6_queue.sh <device> [amp]. 30 runs = 3 arms x 10 seeds (P30 recipe exactly,
# transplants after the gated curriculum reaches full depth), interleaved by seed; claimed by lock dirs; resumable.
cd "$(dirname "$0")/.."
AMP=${2:-bf16}
mkdir -p theory/runs/f6_locks
for s in 102 111 113 118 125 130 132 140 146 157; do for arm in aged young mismatch; do
  tag=f6${arm:0:1}_s$s; out=theory/runs/$tag
  [ -f $out/model.pt ] && [ -f $out/eval_done ] && continue
  mkdir theory/runs/f6_locks/$tag 2>/dev/null || continue
  python -u theory/train_arch.py --out $out --seed $s --K 6 --steps 30000 --bs 512 --Q 16 --lr 5e-4 --warmup 300 \
    --curriculum gated --gate 0.9 --init gpt2 --aux_steps 10000 --aux_anneal 5000 --children_weight 1 --no_step_emb \
    --save_every 1000 --amp $AMP --resume --transplant 1 --transplant_mode $arm --transplant_ages 2:4 --transplant_bs 128 \
    --device $1 >> logs/f6_$tag.log 2>&1
  rc=$?
  [ $rc -eq 0 ] && python -u theory/eval_arch.py $out --device $1 >> logs/f6_$tag.log 2>&1 && touch $out/eval_done
  echo "$(date +%T) $tag train rc $rc ($1)" >> logs/f6_queue.log
done; done
echo "$(date +%T) WORKER_DONE $1" >> logs/f6_queue.log
