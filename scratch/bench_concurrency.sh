#!/usr/bin/env bash
# Throughput of N simultaneous training runs on one GPU (cloud capacity planning, not a scored run).
#   bash scratch/bench_concurrency.sh "1 2 4 6" [extra train args]
# Each copy trains the seed-0 main recipe (K=6) for 1200 steps; ms/step is read from log.jsonl
# between steps 400 and 1200 (after CUDA-graph capture and warm-up).
set -u
NS=${1:-"1 2 4 6"}; shift || true
mkdir -p bench
for N in $NS; do
  pids=()
  for i in $(seq 1 "$N"); do
    out=bench/n${N}_$i; rm -rf "$out"
    python -m latent.train --mode coconut --K 6 --bs 512 --steps 1200 --warmup 300 --eval_every 100000 \
      --aux_steps 10000 --aux_anneal 5000 --children_weight 1 --curriculum gated --init gpt2 \
      --seed "$i" --device cuda:0 --out "$out" "$@" > "$out.out" 2> "$out.err" &
    pids+=($!)
  done
  wait "${pids[@]}"
  python - "$N" <<'EOF'
import json, sys
N = int(sys.argv[1]); rates = []
for i in range(1, N + 1):
    L = [json.loads(l) for l in open(f"bench/n{N}_{i}/log.jsonl") if '"loss"' in l]
    a = next(x for x in L if x["step"] >= 400); b = L[-1]
    rates.append((b["sec"] - a["sec"]) / (b["step"] - a["step"]))
per = sum(rates) / N
print(f"N={N}: {per*1000:.0f} ms/step per run, aggregate {N/per:.1f} steps/s, "
      f"= {N*0.133/per:.2f} x one RTX 4060 K=6 run, peak {max(json.loads(l).get('peak_mb',0) for l in open(f'bench/n{N}_1/log.jsonl')):.0f} MB")
EOF
done
