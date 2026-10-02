#!/usr/bin/env bash
# THEORY §61b (F1-S replication) on a Vast host: bash scratch/f1sb_vast_job.sh <tag> <device> <queue file>; lines "<run> <seed>".
# Per run: host.txt (GPU name/driver, torch/CUDA versions) in the run dir; frozen prosqa/ail_f1s.py train + eval; then
# bench prosqa_deep_na --per_depth 150 with f1s:<run dir>?P=16. All queue lines run in parallel on the GPU (~4 GB each).
cd "$(dirname "$0")/.."
tag=$1; dev=$2; q=$3; log=logs/queue_$tag.out; mkdir -p logs prosqa/runs_ail prosqa/ckpt bench/results
pip install -q tiktoken > logs/pip_ail.log 2>&1
W=prosqa/ckpt/gpt2_pretrained.bin
for try in 1 2 3; do sha256sum $W 2>/dev/null | grep -q '^7c5d3f4b8b76583b' && break; rm -f $W; curl -sL -o $W https://huggingface.co/openai-community/gpt2/resolve/main/pytorch_model.bin; done   # retry a bad download
sha256sum $W | grep -q '^7c5d3f4b8b76583b' || { echo "$(date +%T) GPT2_SHA_MISMATCH rc 9" >> $log; echo "$(date +%T) QUEUE_DONE" >> $log; exit 1; }
one() {
  r=$1; seed=$2
  d=prosqa/runs_ail/$r; mkdir -p $d
  { nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader; python -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda,'bf16',torch.cuda.is_bf16_supported())"; } > $d/host.txt 2>&1
  echo "$(date +%T) start $r" >> $log
  python -u prosqa/ail_f1s.py --out $d --seed $seed --device $dev > logs/$r.out 2>&1
  python -u prosqa/ail_f1s.py --eval $d --device $dev >> logs/$r.out 2>&1
  python -u -m bench.run --task prosqa_deep_na --model "f1s:$d?P=16" --per_depth 150 --device $dev > logs/bench_$r.log 2>&1
  rc=0; [ -s $d/eval.json ] || rc=7; grep -q WROTE logs/bench_$r.log || rc=8
  echo "$(date +%T) evaluated $r rc $rc" >> $log
}
while read -r r seed; do case "$r" in \#*|"") continue;; esac; one $r $seed & done < "$q"
wait
echo "$(date +%T) QUEUE_DONE" >> $log
