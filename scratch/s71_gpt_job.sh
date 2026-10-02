#!/usr/bin/env bash
# THEORY §71 GPT-2 arms on a Vast host: bash scratch/s71_gpt_job.sh <tag> <device> <queue file> [P]
# Lines "<out dir> <boxes.s5 train args...>". Regenerates S5=128 / S5=512 with DeltaProduct's generator (seeds 7101/7102;
# sha256 must match ours, else rc 9), downloads GPT-2 (sha checked), then runs the lines (P concurrent, default 1).
cd "$(dirname "$0")/.."
tag=$1; dev=$2; q=$3; P=${4:-1}; log=logs/queue_$tag.out; mkdir -p logs prosqa/ckpt boxes/runs_s5 bench/data/s5
SHA128=6ee3ef95405dc43b094824dff8973753c587fe95a00762d2023bd64a941504f2
SHA512=a0087fe4a7496a47f76d9436e6bb089c8a8869f3f343f71057355e127d0c33da
pip install -q tiktoken polars fire pyrootutils "git+https://github.com/jopetty/abstract_algebra" > logs/s71_gpt_setup.log 2>&1
G=/workspace/s71gen; mkdir -p $G/src $G/data; cp scratch/s71_gen/generate_data.py $G/src/; touch $G/.project-root
( cd $G && PYTHONPATH=$G python src/generate_data.py --group=S5 --k=128 --samples=100000 --seed=7101 && \
  PYTHONPATH=$G python src/generate_data.py --group=S5 --k=512 --samples=100000 --seed=7102 ) >> logs/s71_gpt_setup.log 2>&1
cp $G/data/S5=128.csv $G/data/S5=512.csv bench/data/s5/
s1=$(sha256sum bench/data/s5/S5=128.csv | cut -c1-64); s2=$(sha256sum bench/data/s5/S5=512.csv | cut -c1-64)
echo "$(date +%T) data sha $s1 $s2" >> $log
if [ "$s1" != "$SHA128" ] || [ "$s2" != "$SHA512" ]; then echo "$(date +%T) DATA_SHA_MISMATCH rc 9" >> $log; echo "$(date +%T) QUEUE_DONE" >> $log; exit 1; fi
W=prosqa/ckpt/gpt2_pretrained.bin
for try in 1 2 3; do sha256sum $W 2>/dev/null | grep -q '^7c5d3f4b8b76583b' && break; rm -f $W; curl -sL -o $W https://huggingface.co/openai-community/gpt2/resolve/main/pytorch_model.bin; done   # retry a bad download
sha256sum $W | grep -q '^7c5d3f4b8b76583b' || { echo "$(date +%T) GPT2_SHA_MISMATCH rc 9" >> $log; echo "$(date +%T) QUEUE_DONE" >> $log; exit 1; }
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader >> $log
grep -v '^#' "$q" | grep -v '^$' | while read -r out rest; do n=$(basename $out)
  echo "echo \"\$(date +%T) start $out\" >> $log; python -u -m boxes.s5 train $rest --device $dev --out $out > logs/s71_$n.log 2>&1; rc=\$?; [ -s $out/eval.json ] || rc=7; echo \"\$(date +%T) evaluated $out rc \$rc\" >> $log"
done > /tmp/s71_cmds.txt
xargs -d '\n' -P $P -I{} bash -c '{}' < /tmp/s71_cmds.txt
echo "$(date +%T) QUEUE_DONE" >> $log
