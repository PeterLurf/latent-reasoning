#!/usr/bin/env bash
# Generic Vast job (THEORY §91/§92): bash scratch/gen_vast_job.sh <tag> <device> <queue file> [P]
# Queue lines "<out dir> <module> <args...>"; P concurrent (default 1). A run is "evaluated" iff <out>/eval.json exists
# (rc captured right after python). Setup: transformers 4.46.2 + tiktoken; GPT-2 weights (uploaded, sha-checked) for
# boxes.*; S5 data regenerated with DeltaProduct's generator (sha-checked, as §71) if the queue uses boxes.latent86 or boxes.gen97.
cd "$(dirname "$0")/.."
tag=$1; dev=$2; q=$3; P=${4:-1}; log=logs/queue_$tag.out; mkdir -p logs
pq=$(grep -o "^# P=[0-9]*" "$q" | head -1 | cut -d= -f2); [ -n "$pq" ] && P=$pq          # concurrency from the queue header
for try in 1 2 3 4; do pip install -q transformers==4.46.2 tiktoken >> logs/pip_gen.log 2>&1; python -c "import transformers, tiktoken" >> logs/pip_gen.log 2>&1 && break; sleep 30; done   # 2026-10-06: a pip network blip left e32 without tiktoken
python -c "import transformers, tiktoken" 2>/dev/null || { echo "$(date +%T) PIP_FAILED rc 9 (transformers / tiktoken missing after 4 tries)" >> $log; echo "$(date +%T) QUEUE_DONE" >> $log; exit 1; }
for try in 1 2 3; do python -c "from transformers import GPT2LMHeadModel, GPT2TokenizerFast; GPT2LMHeadModel.from_pretrained('openai-community/gpt2'); GPT2TokenizerFast.from_pretrained('openai-community/gpt2')" >> logs/pip_gen.log 2>&1 && break; sleep 30; done
G=prosqa/ckpt/gpt2_pretrained.bin                                       # Audit 2026-10-06: fetched on-host from the HF hub unless shipped
if [ ! -s $G ]; then                                                    # (SHIP_GPT2=1 at deploy); only a sha-verified file is kept
  mkdir -p prosqa/ckpt; for try in 1 2 3 4; do curl -sL --retry 3 -o $G.tmp https://huggingface.co/openai-community/gpt2/resolve/main/pytorch_model.bin && sha256sum $G.tmp | grep -q "^7c5d3f4b8b76583b" && mv $G.tmp $G && break; sleep 20; done
  echo "$(date +%T) gpt2 bin from the HF hub: $([ -s $G ] && echo ok || echo FAILED)" >> $log
fi
if grep -q "boxes.fade89" "$q"; then sha256sum prosqa/ckpt/gpt2_pretrained.bin | grep -q "^7c5d3f4b8b76583b" || { echo "$(date +%T) GPT2_SHA_MISMATCH rc 9" >> $log; echo "$(date +%T) QUEUE_DONE" >> $log; exit 1; }; fi
if grep -q -E "boxes.(latent86|gen97|snap101|self103|mk108|self110|fsq120|base119)" "$q"; then
  W=prosqa/ckpt/gpt2_pretrained.bin; sha256sum $W | grep -q '^7c5d3f4b8b76583b' || { echo "$(date +%T) GPT2_SHA_MISMATCH rc 9" >> $log; echo "$(date +%T) QUEUE_DONE" >> $log; exit 1; }
  SHA128=6ee3ef95405dc43b094824dff8973753c587fe95a00762d2023bd64a941504f2; SHA512=a0087fe4a7496a47f76d9436e6bb089c8a8869f3f343f71057355e127d0c33da
  for try in 1 2 3 4; do pip install -q polars fire pyrootutils "git+https://github.com/jopetty/abstract_algebra" >> logs/pip_gen.log 2>&1; python -c "import polars, fire, pyrootutils" >> logs/pip_gen.log 2>&1 && break; sleep 30; done   # 2026-10-06: b68 lost fire to a pip build error -> empty data sha
  G=/workspace/s71gen; mkdir -p $G/src $G/data bench/data/s5; cp scratch/s71_gen/generate_data.py $G/src/; touch $G/.project-root
  for gtry in 1 2; do                                                    # 2026-10-06 b80: a CPU without polars' required features crashed the
    [ $gtry = 2 ] && pip install -q "polars[rtcompat]" >> logs/pip_gen.log 2>&1   # generator -> retry once with the compatibility build
    ( cd $G && PYTHONPATH=$G python src/generate_data.py --group=S5 --k=128 --samples=100000 --seed=7101 && \
      PYTHONPATH=$G python src/generate_data.py --group=S5 --k=512 --samples=100000 --seed=7102 ) >> logs/pip_gen.log 2>&1
    cp $G/data/S5=128.csv $G/data/S5=512.csv bench/data/s5/
    s1=$(sha256sum bench/data/s5/S5=128.csv | cut -c1-64); s2=$(sha256sum bench/data/s5/S5=512.csv | cut -c1-64)
    echo "$(date +%T) data sha $s1 $s2" >> $log
    [ "$s1" = "$SHA128" ] && [ "$s2" = "$SHA512" ] && break
  done
  [ "$s1" = "$SHA128" ] && [ "$s2" = "$SHA512" ] || { echo "$(date +%T) DATA_SHA_MISMATCH rc 9" >> $log; echo "$(date +%T) QUEUE_DONE" >> $log; exit 1; }
fi
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader >> $log
if [ -f scratch/hostcheck.py ]; then                                     # Theory note 135-6: VRAM pattern test over most of the card (b78 fault at 6.75-14 GiB)
  gib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | awk '{printf "%.0f", $1 / 1024 - 2}')
  python -u scratch/hostcheck.py --gib $gib > logs/hostcheck.log 2>&1; tail -1 logs/hostcheck.log >> $log
  tail -1 logs/hostcheck.log | grep -q "HOSTCHECK PASS" || { echo "$(date +%T) HOSTCHECK_FAIL rc 9" >> $log; echo "$(date +%T) QUEUE_DONE" >> $log; exit 1; }
fi
grep -v '^#' "$q" | grep -v '^$' | while read -r out mod rest; do n=$(basename $out)
  echo "echo \"\$(date +%T) start $out\" >> $log; python -u -m $mod $rest --device $dev --out $out > logs/gen_$n.log 2>&1; rc=\$?; [ -s $out/eval.json ] || rc=\"\$rc no-eval\"; echo \"\$(date +%T) evaluated $out rc \$rc\" >> $log"
done > /tmp/gen_cmds.txt
xargs -d '\n' -P $P -I{} bash -c '{}' < /tmp/gen_cmds.txt
echo "$(date +%T) QUEUE_DONE" >> $log
