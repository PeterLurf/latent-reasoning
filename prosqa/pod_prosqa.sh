#!/usr/bin/env bash
# On the pod (PROTOCOL6 addenda C/D): set up the official Coconut code, train GPT-2 on ProsQA on one GPU keeping the
# stage-boundary checkpoints, then run the frozen analyses (prosqa/analyze.py) and leave JSON in /workspace/prosqa/res.
#   nohup bash /workspace/prosqa/pod_prosqa.sh > /workspace/logs/prosqa.out 2>&1 &
set -u
export HF_HUB_ENABLE_HF_TRANSFER=0          # the image enables it without installing hf_transfer
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p /workspace/logs /workspace/ckpt /workspace/prosqa/res && cd /workspace
[ -d coconut ] || git clone -q https://github.com/facebookresearch/coconut.git
cd coconut && echo "COMMIT $(git log -1 --format=%H)"
python /workspace/prosqa/patch_coconut.py /workspace/coconut      # speed patch, checked by test_patch.py
pip install -q transformers==4.46.2 wandb==0.18.7 datasets==3.1.0 tqdm pyyaml tiktoken > /workspace/logs/pip.log 2>&1
python -c "import torch,transformers;print('ENVOK',torch.__version__,transformers.__version__,torch.cuda.get_device_name())"
cp /workspace/prosqa/prosqa_1gpu.yaml args/
C=/workspace/ckpt/prosqa-coconut
if [ ! -f $C/checkpoint_50 ]; then
  # keep checkpoints at stage boundaries (every 5 epochs) and the latest one; the rest are ~0.5 GB each
  ( while sleep 120; do
      [ -d $C ] || continue
      last=$(ls $C | sed -n 's/checkpoint_//p' | sort -n | tail -1)
      for f in $C/checkpoint_*; do [ -e "$f" ] || continue; n=${f##*_}; [ $((n % 5)) -ne 0 ] && [ "$n" != "$last" ] && rm -f $f; done
    done ) &
  WANDB_MODE=disabled torchrun --nnodes 1 --nproc_per_node 1 run.py args/prosqa_1gpu.yaml
  echo "TRAIN_EXIT $?"
fi
[ -f $C/checkpoint_50 ] || { echo "NO_FINAL_CKPT"; exit 1; }
cd /workspace/prosqa && D=/workspace/coconut/data && R=res
A() { out=$1; shift; [ -s $R/$out ] || python analyze.py --out $R/$out "$@" || echo "FAIL $out"; }
A trace_valid_e50.json --ckpt $C/checkpoint_50 --data $D/prosqa_valid.json --what trace
A trace_test_e50.json  --ckpt $C/checkpoint_50 --data $D/prosqa_test.json  --what trace
A block_test_e50.json  --ckpt $C/checkpoint_50 --data $D/prosqa_test.json  --what block
A kprime_test_e50.json --ckpt $C/checkpoint_50 --data $D/prosqa_test.json  --what kprime
for e in 30 35 40 45; do A trace_test_e$e.json --ckpt $C/checkpoint_$e --data $D/prosqa_test.json --what trace; done
grep -a "Accuracy\|acc" /workspace/logs/prosqa.out | tail -60 > $R/train_acc_lines.txt
echo ANALYSIS_DONE
