#!/usr/bin/env bash
# On the pod (PROTOCOL7 T11: Coconut ProsQA recipe, GPT-2 architecture trained from scratch, no answer mask; derived from pod_prosqa_q.sh): set up the official Coconut code, train GPT-2 on ProsQA on one GPU keeping the
# stage-boundary checkpoints, then run the frozen analyses (prosqa/analyze.py) and leave JSON in /workspace/prosqa/res.
#   nohup bash /workspace/prosqa/pod_prosqa_g.sh > /workspace/logs/prosqa.out 2>&1 &
set -u
CFG=${1:-prosqa_s11p}                         # config name in prosqa/ (prosqa_s11p = pilot)
NAME=$(grep "^name:" /workspace/prosqa/$CFG.yaml | cut -d" " -f2)
export HF_HUB_ENABLE_HF_TRANSFER=0          # the image enables it without installing hf_transfer
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p /workspace/logs /workspace/ckpt /workspace/prosqa/res && cd /workspace
[ -d coconut ] || git clone -q https://github.com/facebookresearch/coconut.git
cd coconut && echo "COMMIT $(git log -1 --format=%H)"
python /workspace/prosqa/patch_coconut.py /workspace/coconut      # speed patch, checked by test_patch.py
python /workspace/prosqa/patch_scratch.py /workspace/coconut      # T11: random init (from_scratch, n_layer)
pip install -q transformers==4.46.2 wandb==0.18.7 datasets==3.1.0 tqdm pyyaml tiktoken > /workspace/logs/pip.log 2>&1
python -c "import torch,transformers;print('ENVOK',torch.__version__,transformers.__version__,torch.cuda.get_device_name())"
cp /workspace/prosqa/prosqa_1gpu.yaml /workspace/prosqa/$CFG.yaml args/
python /workspace/prosqa/make_long.py data/prosqa_test.json data/prosqa_long_test.json --m 0,2,4,6 --train data/prosqa_train.json
python /workspace/prosqa/make_long.py data/prosqa_test.json data/prosqa_ctl_test.json --m 0,2,4,6 --seed 1 --control --train data/prosqa_train.json
echo "LONG_MD5 $(md5sum data/prosqa_long_test.json | cut -c1-8) $(md5sum data/prosqa_ctl_test.json | cut -c1-8) $(md5sum data/prosqa_test.json | cut -c1-8)"
C=/workspace/ckpt/$NAME
if [ ! -f $C/checkpoint_50 ]; then
  # keep checkpoints at stage boundaries (every 5 epochs) and the latest one; the rest are ~0.5 GB each
  ( while sleep 120; do
      [ -d $C ] || continue
      last=$(ls $C | sed -n 's/checkpoint_//p' | sort -n | tail -1)
      for f in $C/checkpoint_*; do [ -e "$f" ] || continue; n=${f##*_}; [ $((n % 5)) -ne 0 ] && [ "$n" != "$last" ] && rm -f $f; done
    done ) &
  WANDB_MODE=disabled torchrun --nnodes 1 --nproc_per_node 1 run.py args/$CFG.yaml
  echo "TRAIN_EXIT $?"
fi
[ -f $C/checkpoint_50 ] || { echo "NO_FINAL_CKPT"; exit 1; }
cd /workspace/prosqa && D=/workspace/coconut/data && R=res
A() { out=$1; shift; [ -s $R/$out ] || python analyze.py --out $R/$out "$@" || echo "FAIL $out"; }
A trace_test_e50.json  --ckpt $C/checkpoint_50 --data $D/prosqa_test.json  --what trace
A kprime_test_e50.json --ckpt $C/checkpoint_50 --data $D/prosqa_test.json  --what kprime --budgets 0,1,2,6,12
for s in long ctl; do [ -s $R/p_$s.json ] || python long_eval2.py --no_mask --ckpt $C/checkpoint_50 --data $D/prosqa_${s}_test.json --out $R/p_$s.json --device cuda:0 || echo "FAIL p_$s"; done
[ -s $R/p7p_m4.npz ] || python p7_prosqa.py --no_mask --ckpt $C/checkpoint_50 --data $D/prosqa_long_test.json --out $R/p7p_m4.npz --m 4 --Kp 12 || echo "FAIL p7p"
grep -a "Accuracy\|acc" /workspace/logs/prosqa.out | tail -60 > $R/train_acc_lines.txt
echo ANALYSIS_DONE
