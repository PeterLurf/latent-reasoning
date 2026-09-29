#!/usr/bin/env bash
# On the pod (PROTOCOL6 addendum Q: addendum-G recipe trained only on 3-4 hop problems; derived from pod_prosqa_g.sh): set up the official Coconut code, train GPT-2 on ProsQA on one GPU keeping the
# stage-boundary checkpoints, then run the frozen analyses (prosqa/analyze.py) and leave JSON in /workspace/prosqa/res.
#   nohup bash /workspace/prosqa/pod_prosqa_g.sh > /workspace/logs/prosqa.out 2>&1 &
set -u
CFG=${1:-prosqa_q}                            # config name in prosqa/ (prosqa_q, prosqa_q_s1)
NAME=$(grep "^name:" /workspace/prosqa/$CFG.yaml | cut -d" " -f2)
export HF_HUB_ENABLE_HF_TRANSFER=0          # the image enables it without installing hf_transfer
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p /workspace/logs /workspace/ckpt /workspace/prosqa/res && cd /workspace
[ -d coconut ] || git clone -q https://github.com/facebookresearch/coconut.git
cd coconut && echo "COMMIT $(git log -1 --format=%H)"
python /workspace/prosqa/patch_coconut.py /workspace/coconut      # speed patch, checked by test_patch.py
python /workspace/prosqa/patch_mask.py /workspace/coconut         # addendum G answer mask + eager attention
pip install -q transformers==4.46.2 wandb==0.18.7 datasets==3.1.0 tqdm pyyaml tiktoken > /workspace/logs/pip.log 2>&1
python -c "import torch,transformers;print('ENVOK',torch.__version__,transformers.__version__,torch.cuda.get_device_name())"
cp /workspace/prosqa/prosqa_1gpu.yaml /workspace/prosqa/$CFG.yaml args/
python -c "import json;d=json.load(open('data/prosqa_train.json'));s=[x for x in d if len(x['steps'])<=4];json.dump(s,open('data/prosqa_train_short.json','w'));print('SHORT_TRAIN',len(s))"
python /workspace/prosqa/make_long.py data/prosqa_test.json data/prosqa_long_test.json --m 0,2,4,6 --train data/prosqa_train.json
python /workspace/prosqa/make_long.py data/prosqa_test.json data/prosqa_ctl_test.json --m 0,2,4,6 --seed 1 --control --train data/prosqa_train.json
echo "LONG_MD5 $(md5sum data/prosqa_long_test.json | cut -c1-8) $(md5sum data/prosqa_ctl_test.json | cut -c1-8) $(md5sum data/prosqa_test.json | cut -c1-8)"
OMP_NUM_THREADS=8 python /workspace/prosqa/test_mask.py > /workspace/logs/test_mask.out 2>&1
grep -q "MASK_TEST PASS" /workspace/logs/test_mask.out || { echo "MASK_TEST_FAILED"; exit 1; }
echo "MASK_TEST PASS"
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
A() { out=$1; shift; [ -s $R/$out ] || python analyze.py --mask_answer --out $R/$out "$@" || echo "FAIL $out"; }
A trace_test_e50.json  --ckpt $C/checkpoint_50 --data $D/prosqa_test.json  --what trace
A kprime_test_e50.json --ckpt $C/checkpoint_50 --data $D/prosqa_test.json  --what kprime --budgets 0,1,2,6,12
for s in long ctl; do [ -s $R/p_$s.json ] || python long_eval.py --ckpt $C/checkpoint_50 --data $D/prosqa_${s}_test.json --out $R/p_$s.json --device cuda:0 || echo "FAIL p_$s"; done
grep -a "Accuracy\|acc" /workspace/logs/prosqa.out | tail -60 > $R/train_acc_lines.txt
echo ANALYSIS_DONE
