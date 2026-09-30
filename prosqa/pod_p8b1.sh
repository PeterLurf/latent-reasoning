#!/usr/bin/env bash
# PROTOCOL8b step 1 on a Vast host: set up Coconut (speed patch, answer mask, patch_p8), wait for the uploaded Phase 0
# checkpoint (sha-checked), fine-tune at the final stage (resume 44 -> 48) with the given config, then run the
# re-anchor evaluation (end-appended and random-position checkpoints) on the fine-tuned checkpoint_48.
#   bash /workspace/prosqa/pod_p8b1.sh p8b1_<arm>_s<seed>
set -u
CFG=$1
NAME=$(grep "^name:" /workspace/prosqa/$CFG.yaml | cut -d" " -f2)
export HF_HUB_ENABLE_HF_TRANSFER=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p /workspace/logs /workspace/ckpt /workspace/prosqa/res && cd /workspace
[ -d coconut ] || git clone -q https://github.com/facebookresearch/coconut.git
cd coconut && echo "COMMIT $(git log -1 --format=%H)"
python /workspace/prosqa/patch_coconut.py /workspace/coconut
python /workspace/prosqa/patch_mask.py /workspace/coconut
python /workspace/prosqa/patch_p8.py /workspace/coconut
pip install -q transformers==4.46.2 wandb==0.18.7 datasets==3.1.0 tqdm pyyaml tiktoken > /workspace/logs/pip.log 2>&1
python -c "import torch,transformers;print('ENVOK',torch.__version__,transformers.__version__,torch.cuda.get_device_name())"
cp /workspace/prosqa/prosqa_1gpu.yaml /workspace/prosqa/$CFG.yaml args/
cp /workspace/prosqa/p8data/p8_*.json data/
echo "DATA_MD5 $(md5sum data/p8_*.json | cut -c1-10 | tr '\n' ' ')"
B=/workspace/ckpt/p8_coconut_checkpoint_48
until [ -f $B.sha256 ] && [ -s $B ]; do sleep 20; done
(cd /workspace/ckpt && sha256sum -c p8_coconut_checkpoint_48.sha256) || { echo "CKPT_SHA_FAIL"; exit 1; }
echo "CKPT_OK"
C=/workspace/ckpt/$NAME
if [ ! -f $C/checkpoint_48 ]; then
  WANDB_MODE=disabled torchrun --nnodes 1 --nproc_per_node 1 run.py args/$CFG.yaml
  echo "TRAIN_EXIT $?"
fi
[ -f $C/checkpoint_48 ] || { echo "NO_FINAL_CKPT"; exit 1; }
cd /workspace/prosqa
[ -s res/p8b1_$CFG.json ] || python -u reanchor_p8.py --ckpt $C/checkpoint_48 --data p8data/p8_test.json --out res/p8b1_$CFG.json --n_per_m 150 --Kp 10 --device cuda:0 --random_pos || echo "FAIL reanchor"
echo ANALYSIS_DONE
