#!/usr/bin/env bash
# On the pod (PROTOCOL8 confirmatory; derived from pod_p8.sh, adds patch_p8.py for the anytime / random-tight arms): set up the official Coconut code with the speed patch and
# the addendum-G answer mask, train GPT-2 on deep ProsQA (prosqa/p8data, uploaded, md5-checked), keep stage-boundary
# checkpoints, then run the Phase 0 evaluations and leave JSON in /workspace/prosqa/res.
#   bash /workspace/prosqa/pod_p8c.sh p8c_<arm>_s<seed>     (arm: coco | at | rt | cot)
set -u
CFG=$1
NAME=$(grep "^name:" /workspace/prosqa/$CFG.yaml | cut -d" " -f2)
KEEP=$(grep "^epochs_per_stage:" /workspace/prosqa/$CFG.yaml | cut -d" " -f2); case $CFG in p8c_cot_*) KEEP=10;; esac
NEP=$(grep "^num_epochs:" /workspace/prosqa/$CFG.yaml | cut -d" " -f2)
export HF_HUB_ENABLE_HF_TRANSFER=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
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
OMP_NUM_THREADS=8 python /workspace/prosqa/test_mask.py > /workspace/logs/test_mask.out 2>&1
grep -q "MASK_TEST PASS" /workspace/logs/test_mask.out || { echo "MASK_TEST_FAILED"; exit 1; }
echo "MASK_TEST PASS"
(rm -rf /tmp/coco_t && cp -r /workspace/coconut /tmp/coco_t && rm -rf /tmp/coco_t/.git && cd /tmp && CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4   python /workspace/prosqa/test_p8.py /tmp/coco_t /workspace/coconut/data/p8_train.json) > /workspace/logs/test_p8.out 2>&1
grep -q "P8_TEST PASS" /workspace/logs/test_p8.out || { echo "P8_TEST_FAILED"; exit 1; }
echo "P8_TEST PASS"
C=/workspace/ckpt/$NAME
if [ ! -f $C/checkpoint_$NEP ]; then
  ( while sleep 120; do
      [ -d $C ] || continue
      last=$(ls $C | sed -n 's/checkpoint_//p' | sort -n | tail -1)
      for f in $C/checkpoint_*; do [ -e "$f" ] || continue; n=${f##*_}; [ $((n % KEEP)) -ne 0 ] && [ "$n" != "$last" ] && rm -f $f; done
    done ) &
  WANDB_MODE=disabled torchrun --nnodes 1 --nproc_per_node 1 run.py args/$CFG.yaml
  echo "TRAIN_EXIT $?"
fi
[ -f $C/checkpoint_$NEP ] || { echo "NO_FINAL_CKPT"; exit 1; }
cd /workspace/prosqa && D=/workspace/coconut/data && R=res
case $CFG in
  p8c_cot_*)
    [ -s $R/cot_test_e$NEP.json ] || python eval_p8.py --ckpt $C/checkpoint_$NEP --data $D/p8_test.json --mode cot --n_per_m 150 --out $R/cot_test_e$NEP.json || echo "FAIL cot"
    [ -s $R/cot_ctl_e$NEP.json ] || python eval_p8.py --ckpt $C/checkpoint_$NEP --data $D/p8_ctl.json --mode cot --n_per_m 150 --out $R/cot_ctl_e$NEP.json || echo "FAIL cot ctl" ;;
  *)
    B=0,1,2,3,4,6,8,10,12,14,16
    [ -s $R/coco_test_e$NEP.json ] || python long_eval2.py --ckpt $C/checkpoint_$NEP --data $D/p8_test.json --out $R/coco_test_e$NEP.json --budgets $B --ext 0,1,2,3,4,5,6,7,8,9,10 || echo "FAIL coco"
    [ -s $R/coco_ctl_e$NEP.json ] || python long_eval2.py --ckpt $C/checkpoint_$NEP --data $D/p8_ctl.json --out $R/coco_ctl_e$NEP.json --budgets 0,4,10,16 --ext 4,8 || echo "FAIL coco ctl" ;;
esac
grep -a "Accuracy\|acc" /workspace/logs/prosqa.out | tail -80 > $R/train_acc_lines.txt
echo ANALYSIS_DONE
