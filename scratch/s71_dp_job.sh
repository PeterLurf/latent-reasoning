#!/usr/bin/env bash
# THEORY §71 DeltaProduct arms on a Linux Vast host (automl/DeltaProduct state_tracking, unchanged code).
#   bash scratch/s71_dp_job.sh <tag> <device> <queue file>     (queue lines: "<arm> <seed>", arm = dp | dpvar)
# 1. Regenerate S5=128 / S5=512 with the generator and seeds 7101 / 7102; abort (rc 9) unless sha256 matches ours.
# 2. dp: the paper's command (n_h 4, 1 layer, 8 heads, bs 2048, lr 1e-3, 100 epochs, cosine). dpvar: the same with
#    strict_len=False over S5=2..128 (~790 sequences per length, seeds 7200 + L; amendment 71-1).
# 3. Predictions on the first 2,000 test sequences via boxes/s71_dp_eval.py -> boxes/runs_s5/<arm>_s<seed>/.
cd "$(dirname "$0")/.."
tag=$1; dev=$2; q=$3; log=/workspace/Latent/logs/queue_$tag.out; mkdir -p logs boxes/runs_s5
SHA128=6ee3ef95405dc43b094824dff8973753c587fe95a00762d2023bd64a941504f2
SHA512=a0087fe4a7496a47f76d9436e6bb089c8a8869f3f343f71057355e127d0c33da
{ cd /workspace; [ -d DeltaProduct ] || git clone -q https://github.com/automl/DeltaProduct.git
  cd DeltaProduct/state_tracking; touch .project-root
  pip install -q fire humanize polars pyrootutils accelerate datasets python-dotenv ordered-set tokenizers transformers wandb evaluate \
     "git+https://github.com/jopetty/abstract_algebra" "git+https://github.com/jopetty/sfirah"
  touch ../flash-linear-attention/README.md                       # the bundled fla's setup reads a README it lacks
  pip install -q -e ../flash-linear-attention
  python -c "import fire, polars, abstract_algebra, fla, sfirah; print('deps ok')"; } > /workspace/Latent/logs/s71_dp_setup.log 2>&1
W=/workspace/DeltaProduct/state_tracking; cd $W; mkdir -p data
PYTHONPATH=$PWD python src/generate_data.py --group=S5 --k=128 --samples=100000 --seed=7101 >> /workspace/Latent/logs/s71_dp_setup.log 2>&1
PYTHONPATH=$PWD python src/generate_data.py --group=S5 --k=512 --samples=100000 --seed=7102 >> /workspace/Latent/logs/s71_dp_setup.log 2>&1
s1=$(sha256sum data/S5=128.csv | cut -c1-64); s2=$(sha256sum data/S5=512.csv | cut -c1-64)
echo "$(date +%T) data sha S5=128 $s1 S5=512 $s2" >> $log
if [ "$s1" != "$SHA128" ] || [ "$s2" != "$SHA512" ]; then echo "$(date +%T) DATA_SHA_MISMATCH rc 9" >> $log; echo "$(date +%T) QUEUE_DONE" >> $log; exit 1; fi
if grep -q "^dpvar" "$q"; then for L in $(seq 2 127); do PYTHONPATH=$PWD python src/generate_data.py --group=S5 --k=$L --samples=790 --seed=$((7200 + L)) >> /workspace/Latent/logs/s71_dp_setup.log 2>&1; done; fi
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader >> $log
one() { arm=$1; seed=$2; out=/workspace/Latent/boxes/runs_s5/${arm}_s$seed; mkdir -p $out; cd $W
  extra=""; [ $arm = dpvar ] && extra="--strict_len=False"
  echo "$(date +%T) start $arm $seed" >> $log
  WANDB_MODE=disabled PYTHONPATH=$PWD python src/main.py train --group=S5 --k=128 --k_test=512 --n_layers=1 --num_householder=4 \
     --allow_neg_eigval=True --n_heads=8 --batch_size=2048 --lr=1e-3 --epochs=100 --use_scheduler=True --seed=$seed --logging=False $extra \
     > /workspace/Latent/logs/s71_${arm}_s$seed.log 2>&1
  ck=$(ls -td checkpoints/S5_*_$seed 2>/dev/null | head -1)
  rc=7; if [ -n "$ck" ]; then cp $ck/config.json $out/; PYTHONPATH=$PWD python /workspace/Latent/boxes/s71_dp_eval.py --ckpt $ck --test data/S5=512.csv \
       --out $out/preds_test.npy > $out/eval.json 2>> /workspace/Latent/logs/s71_${arm}_s$seed.log && rc=0; fi
  echo "$(date +%T) evaluated boxes/runs_s5/${arm}_s$seed rc $rc" >> $log; }
export -f one; export W log
grep -v '^#' "$q" | grep -v '^$' | xargs -d '\n' -P 3 -I{} bash -c 'one {}'
echo "$(date +%T) QUEUE_DONE" >> $log
