#!/usr/bin/env bash
# THEORY §71 GRU reference (descriptive): Merrill's word-problem code (jopetty/word-problem, MIT), train_gru, 1 layer,
# the code's defaults otherwise, on DeltaProduct's P1 data (S5=128 train, strict length; S5=512 for our scorer).
#   bash scratch/s71_gru_job.sh <tag> <device> <queue file>     (queue lines: "gru <seed>"; run one at a time)
# Interface-only patch: train_gru has no checkpointing ("# TODO: model checkpointing logic here"); we save the best
# validation model there, so it can be scored by the shared scorer. Training code is otherwise unchanged.
cd "$(dirname "$0")/.."
tag=$1; dev=$2; q=$(readlink -f "$3"); log=/workspace/Latent/logs/queue_$tag.out; mkdir -p logs boxes/runs_s5
SHA128=6ee3ef95405dc43b094824dff8973753c587fe95a00762d2023bd64a941504f2
SHA512=a0087fe4a7496a47f76d9436e6bb089c8a8869f3f343f71057355e127d0c33da
{ cd /workspace; [ -d word-problem ] || git clone -q https://github.com/jopetty/word-problem.git
  cd word-problem; touch .project-root; mkdir -p data
  pip install -q fire humanize polars pyrootutils accelerate datasets python-dotenv ordered-set tokenizers "transformers==4.46.2" wandb \
     "git+https://github.com/jopetty/abstract_algebra" "git+https://github.com/jopetty/sfirah" "git+https://github.com/jopetty/s4.git"
  python - <<'PY'
p = "src/main.py"; s = open(p).read()
start = s.index("def train_gru("); end = s.index("def train_lstm(")
seg = s[start:end]
old = "            # TODO: model checkpointing logic here\n"
assert old in seg, "no checkpoint TODO in train_gru"
seg = seg.replace(old, old + "            import os as _os; _os.makedirs('gru_ckpt', exist_ok=True); "
                  "torch.save(accelerator.unwrap_model(model).state_dict(), f'gru_ckpt/best_{seed}.pt')  # §71 interface patch\n", 1)
open(p, "w").write(s[:start] + seg + s[end:]); print("patched train_gru checkpointing")
PY
  python -c "import fire, polars, abstract_algebra, sfirah.rnn; print('deps ok')"; } > /workspace/Latent/logs/s71_gru_setup.log 2>&1
W=/workspace/word-problem; cd $W
G=/workspace/s71gen; mkdir -p $G/src $G/data; cp /workspace/Latent/scratch/s71_gen/generate_data.py $G/src/; touch $G/.project-root
( cd $G && [ -s data/S5=128.csv ] || PYTHONPATH=$G python src/generate_data.py --group=S5 --k=128 --samples=100000 --seed=7101
  cd $G && [ -s data/S5=512.csv ] || PYTHONPATH=$G python src/generate_data.py --group=S5 --k=512 --samples=100000 --seed=7102 ) >> /workspace/Latent/logs/s71_gru_setup.log 2>&1
cp $G/data/S5=128.csv $G/data/S5=512.csv data/
s1=$(sha256sum data/S5=128.csv | cut -c1-64); s2=$(sha256sum data/S5=512.csv | cut -c1-64)
echo "$(date +%T) data sha $s1 $s2" >> $log
if [ "$s1" != "$SHA128" ] || [ "$s2" != "$SHA512" ]; then echo "$(date +%T) DATA_SHA_MISMATCH rc 9" >> $log; echo "$(date +%T) QUEUE_DONE" >> $log; exit 1; fi
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader >> $log
grep -v '^#' "$q" | grep -v '^$' | while read -r arm seed; do
  out=/workspace/Latent/boxes/runs_s5/gru_s$seed; [ -s $out/preds_test.npy ] && continue; mkdir -p $out
  echo "$(date +%T) start gru $seed" >> $log
  WANDB_MODE=disabled PYTHONPATH=$PWD python src/main.py train_gru --group=S5 --k=128 --strict_len=True --n_layers=1 --seed=$seed \
     --logging=False > /workspace/Latent/logs/s71_gru_s$seed.log 2>&1
  rc=7; if [ -s gru_ckpt/best_$seed.pt ]; then PYTHONPATH=$PWD python /workspace/Latent/boxes/s71_gru_eval.py --ckpt gru_ckpt/best_$seed.pt \
       --out $out/preds_test.npy > $out/eval.json 2>> /workspace/Latent/logs/s71_gru_s$seed.log && rc=0; fi
  echo "$(date +%T) evaluated boxes/runs_s5/gru_s$seed rc $rc" >> $log
done
echo "$(date +%T) QUEUE_DONE" >> $log
