#!/usr/bin/env bash
# §221 G-FID(a) zen: one-time setup for scratch/s221_official_zen.py (idempotent; CPU / network only, no GPU).
#  1. the official repo zhenyi4/codi at the pinned commit, LF checkout (core.autocrlf false), into third_party/codi
#  2. an isolated venv (system site packages = the port's torch / transformers) + the repo's pins of peft / accelerate, --no-deps
#  3. HF prefetch: gpt2 (test.py's --model_name_or_path; the base is overwritten by the checkpoint) and gsm8k main test (test.py's load_dataset),
#     and a check that the hub test split equals prosqa/ckpt_ext/hf/data/gsm8k_test.json item for item (question, gold)
set -euo pipefail
ROOT=/d/Latent
REPO=$ROOT/third_party/codi
COMMIT=2c2314662c63e9f482ebc46614ffe9af17a241e5
VENV=$ROOT/third_party/venv_codi
if [ ! -d "$REPO/.git" ]; then
  git clone --no-checkout https://github.com/zhenyi4/codi "$REPO"
  git -C "$REPO" config core.autocrlf false
  git -C "$REPO" checkout -q "$COMMIT"
fi
test "$(git -C "$REPO" rev-parse HEAD)" = "$COMMIT"
test -z "$(git -C "$REPO" status --porcelain)"
if [ ! -x "$VENV/Scripts/python.exe" ]; then
  python -m venv --system-site-packages "$VENV"
  "$VENV/Scripts/python.exe" -m pip install --no-deps --no-cache-dir peft==0.15.2 accelerate==1.7.0
fi
"$VENV/Scripts/python.exe" -c "import peft, accelerate, torch, transformers, datasets; print('venv ok', peft.__version__, accelerate.__version__, torch.__version__, transformers.__version__, datasets.__version__)"
CUDA_VISIBLE_DEVICES="" HF_HUB_DISABLE_XET=1 "$VENV/Scripts/python.exe" -I -c "
import json
from huggingface_hub import snapshot_download
import datasets
p = snapshot_download('gpt2', allow_patterns=['config.json', 'generation_config.json', 'model.safetensors', 'tokenizer*', 'vocab.json', 'merges.txt']); print('gpt2 snapshot', p)  # xet 404s on the 'gpt2' alias (10-21): plain HTTP
t = datasets.load_dataset('openai/gsm8k', 'main')['test']   # = test.py's load_dataset('gsm8k', 'main'); the bare id is rejected by this hub client (D3b)
loc = json.load(open(r'D:/Latent/prosqa/ckpt_ext/hf/data/gsm8k_test.json', encoding='utf-8'))
hq = list(t['question']); hg = [float(s.split('####')[-1].replace(',', '').strip()) for s in t['answer']]
lg = [float(str(x).replace(',', '').split('####')[-1].strip()) for x in loc['answer']]
print('gsm8k test n', len(hq), 'questions equal local', hq == loc['question'], 'golds equal local', hg == lg, 'cache', [c['filename'] for c in t.cache_files])
assert hq == loc['question'] and hg == lg
"
echo "setup ok"
