"""Check the speed patch: official vs patched Coconut forward/backward on real ProsQA batches (stages 1, 3, 6).
    cd /workspace/coconut && OMP_NUM_THREADS=8 python /workspace/prosqa/test_patch.py <checkpoint>
Order mirrors run.py (model before datasets.map, which forks); one model, both forward functions, same dropout seed."""
import os
import sys
sys.path.insert(0, os.getcwd())          # run from the coconut checkout
import time

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

import coconut as P
import coconut_orig as O
from dataset import MyCollator, get_cot_latent_dataset, get_dataset
from utils import Config

cfg = Config(yaml.safe_load(open("args/prosqa_1gpu.yaml")))
tok = AutoTokenizer.from_pretrained("openai-community/gpt2"); tok.pad_token = tok.eos_token
for t in ("<|start-latent|>", "<|end-latent|>", "<|latent|>"):
    tok.add_tokens(t)
lat, st, en = (tok.convert_tokens_to_ids(t) for t in ("<|latent|>", "<|start-latent|>", "<|end-latent|>"))
base_lm = AutoModelForCausalLM.from_pretrained("openai-community/gpt2"); base_lm.resize_token_embeddings(len(tok))
m = P.Coconut(base_lm, lat, st, en, tok.eos_token_id)
print(m.load_state_dict(torch.load(sys.argv[1], map_location="cpu"), strict=False), flush=True)
m = m.cuda(); m.train()
base = get_dataset("data/prosqa_valid.json", tok, max_size=64)
col = MyCollator(tok, latent_id=lat, label_pad_token_id=-100)
batches = {s: col([d for d in get_cot_latent_dataset(s, base, cfg, st, lat, en).select(range(8))]) for s in (1, 3, 6)}
worst = 0.0
for stage in (1, 3, 6):
    batch = {k: v.cuda() for k, v in batches[stage].items() if k != "idx"}
    res = []
    for M in (O, P):
        m.zero_grad(set_to_none=True)
        torch.manual_seed(0); torch.cuda.synchronize(); t0 = time.time()
        out = M.Coconut.forward(m, **batch); out.loss.backward(); torch.cuda.synchronize()
        g = torch.cat([p.grad.flatten() for p in m.parameters() if p.grad is not None])
        res.append((out.loss.item(), g.clone(), time.time() - t0))
    dl = abs(res[0][0] - res[1][0]); dg = ((res[0][1] - res[1][1]).norm() / res[0][1].norm()).item()
    worst = max(worst, dl, dg)
    print(f"stage {stage}: loss {res[0][0]:.6f} vs {res[1][0]:.6f} |dloss| {dl:.2e} rel grad diff {dg:.2e} "
          f"time {res[0][2]:.2f}s vs {res[1][2]:.2f}s", flush=True)
print("PATCH_TEST", "PASS" if worst < 1e-4 else "FAIL", worst, flush=True)
