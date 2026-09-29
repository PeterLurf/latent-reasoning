"""Check the addendum-G answer mask on the pod: zero attention mass may cross it (training forward at stages 0, 3, 6
and generation). Uses untrained GPT-2 (the mask is structural).
    cd /workspace/coconut && OMP_NUM_THREADS=8 python /workspace/prosqa/test_mask.py"""
import os
import sys
sys.path.insert(0, os.getcwd())

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

import coconut as C
from dataset import MyCollator, get_cot_latent_dataset, get_dataset, get_question_latent_dataset
from utils import Config

cfg = Config(yaml.safe_load(open("args/prosqa_1gpu.yaml")))
tok = AutoTokenizer.from_pretrained("openai-community/gpt2"); tok.pad_token = tok.eos_token
for t in ("<|start-latent|>", "<|end-latent|>", "<|latent|>"):
    tok.add_tokens(t)
lat, st, en = (tok.convert_tokens_to_ids(t) for t in ("<|latent|>", "<|start-latent|>", "<|end-latent|>"))
def build(impl):
    lm = AutoModelForCausalLM.from_pretrained("openai-community/gpt2", attn_implementation=impl)
    lm.resize_token_embeddings(len(tok))
    return C.Coconut(lm, lat, st, en, tok.eos_token_id)


m = build("eager")
ms = build("sdpa")
ms.load_state_dict(m.state_dict())
base = get_dataset("data/prosqa_valid.json", tok, max_size=16)
col = MyCollator(tok, latent_id=lat, label_pad_token_id=-100)
batches = {s: col([d for d in get_cot_latent_dataset(s, base, cfg, st, lat, en).select(range(8))]) for s in (0, 3, 6)}
qds = get_question_latent_dataset(6, base, cfg, st, lat, en)
m = m.cuda(); ms = ms.cuda(); m.eval(); ms.eval()      # Coconut.eval() returns None
ok = True
for s, b in batches.items():
    C.MASK.crossed = 0.0
    b = {k: v.cuda() for k, v in b.items() if k != "idx"}
    with torch.no_grad():
        C.MASK.check = True
        out = m(**b)
        C.MASK.check = False
        out_s = ms(**b)
        C.MASK.extra = None
    ids = b["input_ids"]
    n_ans = int((ids == C.ANS_ID).sum())
    d = abs(out.loss.item() - out_s.loss.item())
    print(f"stage {s}: loss eager {out.loss.item():.5f} sdpa {out_s.loss.item():.5f} |d| {d:.1e} answers {n_ans}/8 "
          f"crossed {C.MASK.crossed:.2e}", flush=True)
    ok &= C.MASK.crossed == 0.0 and n_ans == 8 and d < 1e-3
C.MASK.crossed = 0.0
with torch.no_grad():
    q = torch.tensor(qds[0]["input_ids"], device="cuda")[None]
    C.MASK.check = True
    g = m.generate(q, torch.ones_like(q), max_new_tokens=12)
    C.MASK.check = False
    gs = ms.generate(q, torch.ones_like(q), max_new_tokens=12)
print("generate eager:", repr(tok.decode(g[0, q.shape[1]:])), "sdpa:", repr(tok.decode(gs[0, q.shape[1]:])),
      f"crossed {C.MASK.crossed:.2e}", flush=True)
ok &= C.MASK.crossed == 0.0 and g.tolist() == gs.tolist()
# the mask must matter: without it the stage-6 loss differs
b = {k: v.cuda() for k, v in batches[6].items() if k != "idx"}
with torch.no_grad():
    l_mask = ms(**b).loss.item()
    saved = C._build_extra
    C._build_extra = lambda *a, **k: (None, None)
    l_nomask = ms(**b).loss.item()
    C._build_extra = saved
print(f"stage 6 loss with mask {l_mask:.4f} without {l_nomask:.4f}", flush=True)
ok &= abs(l_mask - l_nomask) > 1e-4
# the mask must actually block something: with the mask off, answer positions do attend to the question
C.MASK.check = False
print("MASK_TEST", "PASS" if ok else "FAIL", flush=True)
