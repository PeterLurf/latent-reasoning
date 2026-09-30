"""Checks for patch_p8.py on a tiny random GPT-2 (CPU):
  python test_p8.py <patched coconut dir> <p8 train json>
1. The anytime branch at j = K reproduces the ordinary pass's answer logits: same cache, positions and answer mask.
2. With no supervised (example, j) pairs (C = -100), the training loss equals the ordinary loss exactly.
3. Anytime with C = 0 gives a finite loss larger than the ordinary one, and backward works.
4. random_tight: at stage 5 every example has 1..5 latents, the skipped steps equal the latents, and two calls (two
   epochs) draw differently. (Independence from the example holds by construction: K is a hash of (salt, idx).)
Prints P8_TEST PASS or raises.
"""
import json
import os
import sys

import torch

sys.path.insert(0, sys.argv[1])
os.chdir(sys.argv[1])
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel  # noqa: E402

from coconut import Coconut  # noqa: E402
from dataset import MyCollator, get_cot_latent_dataset, get_dataset  # noqa: E402
from utils import Config  # noqa: E402

torch.manual_seed(0)
data = json.load(open(sys.argv[2]))[:16]
tmp = "/tmp/p8_test_train.json"
json.dump(data, open(tmp, "w"))
tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
tok.pad_token = tok.eos_token
for t in ("<|start-latent|>", "<|end-latent|>", "<|latent|>"):
    tok.add_tokens(t)
lat, st, en = (tok.convert_tokens_to_ids(t) for t in ("<|latent|>", "<|start-latent|>", "<|end-latent|>"))
base = GPT2LMHeadModel(GPT2Config(n_layer=2, n_embd=64, n_head=2, vocab_size=len(tok), n_positions=1024,
                                   resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0))   # deterministic in train mode
model = Coconut(base, lat, st, en, tok.eos_token_id)
Coconut.DEPTH = {i: len(x["steps"]) for i, x in enumerate(data)}
cfg = Config({"uniform_prob": 0.0, "max_latent_stage": 10, "pad_latent_to_max": True, "no_cot": False,
              "c_thought": 1, "random_tight": False})
ds = get_dataset(tmp, tok)
col = MyCollator(tok, latent_id=lat, label_pad_token_id=-100)


def batch_at(stage, cfg, n=8):
    d = get_cot_latent_dataset(stage, ds, cfg, st, lat, en, no_special_marker=False, shuffle=False)
    b = col([d[i] for i in range(n)])
    return b


# final stage (11 > max 10): all steps latent, answer follows <|end-latent|>
b = batch_at(11, cfg)
idx = b.pop("idx").tolist()
model.train()
model.anytime_C = None
out0 = model(**b, sample_idx=idx)
cap = {}
orig = Coconut._anytime_loss


def spy(self, *a, **k):
    cap["args"] = a
    return orig(self, *a, **k)


Coconut._anytime_loss = spy
model.anytime_C = -100
torch.manual_seed(0)
out1 = model(**b, sample_idx=idx)
assert torch.allclose(out0.loss, out1.loss, atol=0, rtol=0), (out0.loss, out1.loss)
print("2. no supervised pairs: loss identical", float(out0.loss))

input_ids, attention_mask, position_ids, latent_lists, kv, sidx = cap["args"]
K = len(latent_lists[0])
keep = orig(model, input_ids, attention_mask, position_ids, latent_lists, kv, sidx, only_j=K, return_logits=True)
sel, lg = keep[K]
full = out1.logits
mx = 0.0
for r, bb in enumerate(sel):
    last = int(attention_mask[bb].nonzero().max())
    hits = (input_ids[bb, : last + 1] == 21017).nonzero()
    a0 = int(hits.max())
    La = last + 1 - a0
    pe = latent_lists[bb][-1] + 1                       # <|end-latent|> position
    assert int(input_ids[bb, pe]) == en
    mx = max(mx, float((full[bb, pe: pe + La] - lg[r, :La]).abs().max()))
assert mx < 1e-4, mx
print("1. branch at j = K reproduces the ordinary answer logits, max |diff|", mx)

model.anytime_C = 0
out2 = model(**b, sample_idx=idx)
assert torch.isfinite(out2.loss) and float(out2.loss) > float(out0.loss) - 1e-6
out2.loss.backward()
assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
print("3. anytime C = 0: loss", float(out2.loss), "> ordinary", float(out0.loss), "; backward ok")

cfg.random_tight = True
draws = []
for rep in range(2):
    d = get_cot_latent_dataset(5, ds, cfg, st, lat, en, no_special_marker=False, shuffle=False)
    ks = []
    for i in range(len(d)):
        ids = d[i]["input_ids"]
        k = ids.count(lat)
        n = len(data[d[i]["idx"]]["steps"])
        assert 1 <= k <= 5, k
        # skipped steps == k: explicit steps after <|end-latent|> equal n - k (or none)
        after = tok.decode(ids[ids.index(en) + 1:])
        n_explicit = after.count("\n") - 0
        assert n_explicit == max(n - k, 0), (k, n, n_explicit, after[:120])
        ks.append(k)
    draws.append(ks)
assert draws[0] != draws[1], "two calls drew identical budgets"
print("4. random_tight: budgets", draws[0], "then", draws[1])
# 5. jitter: off -> identical to the plain loss; on -> finite and different in train mode, identical in eval mode
cfg.random_tight = False
b = batch_at(11, cfg)
idx = b.pop("idx").tolist()
model.anytime_C = None
model.jitter = None
model.train()
l0 = float(model(**b, sample_idx=idx).loss)
model.jitter, model.jitter_p = 2.0, 1.0
l1 = float(model(**b, sample_idx=idx).loss)
assert l1 == l1 and abs(l1 - l0) > 1e-6, (l0, l1)
model.eval()
e1 = float(model(**b).loss)
model.jitter = None
e0 = float(model(**b).loss)
assert e0 == e1, (e0, e1)
print("5. jitter: train loss", l0, "->", l1, "; eval unchanged", e0)
# 6. run.py passes sample_idx only to Coconut models (plain GPT-2 CoT / no-CoT arms reject the keyword)
src = open(os.path.join(sys.argv[1], "run.py")).read()
assert 'if configs.coconut else {}' in src and "sample_idx=sample_idx)" not in src, "sample_idx not guarded in run.py"
try:
    GPT2LMHeadModel(GPT2Config(n_layer=1, n_embd=32, n_head=2, vocab_size=len(tok)))(**b)
except TypeError as e:
    raise AssertionError(f"plain GPT-2 forward failed on a Coconut batch: {e}")
print("6. run.py guards sample_idx; plain GPT-2 forward accepts the batch without it")
print("P8_TEST PASS")
