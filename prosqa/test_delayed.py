"""Check addendum J (delayed query) on the pod before training: the query sits right after <|end-latent|> in training
and evaluation sequences, is never a target, and exactly the intended positions are blocked from the prefix; eager and
SDPA agree and no attention mass crosses the mask. Untrained GPT-2 (the checks are structural).
    cd /workspace/coconut && OMP_NUM_THREADS=8 python /workspace/prosqa/test_delayed.py"""
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
base = get_dataset("data/prosqa_d_valid.json", tok, max_size=16)
col = MyCollator(tok, latent_id=lat, label_pad_token_id=-100)
ok = True


def layout(ids):
    """Unpadded positions: <start>, <end>, the query's '?', '###' (or None)."""
    e = ids.index(en)
    return ids.index(st), e, ids.index(30, e), (ids.index(C.ANS_ID) if C.ANS_ID in ids else None)


for s in (0, 3, 6):
    raw = [d for d in get_cot_latent_dataset(s, base, cfg, st, lat, en).select(range(8))]
    lays = []
    for d, b0 in zip(raw, [base[i] for i in range(8)]):
        ids, lab = list(d["input_ids"]), list(d["labels"])
        e = ids.index(en)
        qt = b0["query_tokenized"]
        good = ids[e + 1:e + 1 + len(qt)] == qt and all(l == -100 for l in lab[:e + 1 + len(qt)]) and len(qt) > 3
        ok &= good
        if not good:
            print("SEQUENCE LAYOUT WRONG at stage", s, tok.decode(ids[e - 2:e + 1 + len(qt) + 2]), flush=True)
        lays.append(layout(ids))
    b = col([dict(d) for d in raw])                               # the collator pads in place
    ids = b["input_ids"]
    T = ids.shape[1]
    _, M = C._build_extra(ids, st, en, False)                     # (B, T, T) bool
    for i, (sp, e, qm, ap) in enumerate(lays):
        off = ids[i].tolist().index(st) - sp                      # left padding (latent alignment)
        rows = set(range(e + off, qm + 2 + off)) | (set(range(ap + off, T)) if ap is not None else set())
        got_rows = set(torch.nonzero(M[i].any(1)).flatten().tolist())
        got_cols = set(torch.nonzero(M[i].any(0)).flatten().tolist())
        if got_rows != rows or got_cols != set(range(sp + off)):
            ok = False
            print(f"MASK ROWS WRONG stage {s} ex {i}: extra {sorted(got_rows - rows)[:6]} missing {sorted(rows - got_rows)[:6]}"
                  f" cols {len(got_cols)} vs {sp + off}", flush=True)
    print(f"stage {s}: layout and mask rows checked on {len(lays)} examples", flush=True)
    C.MASK.crossed = 0.0
    bb = {k: v.cuda() for k, v in b.items() if k != "idx"}
    m = m.cuda(); ms = ms.cuda(); m.eval(); ms.eval()          # Coconut.eval() returns None
    with torch.no_grad():
        C.MASK.check = True
        out = m(**bb)
        C.MASK.check = False
        out_s = ms(**bb)
        C.MASK.extra = None
    dl = abs(out.loss.item() - out_s.loss.item())
    print(f"stage {s}: loss eager {out.loss.item():.5f} sdpa {out_s.loss.item():.5f} |d| {dl:.1e} crossed {C.MASK.crossed:.2e}",
          flush=True)
    ok &= C.MASK.crossed == 0.0 and dl < 1e-3

qds = get_question_latent_dataset(6, base, cfg, st, lat, en)
q = torch.tensor(qds[0]["input_ids"], device="cuda")[None]
qt = base[0]["query_tokenized"]
ok &= q[0, -len(qt):].tolist() == qt
print("eval prompt tail:", repr(tok.decode(q[0, -len(qt) - 3:])), flush=True)
C.MASK.crossed = 0.0
with torch.no_grad():
    C.MASK.check = True
    g = m.generate(q, torch.ones_like(q), max_new_tokens=12)
    C.MASK.check = False
    gs = ms.generate(q, torch.ones_like(q), max_new_tokens=12)
print("generate eager:", repr(tok.decode(g[0, q.shape[1]:])), "sdpa:", repr(tok.decode(gs[0, q.shape[1]:])),
      f"crossed {C.MASK.crossed:.2e}", flush=True)
ok &= C.MASK.crossed == 0.0 and g.tolist() == gs.tolist()
print("DELAYED_TEST", "PASS" if ok else "FAIL", flush=True)
