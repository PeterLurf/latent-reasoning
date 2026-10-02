"""THEORY §80 premise P0 (21da4fc): does GPT-2, fine-tuned in text on k-op words, compose UNSEEN k-op words in its
forward pass?
S3 in c1's element order (c1.core PERMS, COMP[a, s] = a o s; state 0 = identity). Ops are the single-token symbols
" A".." F" (c1.core OP_TOK order: element i <-> the i-th letter). The arrangement text of state s is the letters
"XYZ"[s[i]] for boxes i = 0..2 (letters X Y Z, disjoint from the op symbols).
Words: all k-op sequences (k = 2: 36, k = 3: 216). Held-out sets, fixed by split seed 80 + k (committed in
c1/p80_split_k<k>.json before any training):
  (b) 30 % of words (rounded) never appear in training, with any start state;
  (a) a further 10 % (rounded) of the (state, word) pairs of the remaining (seen) words are excluded from training.
Text (§71 WC0 format, end state only): "start: X Y Z. ops: A, D. a:" -> " > Y X Z" + EOS; loss on the target.
Training: base GPT-2 small (HF), AdamW lr 1e-4, wd 0.01, bs 64 drawn uniformly from the training pairs, --steps
updates (default 1,500), fp32 (bf16 autocast if --bf16). Eval (greedy, exact match of the 3 letters): accuracy on the
training pairs, on (b) (every unseen word x every start state) and on (a). Rule (§80): if (b) < 0.95, §80 stops.
    python -m c1.p80_p0 split                        # writes c1/p80_split_k2.json, c1/p80_split_k3.json
    python -m c1.p80_p0 train --k 2 --device cuda:0 --out c1/runs/p80_p0_k2
"""
import argparse
import itertools
import json
import os
import random
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
PERMS = sorted(itertools.permutations(range(3)))
COMP = np.array([[PERMS.index(tuple(a[s[i]] for i in range(3))) for s in PERMS] for a in PERMS])   # = c1.core.COMP (S3)
SYM = "ABCDEF"; LET = "XYZ"; EOS = 50256


def arr(s):
    return " ".join(LET[PERMS[s][i]] for i in range(3))


def run(s, word):
    for a in word:
        s = COMP[a, s]
    return int(s)


def prompt(s, word):
    return f"start: {arr(s)}. ops: {', '.join(SYM[a] for a in word)}. a:"


def target(s, word):
    return " > " + arr(run(s, word))


def make_split(k):
    rng = random.Random(80 + k)
    words = [list(w) for w in itertools.product(range(6), repeat=k)]
    idx = list(range(len(words))); rng.shuffle(idx)
    nb = round(0.30 * len(words)); unseen = sorted(idx[:nb]); seen = sorted(idx[nb:])
    pairs = [(s, w) for w in seen for s in range(6)]; rng.shuffle(pairs)
    na = round(0.10 * len(pairs)); held_a = sorted(pairs[:na]); train = sorted(pairs[na:])
    return {"k": k, "split_seed": 80 + k, "words": words, "unseen_words_b": unseen, "seen_words": seen,
            "heldout_pairs_a": held_a, "train_pairs": train}


@torch.no_grad()
def evaluate(model, tok, pairs, words, dev, bs=128):
    model.eval(); tok.padding_side = "left"; ok = 0
    for i in range(0, len(pairs), bs):
        b = pairs[i:i + bs]
        enc = tok([prompt(s, words[w]) for s, w in b], return_tensors="pt", padding=True).to(dev)
        out = model.generate(**enc, max_new_tokens=8, do_sample=False, pad_token_id=EOS)
        for (s, w), o in zip(b, out):
            t = tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True)
            ok += int(t.startswith(target(s, words[w])))
    return ok / max(1, len(pairs))


def train(a):
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    sp = json.load(open(f"c1/p80_split_k{a.k}.json")); words = sp["words"]
    torch.manual_seed(a.seed); rng = random.Random(a.seed)
    tok = GPT2TokenizerFast.from_pretrained("openai-community/gpt2"); tok.pad_token = tok.eos_token
    model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").to(a.device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    tr = [tuple(p) for p in sp["train_pairs"]]
    enc = {p: (tok.encode(prompt(p[0], words[p[1]])), tok.encode(target(p[0], words[p[1]])) + [EOS]) for p in tr}
    os.makedirs(a.out, exist_ok=True); json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w", newline="\n"), indent=1)
    log = open(os.path.join(a.out, "log.jsonl"), "w", newline="\n"); t0 = time.time()
    amp = a.bf16 and a.device.startswith("cuda")
    for step in range(1, a.steps + 1):
        model.train(); b = [enc[rng.choice(tr)] for _ in range(a.bs)]; L = max(len(q) + len(r) for q, r in b)
        ids = torch.full((len(b), L), EOS); lab = torch.full((len(b), L), -100); att = torch.zeros(len(b), L, dtype=torch.long)
        for j, (q, r) in enumerate(b):
            s = q + r; ids[j, :len(s)] = torch.tensor(s); att[j, :len(s)] = 1; lab[j, len(q):len(s)] = torch.tensor(r)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            loss = model(input_ids=ids.to(a.device), attention_mask=att.to(a.device), labels=lab.to(a.device)).loss
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0 or step == a.steps:
            log.write(json.dumps({"step": step, "loss": loss.item(), "sec": round(time.time() - t0)}) + "\n"); log.flush()
    unseen_b = [(s, w) for w in sp["unseen_words_b"] for s in range(6)]
    res = {"k": a.k, "steps": a.steps, "train_sec": round(time.time() - t0), "n_train_pairs": len(tr),
           "acc_train": evaluate(model, tok, tr, words, a.device),
           "acc_b_unseen_words": evaluate(model, tok, unseen_b, words, a.device), "n_b": len(unseen_b),
           "acc_a_heldout_pairs": evaluate(model, tok, [tuple(p) for p in sp["heldout_pairs_a"]], words, a.device),
           "n_a": len(sp["heldout_pairs_a"])}
    res["P0_passes"] = res["acc_b_unseen_words"] >= 0.95
    json.dump(res, open(os.path.join(a.out, "eval.json"), "w", newline="\n"), indent=1); print(res, flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["split", "train"])
    ap.add_argument("--k", type=int, default=2); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--steps", type=int, default=1500); ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.cmd == "split":
        for k in (2, 3):
            sp = make_split(k); json.dump(sp, open(f"c1/p80_split_k{k}.json", "w", newline="\n"))
            print(k, "words", len(sp["words"]), "unseen(b)", len(sp["unseen_words_b"]), "train pairs", len(sp["train_pairs"]),
                  "heldout(a)", len(sp["heldout_pairs_a"]))
    else:
        train(a)


if __name__ == "__main__":
    main()
