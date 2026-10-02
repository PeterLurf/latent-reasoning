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
updates (default 4,000; Theory, before any P0 result; held-out also logged at 1,500), fp32 (bf16 autocast if --bf16). Eval (greedy, exact match of the 3 letters): accuracy on the
training pairs, on (b) (every unseen word x every start state) and on (a). Rule (§80): if (b) at the final 4,000-step checkpoint < 0.95, §80 stops.
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
    unseen_b = [(s, w) for w in sp["unseen_words_b"] for s in range(6)]; mid = None
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
        if step == 1500 and a.steps > 1500:                                    # Theory: held-out accuracy at 1,500 too
            mid = {"step": 1500, "acc_train": evaluate(model, tok, tr, words, a.device),
                   "acc_b_unseen_words": evaluate(model, tok, unseen_b, words, a.device),
                   "acc_a_heldout_pairs": evaluate(model, tok, [tuple(p) for p in sp["heldout_pairs_a"]], words, a.device)}
            log.write(json.dumps(mid) + "\n"); log.flush(); print(mid, flush=True)
    res = {"k": a.k, "at_1500": mid, "steps": a.steps, "train_sec": round(time.time() - t0), "n_train_pairs": len(tr),
           "acc_train": evaluate(model, tok, tr, words, a.device),
           "acc_b_unseen_words": evaluate(model, tok, unseen_b, words, a.device), "n_b": len(unseen_b),
           "acc_a_heldout_pairs": evaluate(model, tok, [tuple(p) for p in sp["heldout_pairs_a"]], words, a.device),
           "n_a": len(sp["heldout_pairs_a"])}
    res["P0_passes"] = res["acc_b_unseen_words"] >= 0.95                          # at the final (4,000-step) checkpoint
    json.dump(res, open(os.path.join(a.out, "eval.json"), "w", newline="\n"), indent=1); print(res, flush=True)


# ---------------------------------------------------------------- THEORY §88 (30857ca): loop-aligned composition (M8)
# Same split files, text prompt and held-out sets as P0. GPT-2 small from base: layers 0-3 once, layers 4-7 looped k
# times (shared weights); layers 8-11 are off the loss path (the answer is the head after iteration k) and are skipped
# (note 88-2). After loop iteration i, one shared linear 6-way head on ln_f(residual at the last prompt token)
# predicts the state after op i. Arms: align (CE at every iteration i = 1..k, mean) / end (CE at iteration k only).
# AdamW lr 1e-4, bs 64, 4,000 steps, fp32, training pairs only. eval.json: accuracy (iteration-k head) on training
# pairs, (b) unseen words, (a) held-out pairs; per-iteration accuracy on (b).
#   python -m c1.p80_p0 loop --arm align --k 2 --seed 1 --device cuda:0 --out c1/runs/p88_align_k2_s1

def prefix_states(s, word):
    out = []
    for a in word:
        s = int(COMP[a, s]); out.append(s)
    return out


class Looped(torch.nn.Module):
    def __init__(self, gpt, k):
        super().__init__()
        self.t = gpt.transformer; self.k = k; self.head = torch.nn.Linear(768, 6)

    def forward(self, ids, last):
        """ids (B, T) right-padded prompts, last (B,) index of the last prompt token -> (B, k, 6) per-iteration logits."""
        T = ids.shape[1]; pos = torch.arange(T, device=ids.device)[None]
        h = self.t.wte(ids) + self.t.wpe(pos)
        def run(layer, h):                                             # causal attention; right padding never attends forward
            o = layer(h)
            return o[0] if isinstance(o, tuple) else o
        for L in range(0, 4):
            h = run(self.t.h[L], h)
        outs = []
        for _ in range(self.k):
            for L in range(4, 8):
                h = run(self.t.h[L], h)
            outs.append(self.head(self.t.ln_f(h[torch.arange(len(ids)), last])))
        # layers 8-11 are off the loss path (note 88-2): skipped to save compute; the loss is unchanged
        return torch.stack(outs, 1)


def loop_batch(tok, pairs, words, dev):
    enc = [tok.encode(prompt(s, words[w])) for s, w in pairs]; T = max(map(len, enc))
    ids = torch.full((len(enc), T), EOS); last = torch.tensor([len(e) - 1 for e in enc])
    for j, e in enumerate(enc):
        ids[j, :len(e)] = torch.tensor(e)
    tg = torch.tensor([prefix_states(s, words[w]) for s, w in pairs])
    return ids.to(dev), last.to(dev), tg.to(dev)


@torch.no_grad()
def loop_eval(m, tok, pairs, words, dev, bs=256):
    m.eval(); fin, per = 0, None
    for i in range(0, len(pairs), bs):
        ids, last, tg = loop_batch(tok, pairs[i:i + bs], words, dev)
        p = m(ids, last).argmax(-1)
        fin += int((p[:, -1] == tg[:, -1]).sum()); c = (p == tg).sum(0).cpu().numpy()
        per = c if per is None else per + c
    return fin / max(1, len(pairs)), (per / max(1, len(pairs))).tolist()


def train_loop(a):
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    sp = json.load(open(f"c1/p80_split_k{a.k}.json")); words = sp["words"]
    torch.manual_seed(a.seed); rng = random.Random(a.seed)
    tok = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")
    m = Looped(GPT2LMHeadModel.from_pretrained("openai-community/gpt2"), a.k).to(a.device)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.01)
    tr = [tuple(p) for p in sp["train_pairs"]]
    os.makedirs(a.out, exist_ok=True); json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w", newline="\n"), indent=1)
    log = open(os.path.join(a.out, "log.jsonl"), "w", newline="\n"); t0 = time.time()
    for step in range(1, a.steps + 1):
        m.train(); ids, last, tg = loop_batch(tok, [rng.choice(tr) for _ in range(a.bs)], words, a.device)
        lg = m(ids, last)
        if a.arm == "align":
            loss = torch.nn.functional.cross_entropy(lg.reshape(-1, 6), tg.reshape(-1))
        else:
            loss = torch.nn.functional.cross_entropy(lg[:, -1], tg[:, -1])
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0 or step == a.steps:
            log.write(json.dumps({"step": step, "loss": loss.item(), "sec": round(time.time() - t0)}) + "\n"); log.flush()
    unseen_b = [(s, w) for w in sp["unseen_words_b"] for s in range(6)]
    acc_tr, _ = loop_eval(m, tok, tr, words, a.device)
    acc_b, per_b = loop_eval(m, tok, unseen_b, words, a.device)
    acc_a, _ = loop_eval(m, tok, [tuple(p) for p in sp["heldout_pairs_a"]], words, a.device)
    res = {"section": 88, "arm": a.arm, "k": a.k, "seed": a.seed, "steps": a.steps, "train_sec": round(time.time() - t0),
           "acc_train": acc_tr, "acc_b_unseen_words": acc_b, "per_iteration_acc_b": per_b, "n_b": len(unseen_b),
           "acc_a_heldout_pairs": acc_a, "n_a": len(sp["heldout_pairs_a"]), "n_train_pairs": len(tr)}
    json.dump(res, open(os.path.join(a.out, "eval.json"), "w", newline="\n"), indent=1); print(res, flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["split", "train", "loop"])
    ap.add_argument("--arm", default="align", choices=["align", "end"], help="§88 loop arms")
    ap.add_argument("--k", type=int, default=2); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--steps", type=int, default=4000); ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.cmd == "split":
        for k in (2, 3):
            sp = make_split(k); json.dump(sp, open(f"c1/p80_split_k{k}.json", "w", newline="\n"))
            print(k, "words", len(sp["words"]), "unseen(b)", len(sp["unseen_words_b"]), "train pairs", len(sp["train_pairs"]),
                  "heldout(a)", len(sp["heldout_pairs_a"]))
    elif a.cmd == "loop":
        train_loop(a)
    else:
        train(a)


if __name__ == "__main__":
    main()
