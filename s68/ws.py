"""THEORY §68 WS arm: train and infer (HF transformers GPT-2 small, original weights).

Train:  python -m s68.ws train --train <gsm_train.json> --valid <gsm_valid.json> --out <dir> --seed 1 [--device cuda:0]
        Matched to TC (Coconut gsm_cot.yaml): AdamW lr 1e-4, wd 0.01, effective batch 128 (bs x grad_acc), fp32 unless
        --bf16. Budget: the processed tokens of 25 TC epochs (question + steps + answer + EOS per example), within
        +-10 %; validation (greedy WS inference on gsm_valid) after each TC-epoch-equivalent of tokens; the best
        checkpoint is kept (TC's save_only_improve). Writes model/ (best), train_log.jsonl, budget.json.
Infer:  ws_infer(model, tok, questions, device) -> answers, per-problem compute (tokens processed, passes, written).
"""
import argparse
import json
import math
import os
import random
import time

import torch

from s68.ws_data import calls, state_text, step_value, tc_text

NL = 198            # GPT-2 token for "\n"
EOS = 50256


def norm(s):
    s = str(s).replace(",", "").strip()
    try:
        f = float(s)
        return str(int(f)) if f.is_integer() else str(f)
    except ValueError:
        return s


@torch.no_grad()
def ws_infer(model, tok, questions, device, max_calls=10, max_new=48, bs=64):
    """Greedy WS inference; history cut between calls (each call re-encodes question + values so far)."""
    model.eval(); n = len(questions)
    vals = [[] for _ in range(n)]; ans = [None] * n; done = [False] * n
    comp = [{"tokens_processed": 0, "passes": 0, "written_tokens": 0, "calls": 0} for _ in range(n)]
    tok.padding_side = "left"
    for _ in range(max_calls):
        live = [i for i in range(n) if not done[i]]
        if not live:
            break
        for s in range(0, len(live), bs):
            b = live[s:s + bs]
            enc = tok([state_text(questions[i], vals[i]) for i in b], return_tensors="pt", padding=True).to(device)
            out = model.generate(**enc, max_new_tokens=max_new, do_sample=False, eos_token_id=[NL, EOS],
                                 pad_token_id=EOS)
            gen = out[:, enc["input_ids"].shape[1]:]
            for j, i in enumerate(b):
                g = [t for t in gen[j].tolist() if t != EOS]
                if NL in g:
                    g = g[:g.index(NL)]
                text = tok.decode(g).strip()
                L = int(enc["attention_mask"][j].sum())
                c = comp[i]; c["calls"] += 1; c["written_tokens"] += len(g) + 1
                c["passes"] += len(g) + 1; c["tokens_processed"] += L + len(g)
                if text.startswith("###"):
                    ans[i] = norm(text[3:]); done[i] = True
                else:
                    v = step_value(text)
                    if v is None:
                        ans[i] = None; done[i] = True
                    else:
                        vals[i].append(v)
    return ans, comp, vals


def load_json(p):
    return json.load(open(p, encoding="utf-8"))


def train(a):
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    torch.manual_seed(a.seed); random.seed(a.seed)
    tok = GPT2TokenizerFast.from_pretrained("openai-community/gpt2"); tok.pad_token = tok.eos_token
    model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").to(a.device)
    data = load_json(a.train); valid = load_json(a.valid)
    if a.max_train:
        data = data[:a.max_train]
    t0 = time.time(); items = []; tc_tokens = 0; skipped = 0
    for d in data:
        tc_tokens += len(tok.encode(tc_text(d["question"], d["steps"], d["answer"]))) + 1
        cs = calls(d["question"], d["steps"], d["answer"])
        if cs is None:
            skipped += 1; continue
        for x, y in cs:
            xi, yi = tok.encode(x), tok.encode(y) + [EOS]
            items.append((xi, yi))
    ws_tokens_epoch = sum(len(x) + len(y) for x, y in items)
    budget = 25 * tc_tokens
    eff = a.bs * a.grad_acc
    avg = ws_tokens_epoch / len(items)
    total_steps = math.ceil(budget / avg / eff) if not a.max_steps else a.max_steps
    os.makedirs(a.out, exist_ok=True)
    info = {"tc_tokens_per_epoch": tc_tokens, "tc_tokens_25_epochs": budget, "ws_calls_per_epoch": len(items),
            "ws_tokens_per_epoch": ws_tokens_epoch, "ws_epochs_equivalent": budget / ws_tokens_epoch,
            "optimizer_steps": total_steps, "effective_batch": eff, "skipped_examples": skipped,
            "prep_sec": round(time.time() - t0)}
    json.dump(info, open(os.path.join(a.out, "budget.json"), "w", newline="\n"), indent=1); print(info, flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    amp = torch.bfloat16 if a.bf16 else None
    log = open(os.path.join(a.out, "train_log.jsonl"), "w", newline="\n")
    order = list(range(len(items))); random.shuffle(order); ptr = 0
    seen_tokens = 0; best = -1.0; next_eval = tc_tokens; ev = 0; t0 = time.time()
    valid = valid[:a.max_valid] if a.max_valid else valid
    qv = [d["question"] for d in valid]; av = [norm(d["answer"]) for d in valid]
    for step in range(1, total_steps + 1):
        model.train()
        for _ in range(a.grad_acc):
            if ptr + a.bs > len(order):
                random.shuffle(order); ptr = 0
            b = [items[k] for k in order[ptr:ptr + a.bs]]; ptr += a.bs
            L = max(len(x) + len(y) for x, y in b)
            ids = torch.full((len(b), L), EOS); lab = torch.full((len(b), L), -100); att = torch.zeros(len(b), L, dtype=torch.long)
            for j, (x, y) in enumerate(b):
                s = x + y; ids[j, :len(s)] = torch.tensor(s); att[j, :len(s)] = 1; lab[j, len(x):len(s)] = torch.tensor(y)
                seen_tokens += len(s)
            with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
                loss = model(input_ids=ids.to(a.device), attention_mask=att.to(a.device), labels=lab.to(a.device)).loss
            (loss / a.grad_acc).backward()
        opt.step(); opt.zero_grad()
        if step % 100 == 0:
            log.write(json.dumps({"step": step, "loss": loss.item(), "tokens": seen_tokens, "sec": round(time.time() - t0)}) + "\n"); log.flush()
        if seen_tokens >= next_eval or step == total_steps:
            ev += 1; next_eval += tc_tokens
            pred, _, _ = ws_infer(model, tok, qv, a.device)
            acc = sum(p == y for p, y in zip(pred, av)) / len(av)
            rec = {"eval": ev, "step": step, "tokens": seen_tokens, "valid_acc": acc, "sec": round(time.time() - t0)}
            if acc > best:
                best = acc; model.save_pretrained(os.path.join(a.out, "model")); rec["saved"] = True
            log.write(json.dumps(rec) + "\n"); log.flush(); print(rec, flush=True)
    json.dump({**info, "best_valid": best, "tokens_seen": seen_tokens, "train_sec": round(time.time() - t0)},
              open(os.path.join(a.out, "budget.json"), "w", newline="\n"), indent=1)
    print("WS_DONE", a.out, best, flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["train"])
    ap.add_argument("--train", required=True); ap.add_argument("--valid", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--bs", type=int, default=32); ap.add_argument("--grad_acc", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--max_train", type=int, default=0, help="smoke / pilot only")
    ap.add_argument("--max_steps", type=int, default=0, help="smoke / pilot only")
    ap.add_argument("--max_valid", type=int, default=0, help="smoke / pilot only")
    a = ap.parse_args()
    train(a)


if __name__ == "__main__":
    main()
