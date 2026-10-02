"""GSM8K length split prep (no training before registration): does the Dilgren & Wiegreffe GPT-2 CoT checkpoint
(2604.04902, MIT; checkpoint_25, benched at 41.8 on GSM8k-Aug test) fine-tune in our pipeline in bf16, and how long
is an epoch of the short-only subset?
* Loads the plain HF GPT2LMHeadModel state dict (strict; the attention-mask buffers are not parameters).
* Sanity: greedy accuracy on the first --n_eval test items before any update (CoT format, answer after "###").
* Fine-tune timing: CoT sequences question + "\\n" + steps joined by "\\n" + "\\n### " + answer + EOS (the Coconut
  repo's stage-0 format), loss on everything after the question, bf16 autocast, AdamW, --updates optimiser steps on
  short-only items (step count <= --short_max, gsmlen/step_counts.json). Reports seconds per step, tokens per
  second, the projected time for one short-only epoch, and that the loss is finite and falls.
    python gsmlen/ft_check.py --device cuda:0 --bs 32 --grad_acc 1 --updates 60 --n_eval 200 --out gsmlen/ft_check_<gpu>.json
"""
import argparse
import json
import random
import re
import time

import torch

CK = "prosqa/ckpt_ext/hf/dw_gsm/cot_checkpoint_25"
DATA = "D:/ext/coconut/data"


def norm(s):
    s = s.strip().replace(",", "")
    try:
        f = float(s); return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return s


@torch.no_grad()
def greedy_acc(model, tok, items, dev, bs=32, max_new=128):
    model.eval(); ok = 0; tok.padding_side = "left"
    for i in range(0, len(items), bs):
        b = items[i:i + bs]
        enc = tok([x["question"] + "\n" for x in b], return_tensors="pt", padding=True).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.startswith("cuda")):
            out = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
        for x, o in zip(b, out):
            t = tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True)
            m = re.search(r"###\s*([^\n]*)", t)
            ok += int(m is not None and norm(m.group(1)) == norm(x["answer"]))
    return ok / len(items)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--grad_acc", type=int, default=1); ap.add_argument("--updates", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=5); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--short_max", type=int, default=2); ap.add_argument("--n_eval", type=int, default=200)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("openai-community/gpt2"); tok.pad_token = tok.eos_token
    model = GPT2LMHeadModel(GPT2Config.from_pretrained("openai-community/gpt2"))
    sd = torch.load(CK, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    model.to(a.device)
    res = {"checkpoint": CK, "missing": missing, "unexpected": unexpected, "device": a.device,
           "gpu": torch.cuda.get_device_name(a.device) if a.device.startswith("cuda") else "cpu"}
    test = json.load(open(f"{DATA}/gsm_test.json", encoding="utf-8"))
    if a.n_eval:
        t0 = time.time(); res["test_acc_first_n"] = greedy_acc(model, tok, test[:a.n_eval], a.device)
        res["n_eval"] = a.n_eval; res["eval_sec"] = round(time.time() - t0, 1); print(res, flush=True)
    counts = json.load(open("gsmlen/step_counts.json"))["train"]
    train = json.load(open(f"{DATA}/gsm_train.json", encoding="utf-8"))
    short = [x for x, n in zip(train, counts) if 1 <= n <= a.short_max]
    del train
    random.seed(0); random.shuffle(short)
    enc = []
    for x in short[: a.bs * a.grad_acc * (a.updates + a.warmup)]:
        q = tok.encode(x["question"] + "\n")
        r = tok.encode("\n".join(s for s in x["steps"] if s.strip()) + "\n### " + x["answer"]) + [tok.eos_token_id]
        enc.append((q, r))
    tok_per_item = sum(len(q) + len(r) for q, r in enc) / len(enc)
    model.train(); opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    losses = []; ptr = 0; t_start = None; ntok = 0
    for u in range(a.warmup + a.updates):
        if u == a.warmup:
            if a.device.startswith("cuda"):
                torch.cuda.synchronize(a.device)
            t_start = time.time(); ntok = 0
        for _ in range(a.grad_acc):
            b = enc[ptr:ptr + a.bs]; ptr += a.bs
            L = max(len(q) + len(r) for q, r in b)
            ids = torch.full((len(b), L), tok.eos_token_id); lab = torch.full((len(b), L), -100); att = torch.zeros(len(b), L, dtype=torch.long)
            for j, (q, r) in enumerate(b):
                s = q + r; ids[j, :len(s)] = torch.tensor(s); att[j, :len(s)] = 1; lab[j, len(q):len(s)] = torch.tensor(r)
                ntok += len(s)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.device.startswith("cuda")):
                loss = model(input_ids=ids.to(a.device), attention_mask=att.to(a.device), labels=lab.to(a.device)).loss
            (loss / a.grad_acc).backward()
        opt.step(); opt.zero_grad(); losses.append(loss.item())
    if a.device.startswith("cuda"):
        torch.cuda.synchronize(a.device)
    dt = time.time() - t_start; sps = dt / a.updates
    items_per_epoch = len(short); steps_per_epoch = items_per_epoch / (a.bs * a.grad_acc)
    res.update({"short_max": a.short_max, "short_items": items_per_epoch, "tokens_per_item": round(tok_per_item, 1),
                "bs": a.bs, "grad_acc": a.grad_acc, "bf16_autocast": a.device.startswith("cuda"), "sec_per_step": round(sps, 4),
                "tokens_per_sec": round(ntok / dt), "epoch_hours_short_only": round(steps_per_epoch * sps / 3600, 3),
                "loss_first5": [round(x, 4) for x in losses[:5]], "loss_last5": [round(x, 4) for x in losses[-5:]],
                "finite": all(x == x for x in losses),
                "peak_mem_gb": round(torch.cuda.max_memory_allocated(a.device) / 2**30, 2) if a.device.startswith("cuda") else None})
    json.dump(res, open(a.out, "w", newline="\n"), indent=1); print(res, flush=True)


if __name__ == "__main__":
    main()
