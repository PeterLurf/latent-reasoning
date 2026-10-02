"""THEORY §78 (75ec955): train TC / WS / WS-pad from base GPT-2 small on the short split, select by short-valid accuracy,
and predict the test set (and the secondary aug_long set).
    python gsmlen/train78.py train --arm {tc,ws,wspad} --seed 1 --epochs E --bs 32 --device cuda:0 --out gsmlen/runs/<name>
    python gsmlen/train78.py predict --arm ... --ckpt <run>/model --split test --out <run>/pred_test.json
Formats (fixed before any run; reply to Theory 2026-10-02 11:40):
* TC: question + "\\n" + steps joined by "\\n" + "\\n#### " + answer + EOS; loss on everything after the question.
* WS (§68 state format, s68/ws_data.py): input question + "\\nvalues so far:" + " v1 v2 ..." + "\\n"; target the next
  "<<expr=value>>" + "\\n", or "#### " + answer + "\\n", then EOS. History cut; inference <= 12 calls.
* WS-pad: as WS, but each training call's values list is padded to L ~ Uniform{len(true), ..., 8} with distractor
  numbers (drawn with replacement from the question's numbers and the earlier true values), inserted at random
  positions; the true values keep their order. Fresh padding each time a call is drawn. Inference is unpadded.
Training: AdamW lr 1e-4, wd 0.01, constant lr, bf16 autocast (--fp32 to disable), effective batch --bs * --grad_acc,
--epochs passes over the 206,397 short items (every arm sees each labelled step once per epoch), short-valid greedy
accuracy after every epoch (and every --eval_every steps if set), best checkpoint kept. --max_steps: throughput pilot.
Compute per problem (prediction files): tokens_processed (prompt + generated, KV cache within a call; WS re-encodes its
state at every call), forward passes, written tokens, calls.
"""
import argparse
import json
import os
import random
import re
import time

import torch

NL, EOS = 198, 50256
NUMRE = re.compile(r"\d+(?:\.\d+)?")


def norm(s):
    s = str(s).replace(",", "").strip().rstrip(".")
    try:
        f = float(s); return str(int(f)) if f.is_integer() else str(f)
    except ValueError:
        return s


def step_value(step):
    m = re.search(r"=([^=<>]*)>>\s*$", step.strip())
    return m.group(1).strip() if m else None


def state_text(q, values):
    return q + "\nvalues so far:" + "".join(" " + v for v in values) + "\n"


def ws_calls(x):
    out, vals = [], []
    for s in x["steps"]:
        v = step_value(s)
        if v is None:
            return None
        out.append((list(vals), s + "\n")); vals.append(v)
    out.append((list(vals), "#### " + x["answer"] + "\n"))
    return out


def pad_values(q, vals, rng, maxlen=8):
    L = rng.randint(len(vals), max(len(vals), maxlen))
    pool = NUMRE.findall(q) + list(vals)
    if not pool or L == len(vals):
        return list(vals)
    out = list(vals)
    for _ in range(L - len(vals)):
        out.insert(rng.randint(0, len(out)), rng.choice(pool))
    return out


class Data:
    def __init__(self, arm, items, tok, seed):
        self.arm, self.items, self.tok, self.rng = arm, items, tok, random.Random(seed)
        self.qtok = [None] * len(items)
        if arm == "tc":
            self.units = list(range(len(items)))
        else:
            self.calls = [ws_calls(x) for x in items]
            self.units = [(i, k) for i, c in enumerate(self.calls) if c for k in range(len(c))]
            self.ttok = {}

    def example(self, u):
        tok = self.tok
        if self.arm == "tc":
            x = self.items[u]
            q = tok.encode(x["question"] + "\n")
            r = tok.encode("\n".join(x["steps"]) + "\n#### " + x["answer"]) + [EOS]
            return q, r
        i, k = u; vals, tgt = self.calls[i][k]; x = self.items[i]
        if self.arm == "wspad":
            vals = pad_values(x["question"], vals, self.rng)
        return tok.encode(state_text(x["question"], vals)), tok.encode(tgt) + [EOS]


@torch.no_grad()
def predict_tc(model, tok, qs, dev, bs=48, max_new=320):
    model.eval(); tok.padding_side = "left"; ans, comp = [], []
    for s in range(0, len(qs), bs):
        enc = tok([q + "\n" for q in qs[s:s + bs]], return_tensors="pt", padding=True).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.startswith("cuda")):
            out = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=EOS, eos_token_id=EOS)
        for j, o in enumerate(out):
            g = [t for t in o[enc["input_ids"].shape[1]:].tolist() if t != EOS]
            t = tok.decode(g); m = re.search(r"####\s*([^\n]*)", t)
            ans.append(norm(m.group(1)) if m else None)
            L = int(enc["attention_mask"][j].sum())
            comp.append({"tokens_processed": L + len(g), "passes": len(g) + 1, "written_tokens": len(g) + 1, "calls": 1})
    return ans, comp, [None] * len(qs)


@torch.no_grad()
def predict_ws(model, tok, qs, dev, max_calls=12, max_new=48, bs=64):
    model.eval(); tok.padding_side = "left"; n = len(qs)
    vals = [[] for _ in range(n)]; ans = [None] * n; done = [False] * n
    comp = [{"tokens_processed": 0, "passes": 0, "written_tokens": 0, "calls": 0} for _ in range(n)]
    for _ in range(max_calls):
        live = [i for i in range(n) if not done[i]]
        if not live:
            break
        for s in range(0, len(live), bs):
            b = live[s:s + bs]
            enc = tok([state_text(qs[i], vals[i]) for i in b], return_tensors="pt", padding=True).to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.startswith("cuda")):
                out = model.generate(**enc, max_new_tokens=max_new, do_sample=False, eos_token_id=[NL, EOS], pad_token_id=EOS)
            for j, i in enumerate(b):
                g = [t for t in out[j, enc["input_ids"].shape[1]:].tolist() if t != EOS]
                if NL in g:
                    g = g[:g.index(NL)]
                text = tok.decode(g).strip(); L = int(enc["attention_mask"][j].sum())
                c = comp[i]; c["calls"] += 1; c["written_tokens"] += len(g) + 1; c["passes"] += len(g) + 1
                c["tokens_processed"] += L + len(g)
                if text.startswith("####"):
                    ans[i] = norm(text[4:]); done[i] = True
                else:
                    v = step_value(text)
                    if v is None:
                        done[i] = True
                    else:
                        vals[i].append(v)
    return ans, comp, [len(v) for v in vals]


def predict(arm, model, tok, items, dev):
    qs = [x["question"] for x in items]
    return predict_tc(model, tok, qs, dev) if arm == "tc" else predict_ws(model, tok, qs, dev)


def acc(pred, items):
    return sum(p is not None and p == norm(x["answer"]) for p, x in zip(pred, items)) / len(items)


def train(a):
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    torch.manual_seed(a.seed); random.seed(a.seed)
    tok = GPT2TokenizerFast.from_pretrained("openai-community/gpt2"); tok.pad_token = tok.eos_token
    model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").to(a.device)
    items = json.load(open("gsmlen/d78/train_short.json", encoding="utf-8"))
    valid = json.load(open("gsmlen/d78/valid_short.json", encoding="utf-8"))
    valid = valid[:a.max_valid] if a.max_valid else valid                # smoke only
    data = Data(a.arm, items, tok, a.seed)
    os.makedirs(a.out, exist_ok=True)
    json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w", newline="\n"), indent=1)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    log = open(os.path.join(a.out, "train_log.jsonl"), "w", newline="\n")
    eff = a.bs * a.grad_acc; steps_per_epoch = len(data.units) // eff
    total = a.max_steps or a.epochs * steps_per_epoch
    info = {"units_per_epoch": len(data.units), "steps_per_epoch": steps_per_epoch, "total_steps": total, "eff_batch": eff}
    if a.arm != "tc":                                                   # Theory: the range gap on record
        info["frac_states_true_len_ge3"] = sum(len(data.calls[i][k][0]) >= 3 for i, k in data.units) / len(data.units)
        info["true_len_dist"] = {str(L): sum(len(data.calls[i][k][0]) == L for i, k in data.units) for L in range(4)}
    print(info, flush=True); log.write(json.dumps(info) + "\n")
    order, ptr, best, t0, ntok = [], 0, -1.0, time.time(), 0
    amp = not a.fp32 and a.device.startswith("cuda")
    for step in range(1, total + 1):
        model.train()
        for _ in range(a.grad_acc):
            if ptr + a.bs > len(order):
                order = list(range(len(data.units))); random.shuffle(order); ptr = 0
            b = [data.example(data.units[k]) for k in order[ptr:ptr + a.bs]]; ptr += a.bs
            L = max(len(q) + len(r) for q, r in b)
            ids = torch.full((len(b), L), EOS); lab = torch.full((len(b), L), -100); att = torch.zeros(len(b), L, dtype=torch.long)
            for j, (q, r) in enumerate(b):
                s = q + r; ids[j, :len(s)] = torch.tensor(s); att[j, :len(s)] = 1; lab[j, len(q):len(s)] = torch.tensor(r)
                ntok += len(s)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                loss = model(input_ids=ids.to(a.device), attention_mask=att.to(a.device), labels=lab.to(a.device)).loss
            (loss / a.grad_acc).backward()
        opt.step(); opt.zero_grad()
        if step % 100 == 0 or step == total:
            log.write(json.dumps({"step": step, "loss": loss.item(), "tokens": ntok, "sec": round(time.time() - t0)}) + "\n"); log.flush()
        if (a.max_steps == 0 and step % steps_per_epoch == 0) or step == total or (a.eval_every and step % a.eval_every == 0):
            te = time.time(); pred, _, _ = predict(a.arm, model, tok, valid, a.device); v = acc(pred, valid)
            rec = {"step": step, "epoch": round(step / steps_per_epoch, 3), "valid_short_acc": v, "eval_sec": round(time.time() - te),
                   "sec": round(time.time() - t0)}
            if v > best and not a.max_steps:
                best = v; model.save_pretrained(os.path.join(a.out, "model")); rec["saved"] = True
            log.write(json.dumps(rec) + "\n"); log.flush(); print(rec, flush=True)
    if a.max_steps:                                                     # pilot / smoke: keep the final model for the predict path
        model.save_pretrained(os.path.join(a.out, "model"))
    res = {**info, "best_valid_short": best, "train_sec": round(time.time() - t0), "tokens": ntok,
           "sec_per_step": round((time.time() - t0) / total, 4)}
    json.dump(res, open(os.path.join(a.out, "train_done.json"), "w", newline="\n"), indent=1); print("DONE", res, flush=True)


def run_predict(a):
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("openai-community/gpt2"); tok.pad_token = tok.eos_token
    model = GPT2LMHeadModel.from_pretrained(a.ckpt).to(a.device)
    items = json.load(open(f"gsmlen/d78/{a.split}.json", encoding="utf-8"))
    items = items[:a.max_valid] if a.max_valid else items                # smoke only
    t0 = time.time(); pred, comp, nvals = predict(a.arm, model, tok, items, a.device)
    rows = [{"i": i, "pred": p, "gold": norm(x["answer"]), "correct": p is not None and p == norm(x["answer"]), "n": x["n"],
             "depth": x["depth"], "bin": x.get("bin"), "values_len": nv, **c}
            for i, (x, p, c, nv) in enumerate(zip(items, pred, comp, nvals))]
    json.dump({"arm": a.arm, "ckpt": a.ckpt, "split": a.split, "sec": round(time.time() - t0), "acc": acc(pred, items), "rows": rows},
              open(a.out, "w", newline="\n"))
    print(a.split, "acc", acc(pred, items), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "predict"]); ap.add_argument("--arm", required=True, choices=["tc", "ws", "wspad"])
    ap.add_argument("--seed", type=int, default=1); ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--bs", type=int, default=32); ap.add_argument("--grad_acc", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--max_steps", type=int, default=0, help="throughput pilot only"); ap.add_argument("--eval_every", type=int, default=0); ap.add_argument("--max_valid", type=int, default=0, help="smoke only")
    ap.add_argument("--ckpt"); ap.add_argument("--split", default="test")
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    train(a) if a.cmd == "train" else run_predict(a)


if __name__ == "__main__":
    main()
