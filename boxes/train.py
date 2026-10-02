"""THEORY §66: train one arm from original GPT-2 small, then (WC arms) the written-state edit test.

    python -m boxes.train --arm {nc,tc,tc2,cl,wc,wc0} [--k 4] --seed 1 --steps 3000 --bs 32 --out boxes/runs/<name> [--device cuda:0]
Data: every step draws one shape (N, question type) and bs fresh random problems of that shape; the same number of
problems (steps x bs) for every arm. NC/TC/TC2/CL: N ~ U{1..8}, question all/where 50/50. WC/WC0 solver: k' ~ U{1..k},
random start, the end arrangement as the answer (no intermediate labels). Writes model.pt, args.json, log.jsonl and,
for WC/WC0, eval_edit.json (P4: replace a written arrangement mid-chain by a random one; the answer must follow the true
composition from it; N in {16, 24, 32}, "all" questions).
"""
import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys  # noqa: E402
sys.path.insert(0, ROOT)
from boxes.core import LET, GPT2KV, apply, infer, loss_batch, make_problem, truth  # noqa: E402


def load_model(path, dev):
    sd = torch.load(path, map_location="cpu", weights_only=True)
    sd = {k: v for k, v in sd.items()}
    if any(k.startswith("p.") for k in sd):                     # a saved GPT2KV state dict
        base = torch.load(os.path.join(ROOT, "prosqa", "ckpt", "gpt2_pretrained.bin"), map_location="cpu", weights_only=True)
        m = GPT2KV(base); m.load_state_dict(sd)
    else:
        m = GPT2KV(sd)
    return m.to(dev)


def train(a):
    TEST_SEED = 6600                                             # bench/tasks.py boxes_load (amendment 66-1)
    assert a.seed != TEST_SEED and not (10_000 <= TEST_SEED), "training generator seed collides with the test seed"
    dev = a.device; torch.manual_seed(a.seed); rng = random.Random(a.seed)
    m = load_model(a.init, dev)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.01)
    amp = torch.bfloat16 if dev.startswith("cuda") and torch.cuda.is_bf16_supported() else None
    os.makedirs(a.out, exist_ok=True)
    json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w", newline="\n"), indent=1)
    log = open(os.path.join(a.out, "log.jsonl"), "w", newline="\n"); t0 = time.time(); hist = []
    for step in range(1, a.steps + 1):
        for g in opt.param_groups:
            g["lr"] = a.lr * min(1, step / 100) * 0.5 * (1 + math.cos(math.pi * step / a.steps))
        if a.arm in ("wc", "wc0"):
            n = rng.randint(1, a.k); qt = "all"
        else:
            n = rng.randint(1, 8); qt = rng.choice(["all", "where"])
        batch = [make_problem(rng, n, qt) for _ in range(a.bs)]
        with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
            loss = loss_batch(m, a.arm, a.k, batch, dev)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        hist.append(loss.item())
        if step % 100 == 0 or step == a.steps:
            rec = {"step": step, "loss": float(np.mean(hist)), "sec": round(time.time() - t0)}
            hist = []
            if step % 500 == 0 or step == a.steps:                # quick in-range check: N = 8, "all"
                vr = random.Random(10_000 + step); vb = [make_problem(vr, 8, "all") for _ in range(64)]
                m.eval()
                with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
                    txt, _, _ = infer(m, a.arm, a.k, vb, dev)
                m.train()
                rec["acc_N8_all"] = float(np.mean([t == truth(p) for t, p in zip(txt, vb)]))
            log.write(json.dumps(rec) + "\n"); log.flush(); print(rec, flush=True)
    torch.save(m.state_dict(), os.path.join(a.out, "model.pt"))
    last = [json.loads(l) for l in open(os.path.join(a.out, "log.jsonl"))]
    json.dump({"steps": a.steps, "bs": a.bs, "sec": round(time.time() - t0, 1), "sec_per_step": round((time.time() - t0) / a.steps, 4),
               "final_loss": last[-1]["loss"], "acc_N8_all": [r.get("acc_N8_all") for r in last if "acc_N8_all" in r]},
              open(os.path.join(a.out, "eval_train.json"), "w", newline="\n"), indent=1)
    if a.arm in ("wc", "wc0"):
        edit_eval(m, a, dev, amp)
    print("TRAIN_DONE", a.out, f"{time.time() - t0:.0f}s", flush=True)


@torch.no_grad()
def edit_eval(m, a, dev, amp):
    m.eval(); rng = random.Random(6606); out = {}
    for N in (16, 24, 32):
        batch = [make_problem(rng, N, "all") for _ in range(64)]
        nch = math.ceil(N / a.k); c = nch // 2 - 1                    # replace the arrangement written after chunk c
        new = []
        for _ in batch:
            x = list(LET); rng.shuffle(x); new.append("".join(x))
        with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
            txt, _, _ = infer(m, a.arm, a.k, batch, dev, edit={"chunk": c, "arr": new})
        rest = [(c + 1) * a.k] * len(batch)
        want = [" " + " ".join(apply(r, p["ops"][s:])) for r, p, s in zip(new, batch, rest)]
        out[str(N)] = float(np.mean([t == w for t, w in zip(txt, want)]))
    out["IIA"] = float(np.mean([out[k] for k in ("16", "24", "32")]))
    json.dump(out, open(os.path.join(a.out, "eval_edit.json"), "w", newline="\n"), indent=1)
    print("EDIT", out, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["nc", "tc", "tc2", "cl", "wc", "wc0"])
    ap.add_argument("--k", type=int, default=4); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--init", default=os.path.join(ROOT, "prosqa", "ckpt", "gpt2_pretrained.bin"))
    ap.add_argument("--out", required=True)
    train(ap.parse_args())


if __name__ == "__main__":
    main()
