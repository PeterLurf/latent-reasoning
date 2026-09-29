"""PROTOCOL8 Phase 0: candidate accuracy of GPT-2 checkpoints trained WITHOUT continuous thoughts, by extension m.

    python eval_p8.py --ckpt <ckpt> --data p8data/p8_test.json --mode nocot|cot --out <json> [--n_per_m 150]

nocot: the model reads question + "\\n" and the two answers "### <root> is a <option>." are scored directly
       (Coconut no_cot format: question tokens then answer tokens). One-pass capacity.
cot:   greedy generation of the reasoning steps (no KV cache; stops at the "###" answer marker or --max_new tokens),
       then both answers are scored after the generated steps (Coconut cot format). For the stage-0 (language CoT)
       checkpoint of a Coconut run, as a descriptive reference.
Accuracy = log p(correct) > log p(other), as long_eval.py; reported by m and by answer-path length.
"""
import argparse
import json

import numpy as np
import torch

from analyze import parse
from gpt2c import GPT2C, load_state

ANS = 21017   # "###"
EOS = 50256


@torch.no_grad()
def score(model, ids, answers):
    dev = model.wte.device
    out = []
    for ans in answers:
        x = model.embed(torch.tensor(ids + ans, device=dev))[None]
        _, logits, _ = model.forward(x)
        l = logits[0].log_softmax(-1)
        n0 = len(ids)
        out.append(float(l[n0 - 1:n0 - 1 + len(ans)].gather(-1, torch.tensor(ans, device=dev)[:, None]).sum()))
    return out


@torch.no_grad()
def generate(model, ids, max_new):
    dev = model.wte.device
    seq = list(ids)
    for _ in range(max_new):
        _, logits, _ = model.forward(model.embed(torch.tensor(seq, device=dev))[None])
        t = int(logits[0, -1].argmax())
        if t in (ANS, EOS):
            break
        seq.append(t)
    return seq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["nocot", "cot"], required=True)
    ap.add_argument("--n_per_m", type=int, default=None); ap.add_argument("--max_new", type=int, default=220)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    model = GPT2C(load_state(a.ckpt, a.device))
    data, seen = [], {}
    for x in json.load(open(a.data)):
        if a.n_per_m is None or seen.get(x["ext"], 0) < a.n_per_m:
            data.append(x); seen[x["ext"]] = seen.get(x["ext"], 0) + 1
    rows = []
    for x in data:
        P = parse(x)
        ids = P["ids"] if a.mode == "nocot" else generate(model, P["ids"], a.max_new)
        lp = score(model, ids, P["answers"])
        rows.append((x["ext"], len(x["steps"]), lp[0] - lp[1]))
    R = np.array(rows)
    grp = lambda col: {int(v): float((R[R[:, col] == v, 2] > 0).mean()) for v in sorted(set(R[:, col]))}
    cnt = lambda col: {int(v): int((R[:, col] == v).sum()) for v in sorted(set(R[:, col]))}
    out = {"ckpt": a.ckpt, "data": a.data, "mode": a.mode, "n": len(rows), "acc": float((R[:, 2] > 0).mean()),
           "by_ext": grp(0), "n_by_ext": cnt(0), "by_path_len": grp(1), "n_by_path_len": cnt(1),
           "margins": R[:, 2].round(4).tolist(), "ext": R[:, 0].astype(int).tolist()}
    json.dump(out, open(a.out, "w"))
    print(a.mode, "acc", round(out["acc"], 3), "by_ext", {k: round(v, 3) for k, v in out["by_ext"].items()}, flush=True)
    print("DONE", a.out, flush=True)


if __name__ == "__main__":
    main()
