"""PROTOCOL8 amendment 3 (secondary SC): candidate accuracy of a latent-arm checkpoint on p8_test and the cue-free
p8_na_test with one instrument (gpt2c.py rollout, addendum-G answer mask), answer depths >= 7, K' in {10, 16}.

    python prosqa/eval_na_p8.py --ckpt <checkpoint> --out <json> [--device cuda:0]
Rows: file, depth, margin at K' = 10 and 16 (log p(correct answer) - log p(other option)).
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import parse  # noqa: E402
from gpt2c import GPT2C, load_state  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--dmin", type=int, default=7)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    model = GPT2C(load_state(a.ckpt, a.device)); model.mask_answer = True
    rows = []
    for name in ("p8_test", "p8_na_test"):
        for x in json.load(open(os.path.join(HERE, "p8data", name + ".json"))):
            d = len(x["steps"])
            if d < a.dmin:
                continue
            P = parse(x)
            r = {"file": name, "depth": d}
            for K in (10, 16):
                lp, _, _ = model.rollout(P["ids"], K, P["answers"])
                r[f"m{K}"] = float(lp[0, 0] - lp[0, 1])
            rows.append(r)
            if len(rows) % 500 == 0:
                print(f"{len(rows)} problems", flush=True)
    json.dump({"ckpt": a.ckpt, "rows": rows}, open(a.out, "w"))
    print("DONE", a.out, len(rows))


if __name__ == "__main__":
    main()
