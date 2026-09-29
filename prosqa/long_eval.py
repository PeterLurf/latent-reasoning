"""PROTOCOL6 addendum P: accuracy of Coconut GPT-2 checkpoints on ProsQA problems deeper than training (make_long.py).

    python long_eval.py --ckpt <ckpt> --data <long.json> --out <json> [--budgets 0,1,6,12] [--device cuda:0] [--ext 0,2,4,6]

Candidate accuracy (log p(correct answer) > log p(other option), as analyze.kprime) per thought budget K', broken down
by extension m and by answer-path length. All checkpoints are evaluated with --mask_answer semantics (G and J models).
"""
import argparse
import json

import numpy as np
import torch

from analyze import parse
from gpt2c import GPT2C, load_state

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--budgets", default="0,1,6,12"); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--ext", default="0,2,4,6"); ap.add_argument("--n", type=int, default=None)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    model = GPT2C(load_state(a.ckpt, a.device)); model.mask_answer = True
    exts = set(map(int, a.ext.split(",")))
    data = [x for x in json.load(open(a.data)) if x["ext"] in exts][: a.n]
    out = {"ckpt": a.ckpt, "data": a.data, "n": len(data), "budgets": {}}
    for Kp in map(int, a.budgets.split(",")):
        rows = []
        for x in data:
            P = parse(x)
            lp, _, _ = model.rollout(P["ids"], Kp, P["answers"])
            rows.append((x["ext"], len(x["steps"]), float(lp[0, 0] - lp[0, 1])))
        R = np.array(rows)
        grp = lambda col: {int(v): float((R[R[:, col] == v, 2] > 0).mean()) for v in sorted(set(R[:, col]))}
        out["budgets"][Kp] = {"acc": float((R[:, 2] > 0).mean()), "by_ext": grp(0), "by_path_len": grp(1),
                              "margins": R[:, 2].round(4).tolist(), "ext": R[:, 0].astype(int).tolist(),
                              "path_len": R[:, 1].astype(int).tolist()}
        print("K'", Kp, {k: v for k, v in out["budgets"][Kp].items() if k in ("acc", "by_ext")}, flush=True)
    json.dump(out, open(a.out, "w"))
    print("DONE", a.out, flush=True)
