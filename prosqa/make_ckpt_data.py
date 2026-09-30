"""PROTOCOL8b step 1: training data that contains written checkpoints.

    python make_ckpt_data.py --inp p8data/p8_train.json --out p8data/p8_train_ckpt.json [--p 0.5] [--seed 0]

For each problem, with probability p, insert the statement "<root> is a <Y>." where Y is the node reached after k gold
hops (the object of the k-th gold-path statement), k ~ U{1..depth-1}. It goes at a random position among the
statements, and its edge is added. The answer and steps are unchanged, so the label is still only the final answer
(C4-compatible): the checkpoint is a correct shortcut the model can use, never a required one. Problems with depth 1
are left unchanged. Prints the md5 of the output.
"""
import argparse
import hashlib
import json
import re

import numpy as np

STMT = re.compile(r"^(?:Every (\w+)|(\w+)) is an? (\w+)\.$")


def insert(x, rng):
    parts = re.split(r"(?<=\.) ", x["question"])
    stmts, q = parts[:-1], parts[-1]
    depth = len(x["steps"])
    if depth < 2:
        return None
    k = int(rng.integers(1, depth))                                  # 1..depth-1 hops in
    y = STMT.match(x["steps"][k - 1]).group(3)
    sym = x["idx_to_symbol"]; ix = {s: i for i, s in enumerate(sym)}
    root = sym[x["root"]]
    art = "an" if y[:1] in "aeiou" else "a"
    new = f"{root} is {art} {y}."
    pos = int(rng.integers(0, len(stmts) + 1))
    stmts = stmts[:pos] + [new] + stmts[pos:]
    return {**x, "question": " ".join(stmts) + " " + q, "edges": x["edges"] + [[x["root"], ix[y]]],
            "ckpt_hops": k, "ckpt_pos": pos}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--p", type=float, default=0.5); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    d = json.load(open(a.inp))
    out, n = [], 0
    for x in d:
        y = insert(x, rng) if rng.random() < a.p else None
        if y is not None:
            n += 1
        out.append(y if y is not None else {**x, "ckpt_hops": 0})
    with open(a.out, "w", newline="\n") as f:
        json.dump(out, f)
    print(a.out, len(out), "problems;", n, "with a written checkpoint; md5", hashlib.md5(open(a.out, "rb").read()).hexdigest()[:10])


if __name__ == "__main__":
    main()
