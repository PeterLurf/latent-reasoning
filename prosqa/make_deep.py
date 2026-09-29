"""PROTOCOL8 Phase 0: "deep ProsQA" — ProsQA problems made deeper by hanging chains of m new concepts off both options
(make_long.extend: the answer path becomes the original path + m chain statements, and only the base of each option's
chain decides the answer). Base problems are the short ones (original path 3-4 hops), so depth is driven by m.

    python make_deep.py --src data --out p8data [--seed 0]

  p8_train.json  12000 train problems (seeded subsample of the short ones), one m ~ U{0..6} each (depth 3-10)
  p8_valid.json    100 valid problems, m ~ U{0..6} (Coconut's per-epoch validation; kept small, it generates)
  p8_test.json   every short test problem x m = 0..10 (m 7-10 deeper than any training problem)
  p8_ctl.json    every short test problem x m = 4, 8 with the chains hung off non-option concepts (--control): same
                 input length, original answer depth
Prints md5 of each file (recorded in the pre-registration).
"""
import argparse
import hashlib
import json
import os
from collections import Counter

import numpy as np

import make_long


def short(d):
    return [x for x in d if len(x["steps"]) <= 4]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data"); ap.add_argument("--out", default="p8data")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--n_train", type=int, default=12000)
    ap.add_argument("--n_valid", type=int, default=100); ap.add_argument("--m_train", type=int, default=6)
    ap.add_argument("--m_test", type=int, default=10)
    a = ap.parse_args()
    load = lambda s: json.load(open(os.path.join(a.src, f"prosqa_{s}.json")))
    tr, va, te = load("train"), load("valid"), load("test")
    make_long.NAMES = sorted({s for x in tr for s in x["idx_to_symbol"] if s[:1].islower()})
    rng = np.random.default_rng(a.seed)
    os.makedirs(a.out, exist_ok=True)

    def mixed(d, n):
        d = short(d)
        idx = rng.permutation(len(d))
        out = []
        for i in idx:
            y = make_long.extend(d[i], int(rng.integers(0, a.m_train + 1)), rng)
            if y is not None:
                out.append(y)
            if len(out) == n:
                break
        return out

    def grid(d, ms, control=False):
        out, skip = [], Counter()
        for x in short(d):
            for m in ms:
                y = make_long.extend(x, m, rng, control)
                (out.append(y) if y is not None else skip.update([m]))
        return out, skip

    files = {"p8_train.json": mixed(tr, a.n_train), "p8_valid.json": mixed(va, a.n_valid)}
    files["p8_test.json"], s1 = grid(te, range(a.m_test + 1))
    files["p8_ctl.json"], s2 = grid(te, [4, 8], control=True)
    for f, d in files.items():
        p = os.path.join(a.out, f)
        with open(p, "w", newline="\n") as fh:
            json.dump(d, fh)
        md5 = hashlib.md5(open(p, "rb").read()).hexdigest()[:10]
        print(f, len(d), "md5", md5, "by m", sorted(Counter(y["ext"] for y in d).items()),
              "depth", sorted(Counter(len(y["steps"]) for y in d).items()))
    print("skipped (too few unused names) test", dict(s1), "ctl", dict(s2))


if __name__ == "__main__":
    main()
