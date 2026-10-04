"""Transition coverage of the short (length 1-2) problems behind each C4-qualified result (Audit's request; descriptive).
A length-2 problem from the identity (e; a, b) labels the transition (state a, op b); a length-1 problem labels (e, a).
Coverage = distinct (state, op) pairs labelled / 14,400.
  §96 teacher (theory/c1_rl.py, mixP 0.05, M 240, 16k updates, bs 256): exact replay of its rng (length(), then ops).
  §103 R0 (boxes/self103.py): the R0 pool (rng 10300, 50k prefixes of S5=128 rows) and the batches each seed drew in R0
  (1,200 steps x 64; draw() with default_rng(seed)); assumes draw() is each seed rng's first use (as in the code).
    python theory/c4_coverage.py
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "theory")); sys.path.insert(0, ROOT)
import c1_rl  # noqa: E402


def teacher(seed, p=0.05, updates=16000, bs=256, A=120):
    c1_rl.MIX_P = p; rng = np.random.default_rng(seed); seen = set()
    for u in range(updates):
        L = c1_rl.length(rng, u, updates, "mixP")
        ops = rng.integers(0, A, (bs, L))
        if L == 1:
            seen.update(("e", int(a)) for a in ops[:, 0])
        elif L == 2:
            seen.update(zip(ops[:, 0].tolist(), ops[:, 1].tolist()))
    pairs = {x for x in seen if x[0] != "e"}; erow = {x for x in seen if x[0] == "e"}
    return {"pairs_from_len2": len(pairs) / 14400, "row_e_from_len1": len(erow) / 120,
            "coverage_all": (len(pairs) + len({x[1] for x in erow} - {b for (a, b) in pairs if a == 0})) / 14400}


def s103_r0():
    os.environ["BOXES_NB"] = "5"
    from boxes import s5
    from boxes import self103 as S
    Xtr, Ytr = s5.load_csv(os.path.join(ROOT, "bench", "data", "s5", "S5=128.csv"))
    rng = np.random.default_rng(10300); g = S.r0_items(rng, Xtr, Ytr)[0]
    pool2 = {(int(a), int(b)) for a, b in g[2][1]}
    out = {"pool_len2_coverage": len(pool2) / 14400, "pool_len2_items": int(len(g[2][1]))}
    for s in (1, 2, 3):
        r = np.random.default_rng(s); seen = set(); cls = sorted(g); w = np.array([len(g[c][0]) for c in cls], float)
        for _ in range(1200):
            cl = cls[int(r.choice(len(cls), p=w / w.sum()))]; ii = r.integers(0, len(g[cl][0]), 64)
            if cl == 2:
                seen.update((int(a), int(b)) for a, b in g[cl][1][ii])
        out["seed%d_trained_len2_coverage" % s] = len(seen) / 14400
    return out


def main():
    res = {"s96_teacher_mixP0.05": {s: teacher(s) for s in range(2161, 2166)}, "s103_R0": s103_r0()}
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "c4_coverage.json"), "w", newline="\n"), indent=1)


if __name__ == "__main__":
    main()
