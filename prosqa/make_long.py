"""PROTOCOL6 addendum P: ProsQA problems deeper than any training problem (training paths are 3-6 hops).

    python make_long.py <in.json> <out.json> --m 0,2,4,6 [--seed 0]

For each test problem and each extension m, two fresh linear chains of m new concepts (names from the training
vocabulary that the problem does not use) are hung from the correct option (target -> c1 -> ... -> cm) and from the
wrong option (neg_target -> d1 -> ... -> dm). Their 2m statements are inserted at random positions among the existing
ones; the question asks "Is <root> a cm or dm?" in the original option order, the answer path is the original path plus
the m chain statements, so path length = original + m. Both options sit at the end of identical-looking chains, so only
the base of each chain (reachable or not from the root) decides the answer. m = 0 reproduces the problem unchanged.
Problems with fewer than 2m unused concept names are skipped for that m (reported).
--control: the same 2m new statements, but the two chains hang from two random concepts other than the options (and
off the answer path); question, answer and path are unchanged. Separates "longer input" from "deeper answer".
"""
import argparse
import json
import re

import numpy as np

NAMES = None


def extend(x, m, rng, control=False):
    if m == 0:
        return dict(x, ext=0, orig_len=len(x["steps"]))
    sym = list(x["idx_to_symbol"])
    free = [s for s in NAMES if s not in sym]
    if len(free) < 2 * m:
        return None
    pick = list(rng.choice(free, 2 * m, replace=False))
    c, d = pick[:m], pick[m:]
    base = len(sym)
    sym += c + d
    ci = list(range(base, base + m)); di = list(range(base + m, base + 2 * m))
    t, n = x["target"], x["neg_target"]
    bt, bn = t, n                                        # chain bases
    if control:
        on_path = {u for u, _ in x["edges"] if f"Every {x['idx_to_symbol'][u]} is a" in " ".join(x["steps"])}
        pool = [i for i, s in enumerate(x["idx_to_symbol"]) if s[:1].islower() and i not in (t, n) and i not in on_path]
        bt, bn = (int(v) for v in rng.choice(pool, 2, replace=False))
    new_e = [[bt, ci[0]]] + [[ci[i], ci[i + 1]] for i in range(m - 1)] + \
            [[bn, di[0]]] + [[di[i], di[i + 1]] for i in range(m - 1)]
    say = lambda u, v: f"Every {sym[u]} is a {sym[v]}."
    parts = re.split(r"(?<=\.) ", x["question"])
    stmts, q = parts[:-1], parts[-1]
    mm = re.match(r"^Is (\w+) a (\w+) or (\w+)\?$", q)
    assert mm, q
    tn, nn = x["idx_to_symbol"][t], x["idx_to_symbol"][n]
    first_is_target = mm.group(2) == tn
    assert {mm.group(2), mm.group(3)} == {tn, nn}
    new_s = [say(u, v) for u, v in new_e]
    for s in new_s:
        stmts.insert(int(rng.integers(0, len(stmts) + 1)), s)
    root = mm.group(1)
    if control:
        return {**x, "question": " ".join(stmts) + " " + q, "idx_to_symbol": sym, "edges": x["edges"] + new_e,
                "ext": m, "orig_len": len(x["steps"]), "control": True}
    a, b = (sym[ci[-1]], sym[di[-1]]) if first_is_target else (sym[di[-1]], sym[ci[-1]])
    return {**x, "question": " ".join(stmts) + f" Is {root} a {a} or {b}?", "answer": f"{root} is a {sym[ci[-1]]}.",
            "steps": x["steps"] + new_s[:m], "idx_to_symbol": sym, "edges": x["edges"] + new_e,
            "target": ci[-1], "neg_target": di[-1], "ext": m, "orig_len": len(x["steps"])}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("inp"); ap.add_argument("out")
    ap.add_argument("--m", default="0,2,4,6")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train", default="data/prosqa_train.json")
    ap.add_argument("--control", action="store_true")
    a = ap.parse_args()
    NAMES = sorted({s for x in json.load(open(a.train)) for s in x["idx_to_symbol"] if s[:1].islower()})
    rng = np.random.default_rng(a.seed)
    d = json.load(open(a.inp))
    out, skip = [], {}
    for x in d:
        for m in map(int, a.m.split(",")):
            y = extend(x, m, rng, a.control)
            if y is None:
                skip[m] = skip.get(m, 0) + 1
            else:
                out.append(y)
    json.dump(out, open(a.out, "w"))
    from collections import Counter
    print(a.out, len(out), "examples; by m", sorted(Counter(y["ext"] for y in out).items()),
          "skipped", skip, "path lengths", sorted(Counter(len(y["steps"]) for y in out).items()))
