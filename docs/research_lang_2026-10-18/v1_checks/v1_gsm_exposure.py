"""LANG v1 $0: exposed-item counts for the GSM8K single-slot (slot-t-only) VALUE-SWAP arm (i) and the whole-bank arm (ii') of B1,
per gold m ("<<" count) on the 1,319 GSM8K test items (Audit's 'controls must be exposed' addition, relayed 10-18).
An item is EXPOSED for class A at step t (1-based binary DECOMP step) iff
  (a) the item DECOMP-parses (PROOFS_LANG's parser, imported read-only), 2 <= t <= m_bin - 1, m_bin <= 12;
  (b) both operands of step t are question literals that are unique in the question, not equal to any earlier step result
      (unambiguous provenance), and used as an operand by no other step (the donor can redraw them without touching another step);
  (c) step t's result is an ancestor of the final step in the provenance DAG (a later step consumes it: the splice can matter);
  (d) under a redraw of the two literals (each + U{1..9}, recomputed exactly through the DAG) the final answer changes (y_donor != y).
Per item we count the eligible t (an item is exposed if >= 1 t is). Recipients in the registered arm also use 5 R_x variants per item
(rng 22720); this script counts base items only (a lower bound on candidates).
    python -B v1_gsm_exposure.py   -> v1_gsm_exposure.json
"""
import collections
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "proofs_lang"))
from gsm_check import load, tree, lits  # noqa: E402

TEST = "D:/Latent/bench/data/gsm8k_aug/test.txt"


def program(q, eqs):
    """-> list of steps [(op, src_a, src_b, value)], src = ('lit', value) | ('step', index) | ('const', value), or None"""
    L = lits(q); lc = collections.Counter(L); steps = []; vals = []
    for e in eqs:
        if "=" not in e:
            return None
        t = tree(e.rsplit("=", 1)[0])
        if t is None or not t[1]:
            return None
        base = len(steps)
        for op, A, B, v in t[1]:
            srcs = []
            for kind, val in (A, B):
                if kind == "step":
                    srcs.append(("step", base + val)); continue
                er = [i for i, u in enumerate(vals) if abs(u - val) < 1e-9]
                nl = lc.get(val, 0)
                if er and nl:
                    srcs.append(("amb", val))
                elif er:
                    srcs.append(("step", er[-1]))
                elif nl == 1:
                    srcs.append(("lit", val))
                elif nl > 1:
                    srcs.append(("nlit", val))
                else:
                    srcs.append(("const", val))
            steps.append((op, srcs[0], srcs[1], v)); vals.append(v)
    return steps


def evaluate(steps, sub):
    out = []
    for op, a, b, v in steps:
        def g(s):
            if s[0] == "step":
                return out[s[1]]
            return sub.get(s[1], s[1]) if s[0] == "lit" else s[1]
        x, y = g(a), g(b)
        try:
            out.append({"Add": x + y, "Sub": x - y, "Mult": x * y, "Div": x / y}[op])
        except ZeroDivisionError:
            return None
    return out


def main():
    data = [d for d in load(TEST) if d is not None]
    rng = random.Random(22720)
    by_m = collections.defaultdict(lambda: collections.Counter()); by_j = collections.Counter(); cf_j = collections.Counter()
    for q, eqs, ans in data:
        m = len(eqs); row = by_m[m]; row["items"] += 1
        st = program(q, eqs)
        if not st:
            continue
        row["parsed"] += 1; mb = len(st)
        if mb > 12:
            continue
        uses = collections.Counter()
        for _, a, b, _ in st:
            for s in (a, b):
                if s[0] in ("lit", "nlit", "amb", "const"):
                    uses[s[1]] += 1
        anc = set([mb - 1]); stack = [mb - 1]
        while stack:
            k = stack.pop()
            for s in st[k][1:3]:
                if s[0] == "step" and s[1] not in anc:
                    anc.add(s[1]); stack.append(s[1])
        base = evaluate(st, {})
        elig = 0
        for t in range(2, mb):                                   # 1-based t in 2..m_bin-1 -> index t-1
            k = t - 1; op, a, b, v = st[k]
            if not (a[0] == "lit" and b[0] == "lit" and uses[a[1]] == 1 and uses[b[1]] == 1 and a[1] != b[1]):
                continue
            row["class_a_operands"] += 1
            if k not in anc:
                continue
            row["class_a_consumed"] += 1
            sub = {a[1]: a[1] + rng.randint(1, 9), b[1]: b[1] + rng.randint(1, 9)}
            new = evaluate(st, sub)
            if new is None or base is None or abs(new[-1] - base[-1]) < 1e-6:
                continue
            elig += 1; by_j[(m, t)] += 1
            if t <= mb - 1:
                cf_j[t] += 1                                      # CONSUMPTION-FRONT: the swapped step is not the last step (t <= m_bin - 1 always here)
        row["steps_exposed"] += elig; row["items_exposed"] += elig > 0
    out = {"by_m": {m: dict(by_m[m]) for m in sorted(by_m)}, "exposed_steps_by_j": {j: cf_j[j] for j in sorted(cf_j)},
           "exposed_steps_by_m_j": {"%d_%d" % k: v for k, v in sorted(by_j.items())}}
    tot = collections.Counter(); ge3 = collections.Counter()
    for m, r in by_m.items():
        tot.update(r)
        if m >= 3:
            ge3.update(r)
    out["all"] = dict(tot); out["m_ge_3"] = dict(ge3)
    print(json.dumps(out, indent=1))
    json.dump(out, open(os.path.join(HERE, "v1_gsm_exposure.json"), "w", newline="\n"), indent=1)


if __name__ == "__main__":
    main()
