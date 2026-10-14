"""SEEDSTAB2 check H: the height (sequential-reduction depth) of each Feng reduction, on nested170's generator (copied verbatim, numpy only).
height(node) = 0 for a number, 1 + max(height(l), height(r)) for an op node, in the ORIGINAL tree. Reduction t reduces one op node; its value
needs height(node) sequential reductions from the leaves. Output: heights.json beside this file."""
import json, os
import numpy as np

P = 11
OPS = ["+", "-", "*", "/"]
INV = {v: pow(v, P - 2, P) for v in range(1, P)}


def apply(o, a, b):
    return (a + b) % P if o == "+" else (a - b) % P if o == "-" else (a * b) % P if o == "*" else (a * INV[b]) % P


PAIRS = {(o, v): [(a, b) for a in range(P) for b in range(P) if not (o == "/" and b == 0) and apply(o, a, b) == v] for o in OPS for v in range(P)}


class Node:
    __slots__ = ("op", "l", "r", "v", "h")

    def __init__(self, v=None, op=None, l=None, r=None):
        self.v, self.op, self.l, self.r, self.h = v, op, l, r, 0


def generate(rng, n):
    t = int(rng.integers(0, P)); root = Node(v=t); leaves = [root]
    for _ in range(n):
        i = int(rng.integers(0, len(leaves))); leaf = leaves.pop(i); o = OPS[int(rng.integers(0, 4))]; prs = PAIRS[(o, leaf.v)]; a, b = prs[int(rng.integers(0, len(prs)))]
        leaf.op, leaf.l, leaf.r, leaf.v = o, Node(v=a), Node(v=b), None; leaves.insert(i, leaf.r); leaves.insert(i, leaf.l)
    return root, t


def set_h(nd):
    if nd.op is None:
        nd.h = 0; return 0
    nd.h = 1 + max(set_h(nd.l), set_h(nd.r)); return nd.h


def _leftmost_ready(nd, path=()):
    if nd.op is None:
        return None
    r = _leftmost_ready(nd.l, path + ("l",))
    if r is not None:
        return r
    if nd.l.op is None and nd.r.op is None:
        return path
    return _leftmost_ready(nd.r, path + ("r",))


def red_heights(root):
    """heights (original tree) of the reduced nodes, in cot order; reduces the tree in place (it is a throwaway copy here)."""
    out = []
    while root.op is not None:
        path = _leftmost_ready(root); tgt = root
        for p_ in path:
            tgt = getattr(tgt, p_)
        out.append(tgt.h); tgt.v = apply(tgt.op, tgt.l.v, tgt.r.v); tgt.op = tgt.l = tgt.r = None
    return out


rng = np.random.default_rng(20261014)
by_t = {t: [] for t in range(1, 9)}; root_h = {n: [] for n in range(1, 9)}
for _ in range(40000):
    n = int(rng.integers(1, 9)); r, _t = generate(rng, n); set_h(r); root_h[n].append(r.h); hs = red_heights(r)
    assert len(hs) == n
    for t, h in enumerate(hs, 1):
        by_t[t].append(h)
res = {"by_t": {}, "root_height_by_n": {}, "pooled_t_ge2": {}}
for t, hs in by_t.items():
    hs = np.array(hs); res["by_t"][str(t)] = {"n": int(len(hs)), "h1": float((hs == 1).mean()), "h2": float((hs == 2).mean()), "h3plus": float((hs >= 3).mean()), "mean": float(hs.mean())}
for n, hs in root_h.items():
    hs = np.array(hs); res["root_height_by_n"][str(n)] = {"mean": float(hs.mean()), "dist": {str(k): float((hs == k).mean()) for k in range(1, 9) if (hs == k).any()}}
allh = np.array([h for t in range(2, 9) for h in by_t[t]])
res["pooled_t_ge2"] = {"n": int(len(allh)), "h1": float((allh == 1).mean()), "h2": float((allh == 2).mean()), "h3plus": float((allh >= 3).mean())}
json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "heights.json"), "w", newline="\n"), indent=1)
for t in range(1, 9):
    d = res["by_t"][str(t)]; print("t", t, "n", d["n"], "h1 %.3f h2 %.3f h3+ %.3f mean %.2f" % (d["h1"], d["h2"], d["h3plus"], d["mean"]))
print("pooled t>=2", res["pooled_t_ge2"])
for n in range(1, 9):
    print("n", n, "root height mean %.2f" % res["root_height_by_n"][str(n)]["mean"], res["root_height_by_n"][str(n)]["dist"])
