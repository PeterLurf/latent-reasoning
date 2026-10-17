"""pure-numpy replica of boxes/nested170.generate/render/test_set (no torch import); verified against eval_vbhalt rows' y."""
import numpy as np
P = 11
OPS = ["+", "-", "*", "/"]
PREC = {"+": 1, "-": 1, "*": 2, "/": 2}
INV = {v: pow(v, P - 2, P) for v in range(1, P)}
def apply(o, a, b):
    return (a + b) % P if o == "+" else (a - b) % P if o == "-" else (a * b) % P if o == "*" else (a * INV[b]) % P
PAIRS = {(o, v): [(a, b) for a in range(P) for b in range(P) if not (o == "/" and b == 0) and apply(o, a, b) == v] for o in OPS for v in range(P)}
class Node:
    __slots__ = ("op", "l", "r", "v")
    def __init__(self, v=None, op=None, l=None, r=None):
        self.v, self.op, self.l, self.r = v, op, l, r
def generate(rng, n):
    t = int(rng.integers(0, P)); root = Node(v=t); leaves = [root]
    for _ in range(n):
        i = int(rng.integers(0, len(leaves))); leaf = leaves.pop(i); o = OPS[int(rng.integers(0, 4))]; prs = PAIRS[(o, leaf.v)]; a, b = prs[int(rng.integers(0, len(prs)))]
        leaf.op, leaf.l, leaf.r, leaf.v = o, Node(v=a), Node(v=b), None; leaves.insert(i, leaf.r); leaves.insert(i, leaf.l)
    return root, t
def render(nd, parent=None, right=False):
    if nd.op is None:
        return [" %d" % nd.v]
    s = render(nd.l, nd.op, False) + [" " + nd.op] + render(nd.r, nd.op, True)
    if parent is not None and (PREC[nd.op] < PREC[parent] or (right and PREC[nd.op] == PREC[parent])):
        s = [" ("] + s + [" )"]
    return s
def test_set(per_n=1000, n_max=16, seed=17100):
    rng = np.random.default_rng(seed); out = []
    for n in range(1, n_max + 1):
        for _ in range(per_n):
            r, t = generate(rng, n); s = render(r); out.append({"n": n, "syms": s, "t": t, "T": len(s) + 2, "nbr": s.count(" ("), "root": r})
    return out
