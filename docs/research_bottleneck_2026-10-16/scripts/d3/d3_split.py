"""D3 CPU analysis ($0, 1 thread, no torch): exact replicas of boxes/nested170 generate/render/_leftmost_ready (read-only copy) to
(1) the training token-length law (n ~ U{1..8});
(2) the §170 test set (seed 17100) LONG-STEP chains, scored against N2 / N3's eval_rs.json long_step under length / handle-position / count models;
(3) the cell counts of Experiments' pending eval_steppos.json grids: (B) fresh m 1..16, 2,000 per m, seed 21600 + m; (C) offset items m 1..8, seed 21700 + m;
(4) bracket-depth ranges (train vs test) for a structural-position design."""
import json
import math
import sys
from collections import Counter, defaultdict

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


def _leftmost_ready(nd, path=()):
    if nd.op is None:
        return None
    r = _leftmost_ready(nd.l, path + ("l",))
    if r is not None:
        return r
    if nd.l.op is None and nd.r.op is None:
        return path
    return _leftmost_ready(nd.r, path + ("r",))


def _copy(nd):
    return Node(v=nd.v) if nd.op is None else Node(op=nd.op, l=_copy(nd.l), r=_copy(nd.r))


def n_ops(nd):
    return 0 if nd.op is None else 1 + n_ops(nd.l) + n_ops(nd.r)


def reduce1(nd):
    nd = _copy(nd); path = _leftmost_ready(nd); h = nd
    for p_ in path:
        h = getattr(h, p_)
    h.v = apply(h.op, h.l.v, h.r.v); h.op = h.l = h.r = None
    return nd


def depth_max(syms):
    d = m = 0
    for s in syms:
        d += s == " ("; d -= s == " )"; m = max(m, d)
    return m


def handle_end(nd):
    """slot index (in [EOT] + render + [' =']) of the last slot of the leftmost-ready span (its value slot after the step)."""
    s = render(nd); s2 = render(reduce1(nd))
    # first differing index from the left, then the span length = len(s) - len(s2) + 1
    i = 0
    while i < len(s2) and s[i] == s2[i]:
        i += 1
    span = len(s) - len(s2) + 1
    return 1 + i + span - 1, 1 + i                                           # (last span slot, first span slot) in framed coordinates


def T_of(nd):
    return len(render(nd)) + 2


def main():
    out = {}
    # (1) training length law
    rng = np.random.default_rng(1); Ts = []; Tn8 = []; dmax_tr = Counter(); hend_tr = []
    for _ in range(200000):
        n = int(rng.integers(1, 9)); r, _t = generate(rng, n); s = render(r); T = len(s) + 2; Ts.append(T)
        dmax_tr[depth_max(s)] += 1
        if n == 8:
            Tn8.append(T)
    Ts = np.array(Ts); Tn8 = np.array(Tn8)
    out["train"] = {"T_max": int(Ts.max()), "T_q": {q: float(np.quantile(Ts, q)) for q in (0.5, 0.9, 0.99, 0.999)},
                    "P_T_ge": {str(p): float((Ts >= p).mean()) for p in (28, 30, 31, 32, 33, 34)},
                    "n8_T_hist": {str(k): int(v) for k, v in sorted(Counter(Tn8.tolist()).items())},
                    "slot_exposure_frac_of_items": {str(p): float((Ts > p).mean()) for p in (24, 26, 28, 29, 30, 31, 32, 33)},
                    "depth_max_hist": {str(k): v for k, v in sorted(dmax_tr.items())}}
    # (2) test-set chains (seed 17100, 1000 per n)
    rng = np.random.default_rng(17100); TE = []
    for n in range(1, 17):
        for _ in range(1000):
            r, t = generate(rng, n); TE.append((n, r))
    chains = []                                                             # per item: list of (m, T, hend, hstart, b, dmax)
    dmax_te = defaultdict(Counter)
    for n, r in TE:
        cur = r; ch = []
        dmax_te[n][depth_max(render(r))] += 1
        while cur.op is not None:
            s = render(cur); he, hs = handle_end(cur); ch.append((n_ops(cur), len(s) + 2, he, hs, s.count(" ("), depth_max(s))); cur = reduce1(cur)
        chains.append(ch)
    out["test_depth_max_by_n"] = {str(n): {str(k): v for k, v in sorted(c.items())} for n, c in sorted(dmax_te.items())}
    obs = {}
    for cell in ("s211_vb_pa_s1", "s211_vb_pamct_s1"):
        d = json.load(open("D:/Latent/boxes/runs_fsq/%s/eval_rs.json" % cell))["long_step"]
        obs[cell] = {int(k): (v["n_steps"], v["n_exact"]) for k, v in d.items()}

    def sim(pfun):
        """expected (attempted, exact) per m when the step on (m, T, he) is exact w.p. pfun(...); chains stop at the first inexact step."""
        att = defaultdict(float); ex = defaultdict(float)
        for ch in chains:
            reach = 1.0
            for (m, T, he, hs, b, dm) in ch:
                p = pfun(m, T, he, hs, b, dm); att[m] += reach; ex[m] += reach * p; reach *= p
                if reach < 1e-12:
                    break
        return att, ex

    def nll(att, ex, ob):
        """binomial NLL of observed exact given observed attempted, rate = model ex / att, m 6..16 (+ the attempted-count mismatch reported)."""
        L = 0.0
        for m in range(6, 17):
            A, E = ob[m]; p = min(max(ex[m] / att[m], 1e-6), 1 - 1e-6) if att[m] > 0 else 1e-6
            L -= E * math.log(p) + (A - E) * math.log(1 - p)
        return L

    fits = {}
    out["chain_fits"] = fits
    # chain population T law per m (unconditional: every chain step, no stopping) and the share with T <= 33 / 34
    pop = defaultdict(list)
    for ch in chains:
        for (m, T, he, hs, b, dm) in ch:
            pop[m].append(T)
    out["chain_T_by_m_unconditional"] = {str(m): {"n": len(v), "T_min": int(min(v)), "T_max": int(max(v)), "P_T_le_33": float(np.mean(np.array(v) <= 33)),
                                                  "P_T_le_31": float(np.mean(np.array(v) <= 31))} for m, v in sorted(pop.items()) if m >= 7}
    # first-step (fresh test items only) share with T <= x per n
    fs = defaultdict(list)
    for (n, r) in TE:
        fs[n].append(T_of(r))
    out["test_first_step_T"] = {str(n): {"P_T_le_%d" % x: float(np.mean(np.array(v) <= x)) for x in (29, 31, 33, 35)} for n, v in sorted(fs.items()) if n >= 8}
    # (3) Experiments' grids: (B) fresh m 1..16 x 2,000, seed 21600 + m; (C) m 1..8 x 2,000, seed 21700 + m
    gridB = {}
    for m in range(1, 17):
        rng = np.random.default_rng(21600 + m); c = Counter()
        for _ in range(2000):
            r, _t = generate(rng, m); c[len(render(r)) + 2] += 1
        gridB[str(m)] = {str(k): v for k, v in sorted(c.items())}
    out["gridB_counts_m_T"] = gridB
    gridC = {}
    for m in range(1, 9):
        rng = np.random.default_rng(21700 + m); Tl = []
        for _ in range(2000):
            r, _t = generate(rng, m); Tl.append(len(render(r)) + 2)
        Tl = np.array(Tl)
        gridC[str(m)] = {"T_max": int(Tl.max()), "in_range_o8_T_le_34_minus_o": float(np.mean(Tl + 8 <= 34)), "in_range_o16": float(np.mean(Tl + 16 <= 34)),
                         "in_range_o8_le33": float(np.mean(Tl + 8 <= 33)), "in_range_o16_le33": float(np.mean(Tl + 16 <= 33))}
    out["gridC"] = gridC
    json.dump(out, open(sys.argv[1], "w", newline="\n"), indent=1)
    print(json.dumps(out["train"], indent=0)[:1500]); print(json.dumps({k:out[k] for k in ("chain_T_by_m_unconditional","test_first_step_T","gridC")}, indent=0)[:5000])


if __name__ == "__main__":
    main()
