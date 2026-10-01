"""THEORY §60 (boundary memory, Phase A): task data, legal merge schedules and the exact Boolean composer.

A region is a small DAG (local vertices 0..n-1, edges u -> v) with pinned terminals P. A sub-region S (a set of edges)
has ports = its vertices incident to an edge outside S, plus pinned vertices in S (a leaf, one edge, keeps both
endpoints as ports). Its boundary-response table T_S has one row per seed pattern a over its <= 4 port slots
(16 rows) and one column per slot: T_S[a, j] = 1 iff port j is in a or reachable from a port in a using edges of S.
A schedule is a binary merge tree over the edges (leaves = edges) whose every node has <= 4 ports; children of a merge
share >= 1 vertex. Training schedules have no merge whose two children both have >= 3 edges; held-out-order schedules
have at least one such merge. A task is one region with one terminal label T_R[a, j] (any a != 0, trivial j in a included: note 60-3), balanced 50/50.
The model never sees BFS or any intermediate table; the exact composer here is the diagnostic ceiling.
"""
import itertools
from collections import deque

import numpy as np

FAMILIES = ["path", "out_tree", "in_tree", "diamond", "skip_chain", "sparse_dag", "ladder", "zigzag"]
TRAIN_FAMILIES = FAMILIES[:6]
HELDOUT_FAMILIES = ["ladder", "zigzag"]
MAXP = 4
PATTERNS = list(itertools.product([0, 1], repeat=MAXP))          # 16 seed patterns over 4 slots, row r = bits of r


# ---------------------------------------------------------------- motifs
def motif(rng, fam, k):
    """Edges (list of (u, v), u -> v) of a connected DAG with k edges from family fam, or None if k does not fit."""
    if fam == "path":
        return [(i, i + 1) for i in range(k)]
    if fam in ("out_tree", "in_tree"):
        e = [(int(rng.integers(0, i + 1)), i + 1) for i in range(k)]
        return e if fam == "out_tree" else [(v, u) for u, v in e]
    if fam == "diamond":
        if k < 4:
            return None
        e, s, nv = [], 0, 1
        while k - len(e) >= 4:
            a, b, t = nv, nv + 1, nv + 2; nv += 3
            e += [(s, a), (s, b), (a, t), (b, t)]; s = t
        while len(e) < k:
            e.append((s, nv)); s = nv; nv += 1
        return e
    if fam == "skip_chain":
        L = max(2, k - max(1, k // 3)); sk = k - L
        e = [(i, i + 1) for i in range(L)]
        cand = [(i, i + 2) for i in range(L - 1)]
        if sk > len(cand):
            return None
        idx = rng.choice(len(cand), sk, replace=False)
        return e + [cand[i] for i in sorted(idx)]
    if fam == "sparse_dag":
        n = int(rng.integers(max(3, (k + 3) // 2), k + 2))
        e = {(int(rng.integers(0, i)), i) for i in range(1, n)}           # spanning tree, edges low -> high
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n) if (i, j) not in e]
        if len(e) > k or len(e) + len(pairs) < k:
            return None
        extra = rng.choice(len(pairs), k - len(e), replace=False)
        return sorted(e | {pairs[i] for i in extra})
    if fam == "ladder":
        if k < 3:
            return None
        L = max(1, (k + 1) // 3); e = []
        top = list(range(L + 1)); bot = list(range(L + 1, 2 * L + 2))
        e += [(top[i], top[i + 1]) for i in range(L)] + [(bot[i], bot[i + 1]) for i in range(L)]
        rungs = [(top[i], bot[i]) for i in range(L + 1)]
        need = k - len(e)
        if need < 1 or need > len(rungs):
            return None
        idx = sorted(rng.choice(len(rungs), need, replace=False))
        return e + [rungs[i] for i in idx]
    if fam == "zigzag":
        return [(i, i + 1) if i % 2 == 0 else (i + 1, i) for i in range(k)]
    raise ValueError(fam)


def relabel(rng, edges):
    """Random permutation of vertex ids (so ids carry no structure)."""
    vs = sorted({x for e in edges for x in e})
    perm = rng.permutation(len(vs))
    mp = {v: int(perm[i]) for i, v in enumerate(vs)}
    return [(mp[u], mp[v]) for u, v in edges], len(vs)


# ---------------------------------------------------------------- reachability and tables
def reach_from(edges, seeds):
    adj = {}
    for u, v in edges:
        adj.setdefault(u, []).append(v)
    seen = set(seeds); q = deque(seeds)
    while q:
        u = q.popleft()
        for v in adj.get(u, []):
            if v not in seen:
                seen.add(v); q.append(v)
    return seen


def ports_of(sub, all_edges, pinned):
    """Ports of sub-region `sub` (a frozenset of edge indices) within the region."""
    vs = {x for i in sub for x in all_edges[i]}
    outside = {x for i in range(len(all_edges)) if i not in sub for x in all_edges[i]}
    return sorted(v for v in vs if v in outside or v in pinned)


def true_table(edges_sub, ports):
    """16 x 4 table over port slots 0..len(ports)-1 (rows with seeds on unused slots and unused columns are 0)."""
    T = np.zeros((16, MAXP), np.float32)
    for r, pat in enumerate(PATTERNS):
        if any(pat[s] for s in range(len(ports), MAXP)):
            continue
        seeds = [ports[s] for s in range(len(ports)) if pat[s]]
        R = reach_from(edges_sub, seeds)
        for j, p in enumerate(ports):
            T[r, j] = float(p in R)
    return T


def row_mask(nports):
    """(16,) valid rows (seeds only on used slots) and (4,) valid columns."""
    rm = np.array([not any(p[s] for s in range(nports, MAXP)) for p in PATTERNS], bool)
    cm = np.array([s < nports for s in range(MAXP)], bool)
    return rm, cm


# ---------------------------------------------------------------- schedules
def random_schedule(rng, edges, pinned, mode="train", tries=50):
    """Merge tree as a list of nodes: leaves 0..E-1 (one edge each), then merges (a, b) in creation order.
    Returns (nodes, ports) where nodes[i] = frozenset(edge ids), merges[i] = (a, b) or None for leaves; or None."""
    E = len(edges)
    for _ in range(tries):
        nodes = [frozenset([i]) for i in range(E)]; kids = [None] * E
        avail = list(range(E)); ok = True; big_merge = False
        while len(avail) > 1:
            cands = []
            for x, y in itertools.combinations(avail, 2):
                vx = {v for i in nodes[x] for v in edges[i]}; vy = {v for i in nodes[y] for v in edges[i]}
                if not vx & vy:
                    continue
                if mode == "train" and len(nodes[x]) >= 3 and len(nodes[y]) >= 3:
                    continue
                if len(ports_of(nodes[x] | nodes[y], edges, pinned)) > MAXP:
                    continue
                cands.append((x, y))
            if not cands:
                ok = False; break
            x, y = cands[int(rng.integers(len(cands)))]
            if len(nodes[x]) >= 3 and len(nodes[y]) >= 3:
                big_merge = True
            nodes.append(nodes[x] | nodes[y]); kids.append((x, y))
            avail = [z for z in avail if z not in (x, y)] + [len(nodes) - 1]
        if ok and (mode != "heldout" or big_merge):
            return nodes, kids
    return None


def schedule_tables(edges, pinned, nodes, kids):
    """Per node: ports (vertex ids, final node = pinned order) and the true table; plus depth of each node."""
    root = len(nodes) - 1
    out = []
    depth = []
    for i, s in enumerate(nodes):
        pr = ports_of(s, edges, pinned) if i != root else list(pinned)
        out.append(pr)
        depth.append(0 if kids[i] is None else 1 + max(depth[kids[i][0]], depth[kids[i][1]]))
    return out, depth


# ---------------------------------------------------------------- exact composer (diagnostic ceiling)
def compose_exact(TA, pA, TB, pB, pP):
    """Boolean fixed-point composition (Kron reduction over the Boolean semiring) of child tables over their ports
    into the parent's ports. TA, TB: 16 x 4 over slots; pA, pB, pP: vertex lists in slot order."""
    allv = sorted(set(pA) | set(pB))
    def apply(T, ports, cur):
        r = sum(1 << (MAXP - 1 - s) for s, v in enumerate(ports) if v in cur)   # row index = bits as PATTERNS
        return {v for s, v in enumerate(ports) if T[r, s] > 0.5}
    TP = np.zeros((16, MAXP), np.float32)
    for r, pat in enumerate(PATTERNS):
        if any(pat[s] for s in range(len(pP), MAXP)):
            continue
        cur = {pP[s] for s in range(len(pP)) if pat[s]}
        while True:
            new = cur | apply(TA, pA, cur) | apply(TB, pB, cur)
            if new == cur:
                break
            cur = new
        for j, v in enumerate(pP):
            TP[r, j] = float(v in cur)
    return TP


def run_exact(edges, pinned, nodes, kids, ports):
    """Exact tables bottom-up from exact leaf tables."""
    T = []
    for i, s in enumerate(nodes):
        if kids[i] is None:
            T.append(true_table([edges[next(iter(s))]], ports[i]))
        else:
            a, b = kids[i]
            T.append(compose_exact(T[a], ports[a], T[b], ports[b], ports[i]))
    return T


# ---------------------------------------------------------------- tasks
def make_task(rng, fam, k, nP, mode="train"):
    """One task, or (None, reason). mode: train | heldout (held-out merge orders) | any (no order restriction)."""
    e = motif(rng, fam, k)
    if e is None:
        return None, "motif"
    edges, n = relabel(rng, e)
    if nP > n:
        return None, "pins"
    pinned = [int(v) for v in rng.choice(n, nP, replace=False)]
    sch = random_schedule(rng, edges, set(pinned), "heldout" if mode == "heldout" else ("train" if mode == "train" else "any"))
    if sch is None:
        return None, "schedule"
    nodes, kids = sch
    ports, depth = schedule_tables(edges, pinned, nodes, kids)
    TR = true_table(edges, pinned)
    rm, cm = row_mask(nP)
    cand = [(r, j) for r in range(16) if rm[r] and any(PATTERNS[r]) for j in range(nP)]   # all a != 0 entries (60-3)
    pos = [c for c in cand if TR[c] > 0.5]; neg = [c for c in cand if TR[c] < 0.5]
    if not pos or not neg:
        return None, "unbalanced"
    want = int(rng.integers(2))
    r, j = (pos if want else neg)[int(rng.integers(len(pos if want else neg)))]
    return {"family": fam, "k": k, "edges": edges, "n": n, "pinned": pinned, "nodes": [sorted(s) for s in nodes],
            "kids": kids, "ports": ports, "depth": depth, "row": r, "col": j, "label": float(TR[r, j]),
            "table": TR.tolist()}, None


def make_set(rng, n, families, kmin, kmax, mode="train", nPs=(2, 3, 4)):
    """n balanced tasks; also returns rejection counts per family."""
    out, rej, tried = [], {}, {}
    while len(out) < n:
        fam = families[int(rng.integers(len(families)))]
        k = int(rng.integers(kmin, kmax + 1)); nP = int(nPs[int(rng.integers(len(nPs)))])
        tried[fam] = tried.get(fam, 0) + 1
        t, why = make_task(rng, fam, k, nP, mode)
        if t is None:
            rej.setdefault(fam, {}); rej[fam][why] = rej[fam].get(why, 0) + 1
            continue
        out.append(t)
    return out, {"tried": tried, "rejected": rej}
