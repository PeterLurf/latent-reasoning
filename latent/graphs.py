"""Synthetic branching-reachability graphs with a fixed node/edge budget.

Structure of every graph (see docs/PROTOCOL.md, "Task"):
  * a start hub ``s`` and a decoy hub ``s'``;
  * ``NB`` disjoint branches, each a random rooted tree of ``NPB`` nodes;
  * ``W`` branches hang off ``s`` (reachable), the other ``NB - W`` hang off ``s'``
    (structurally identical but unreachable -> degree/shape-matched negatives).

Width therefore changes only which hub a branch is attached to; node count,
edge count, and the multiset of branch shapes are held fixed.

Every edge is one token (src, dst). Edge order is shuffled per graph and node
identities are a fresh random draw from ``V`` ids per graph.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product

import numpy as np

V = 80          # node-id vocabulary
NB = 10         # branches in every graph (fixed budget)
WMAX = 8        # max width -> at least NB - WMAX shape-matched decoy branches remain
NPB = 6         # nodes per branch
E = NB + NB * (NPB - 1)   # 48 edges
UNREACH = 99    # distance sentinel for unreachable-but-present nodes
ABSENT = -1     # distance sentinel for ids not in the graph
N_BUCKETS = 20
SPLITS = {"train": range(0, 14), "discovery": range(14, 17), "confirm": range(17, 20)}


# ---------------------------------------------------------------- tree shapes
def _all_parent_arrays(n):
    """Every recursive-tree parent array: parent[j] in [0, j-1] for j >= 1."""
    return [(-1,) + p for p in product(*[range(j) for j in range(1, n)])]


def _ahu(parent):
    n = len(parent)
    kids = [[] for _ in range(n)]
    for j in range(1, n):
        kids[parent[j]].append(j)

    def enc(u):
        return "(" + "".join(sorted(enc(c) for c in kids[u])) + ")"
    return enc(0)


_PARENTS = _all_parent_arrays(NPB)
_CANON = sorted({_ahu(p) for p in _PARENTS})
N_SHAPES = len(_CANON)
# mixed-radix index of a parent array -> canonical shape id
_RADIX = np.array([int(np.prod(range(1, j))) for j in range(1, NPB)])  # 1,1,2,6,24
SHAPE_OF_PARENT_INDEX = np.zeros(len(_PARENTS), dtype=np.int64)
for p in _PARENTS:
    idx = sum(p[j] * _RADIX[j - 1] for j in range(1, NPB))
    SHAPE_OF_PARENT_INDEX[idx] = _CANON.index(_ahu(p))


def _splitmix64(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.uint64)
    with np.errstate(over="ignore"):
        x = x + np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        x = x ^ (x >> np.uint64(31))
    return x


# ---------------------------------------------------------------- graph batch
@dataclass
class GraphBatch:
    src: np.ndarray        # (B, E) edge source ids, in presented order
    dst: np.ndarray        # (B, E)
    start: np.ndarray      # (B,)
    decoy: np.ndarray      # (B,)
    nodes: np.ndarray      # (B, NB, NPB) ids of branch nodes
    parent: np.ndarray     # (B, NB, NPB) parent index inside branch (-1 root)
    bdepth: np.ndarray     # (B, NB, NPB) depth inside branch (root 0)
    attached: np.ndarray   # (B, NB) bool: branch hangs off start
    hub_pos: np.ndarray    # (B, NB) position of the branch's hub edge in the edge list
    width: np.ndarray      # (B,)
    family: np.ndarray     # (B,) uint64 structural family id (renaming/order invariant)
    bucket: np.ndarray     # (B,) family hash bucket in [0, N_BUCKETS)

    def __len__(self):
        return len(self.start)

    # --- labels derived from construction (validated against BFS in tests)
    def dist(self) -> np.ndarray:
        """(B, V) distance from start: 0..NPB, UNREACH, or ABSENT."""
        B = len(self)
        d = np.full((B, V), ABSENT, dtype=np.int64)
        bi = np.arange(B)[:, None, None]
        val = np.where(self.attached[:, :, None], self.bdepth + 1, UNREACH)
        d[bi, self.nodes] = val
        present_decoy = ~self.attached.all(1)
        d[np.arange(B)[present_decoy], self.decoy[present_decoy]] = UNREACH
        d[np.arange(B), self.start] = 0
        return d

    def subset(self, idx) -> "GraphBatch":
        return GraphBatch(**{k: getattr(self, k)[idx] for k in self.__dataclass_fields__})


def _family(attached, shapes):
    key = attached.astype(np.int64) * N_SHAPES + shapes          # (B, NB) in [0, 2*N_SHAPES)
    key = np.sort(key, axis=1)
    fam = np.zeros(len(key), dtype=np.uint64)
    base = np.uint64(2 * N_SHAPES)
    with np.errstate(over="ignore"):
        for k in range(NB):
            fam = fam * base + key[:, k].astype(np.uint64)
    return fam


# ---------------------------------------------------------------- branch-shape mode
# "default": random recursive trees biased toward chains (every run before fork_*).
# "fork": each graph is, with prob. p_fork, all forks, else all chains. A fork is a two-armed branch whose
#   split sits at within-branch depth f in {0, 1, 2}, one f per graph; decoy branches share the shape. Past
#   the split the frontier is exactly 2W, in chain graphs W, so frontier size and W separate (drawing the
#   shape per branch instead averages out: frontier then tracks W *more* tightly than in "default").
#   Node/edge budget unchanged. Fork graphs span only ~32 structural families, so the family split
#   (train/discovery/confirm) is skipped in this mode; evaluation uses fresh graphs of the same families.
# Module-level so every tool inherits it: train.py sets it from --shapes, and model loading
# (competence.load, development.load_ckpt) re-applies it from the run's config.
SHAPES = {"mode": "default", "p_fork": 0.5}


def set_shapes(mode: str = "default", p_fork: float = 0.5):
    assert mode in ("default", "fork"), mode
    SHAPES.update(mode=mode, p_fork=float(p_fork))


def _fork_template(f):
    """Parent array of a fork split at node f (a chain 0..f, then two arms alternating)."""
    return tuple(-1 if j == 0 else j - 1 if j <= f else f if j <= f + 2 else j - 2 for j in range(NPB))


_FORKS = np.array([_fork_template(f) for f in (0, 1, 2)])            # (3, NPB)
_CHAIN = np.array([-1] + list(range(NPB - 1)))                       # (NPB,)


def sample_graphs(rng: np.random.Generator, B: int, widths=range(1, WMAX + 1),
                  split: str | None = "train", p_chain: float = 0.5,
                  max_bdepth: int | None = None) -> GraphBatch:
    """Sample B graphs whose structural family falls in ``split`` (None = any).

    max_bdepth (training curriculum only) caps within-branch depth by re-parenting
    too-deep nodes to the branch root; node/edge budget is unchanged. At
    max_bdepth=1 every branch is a star, so only 8 families exist (one per width)
    and split filtering is skipped: such all-star graphs have probability ~1e-9
    in full-depth evaluation sampling.
    """
    if (max_bdepth is not None and max_bdepth <= 1) or SHAPES["mode"] == "fork":
        split = None
    out, have = [], 0
    while have < B:
        g = _sample_raw(rng, 2 * B, np.asarray(list(widths)), p_chain, max_bdepth)
        if split is not None:
            keep = np.isin(g.bucket, list(SPLITS[split]))
            g = g.subset(np.nonzero(keep)[0])
        out.append(g)
        have += len(g)
    g = out[0] if len(out) == 1 else GraphBatch(**{
        k: np.concatenate([getattr(o, k) for o in out]) for k in GraphBatch.__dataclass_fields__})
    return g.subset(np.arange(B))


def _sample_raw(rng, B, widths, p_chain, max_bdepth=None):
    ids = np.argsort(rng.random((B, V)), axis=1)[:, : 2 + NB * NPB]
    start, decoy = ids[:, 0], ids[:, 1]
    nodes = ids[:, 2:].reshape(B, NB, NPB)

    # random recursive trees, biased toward chains so depths span 1..NPB-1
    parent = np.full((B, NB, NPB), -1, dtype=np.int64)
    bdepth = np.zeros((B, NB, NPB), dtype=np.int64)
    pidx = np.zeros((B, NB), dtype=np.int64)
    if SHAPES["mode"] == "fork":            # per-branch template; the depth cap below still applies
        is_fork = rng.random(B) < SHAPES["p_fork"]                    # one shape per graph
        tmpl = np.where(is_fork[:, None], _FORKS[rng.integers(0, len(_FORKS), B)], _CHAIN)[:, None, :]
        tmpl = np.broadcast_to(tmpl, (B, NB, NPB))
    for j in range(1, NPB):
        if SHAPES["mode"] == "fork":
            p = tmpl[:, :, j]
        else:
            chain = rng.random((B, NB)) < p_chain
            unif = (rng.random((B, NB)) * j).astype(np.int64)
            p = np.where(chain, j - 1, unif)
        if max_bdepth is not None:
            too_deep = np.take_along_axis(bdepth, p[:, :, None], 2)[:, :, 0] + 1 > max_bdepth
            p = np.where(too_deep, 0, p)
        parent[:, :, j] = p
        bdepth[:, :, j] = np.take_along_axis(bdepth, p[:, :, None], 2)[:, :, 0] + 1
        pidx += p * _RADIX[j - 1]
    shapes = SHAPE_OF_PARENT_INDEX[pidx]

    width = widths[rng.integers(0, len(widths), B)]
    order = np.argsort(rng.random((B, NB)), axis=1)
    attached = np.zeros((B, NB), dtype=bool)
    np.put_along_axis(attached, order, (np.arange(NB)[None] < width[:, None]), 1)

    hub_src = np.where(attached, start[:, None], decoy[:, None])
    int_src = np.take_along_axis(nodes, np.clip(parent[:, :, 1:], 0, None), 2)  # (B, NB, NPB-1)
    src = np.concatenate([hub_src, int_src.reshape(B, -1)], 1)
    dst = np.concatenate([nodes[:, :, 0], nodes[:, :, 1:].reshape(B, -1)], 1)
    perm = np.argsort(rng.random((B, E)), axis=1)
    src = np.take_along_axis(src, perm, 1)
    dst = np.take_along_axis(dst, perm, 1)
    inv = np.argsort(perm, axis=1)
    hub_pos = inv[:, :NB]

    fam = _family(attached, shapes)
    bucket = (_splitmix64(fam) % np.uint64(N_BUCKETS)).astype(np.int64)
    return GraphBatch(src, dst, start, decoy, nodes, parent, bdepth, attached,
                      hub_pos, width, fam, bucket)


# ---------------------------------------------------------------- exact solver
def bfs_dist(src, dst, start) -> np.ndarray:
    """Independent reference solver: BFS distances from start over the edge list."""
    B = len(start)
    d = np.full((B, V), ABSENT, dtype=np.int64)
    for b in range(B):
        adj = {}
        for u, v in zip(src[b], dst[b]):
            adj.setdefault(int(u), []).append(int(v))
            d[b, u] = UNREACH
            d[b, v] = UNREACH
        d[b, start[b]] = 0
        frontier, k = [int(start[b])], 0
        while frontier:
            k += 1
            nxt = []
            for u in frontier:
                for v in adj.get(u, []):
                    if d[b, v] == UNREACH:
                        d[b, v] = k
                        nxt.append(v)
            frontier = nxt
    return d


# ---------------------------------------------------------------- counterfactuals
def move_branch(g: GraphBatch, branch: np.ndarray, to_start: bool, mode: str = "hub",
                rng: np.random.Generator | None = None) -> GraphBatch:
    """Counterfactual partner: re-attach one branch per graph, touching one edge token.

    Only the *source* of that branch's hub edge changes; the edge keeps its
    position, all node ids and every other token are identical.
      to_start=False, mode='hub'   : s  -> s'   (branch becomes unreachable)
      to_start=False, mode='graft' : s  -> random node of an unattached branch
                                     (requires >=1 unattached branch; else falls back to 'hub')
      to_start=True                : s' -> s    (branch becomes reachable)
    """
    B = len(g)
    bi = np.arange(B)
    src = g.src.copy()
    att = g.attached.copy()
    pos = g.hub_pos[bi, branch]
    if to_start:
        assert (~g.attached[bi, branch]).all()
        src[bi, pos] = g.start
        att[bi, branch] = True
    else:
        assert g.attached[bi, branch].all()
        new_src = g.decoy.copy()
        if mode == "graft":
            rng = rng or np.random.default_rng(0)
            free = ~g.attached
            has = free.any(1)
            pick = np.argmax(np.where(free, rng.random((B, NB)), -1), 1)
            node = g.nodes[bi, pick, rng.integers(0, NPB, B)]
            new_src = np.where(has, node, new_src)
        src[bi, pos] = new_src
        att[bi, branch] = False
    # grafted branches are unreachable; construction-based dist() stays valid
    # because it only uses `attached`. Family/bucket are left as the recipient's.
    return replace(g, src=src, attached=att, width=att.sum(1))
