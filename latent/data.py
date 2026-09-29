"""Turn GraphBatch objects into model-ready tensors and balanced queries."""
from __future__ import annotations

import numpy as np
import torch

from .graphs import UNREACH, GraphBatch, V


def _pick(rng, mask, n):
    """For each row pick n entries uniformly (with replacement) among True cells."""
    B = mask.shape[0]
    key = np.where(mask, rng.random((B, V)), 2.0)
    order = np.argsort(key, 1)
    cnt = mask.sum(1)
    assert (cnt > 0).all()
    j = np.minimum((rng.random((B, n)) * cnt[:, None]).astype(np.int64), cnt[:, None] - 1)
    return np.take_along_axis(order, j, 1)


def balanced_queries(rng, g: GraphBatch, Q: int):
    """Q/2 reachable (excluding start) and Q/2 unreachable present nodes per graph."""
    d = g.dist()
    reach = (d >= 1) & (d < UNREACH)
    unr = d == UNREACH
    q = np.concatenate([_pick(rng, reach, Q // 2), _pick(rng, unr, Q - Q // 2)], 1)
    perm = np.argsort(rng.random(q.shape), 1)
    q = np.take_along_axis(q, perm, 1)
    qd = np.take_along_axis(d, q, 1)
    return q, qd


def parent_queries(rng, g: GraphBatch, Qp: int):
    """Auxiliary lookup targets: random branch nodes and the id of their parent."""
    from .graphs import NB, NPB
    B = len(g)
    bi = np.arange(B)[:, None]
    b = rng.integers(0, NB, (B, Qp))
    j = rng.integers(0, NPB, (B, Qp))
    v = g.nodes[bi, b, j]
    par_j = np.clip(g.parent[bi, b, j], 0, None)
    par = np.where(j == 0, np.where(g.attached[bi, b], g.start[:, None], g.decoy[:, None]),
                   g.nodes[bi, b, par_j])
    return v, par


def children_queries(rng, g: GraphBatch, Qc: int):
    """Auxiliary lookup targets: random non-start nodes (branch nodes or s') and a multi-hot
    of their children. The start node is never queried, so reachability is never supervised."""
    from .graphs import NB, NPB
    B = len(g)
    bi = np.arange(B)[:, None]
    flat = np.concatenate([g.nodes.reshape(B, -1), g.decoy[:, None]], 1)       # (B, 61)
    pick = rng.integers(0, flat.shape[1], (B, Qc))
    v = flat[bi, pick]
    C = np.zeros((B, V, V), dtype=np.float32)
    C[bi, g.src, g.dst] = 1.0
    return v, C[bi, v]                                                         # (B,Qc), (B,Qc,V)


def to_torch(g: GraphBatch, device):
    t = lambda a: torch.as_tensor(a, dtype=torch.long, device=device)
    return t(g.src), t(g.dst), t(g.start)


def make_batch(rng, g: GraphBatch, Q: int, device, Qp: int = 8):
    src, dst, start = to_torch(g, device)
    q, qd = balanced_queries(rng, g, Q)
    tq, tqd = balanced_queries(rng, g, 2)
    pick = rng.integers(0, 2, len(g))
    tgt, tgtd = tq[np.arange(len(g)), pick], tqd[np.arange(len(g)), pick]
    pv, pp = parent_queries(rng, g, Qp)
    cv, cy = children_queries(rng, g, Qp)
    T = lambda a: torch.as_tensor(a, device=device)
    return dict(src=src, dst=dst, start=start, pq=T(pv).long(), py=T(pp).long(),
                cq=T(cv).long(), cy=T(cy),
                q=T(q).long(), qd=T(qd), qy=T(((qd >= 1) & (qd < UNREACH)).astype(np.float32)),
                tgt=T(tgt).long(), tgtd=T(tgtd), ty=T(((tgtd >= 1) & (tgtd < UNREACH)).astype(np.float32)))
