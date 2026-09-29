"""Linear readouts of node reachability from continuous thoughts.

Probe for step k: logits = h_k @ W_k + b_k over the V node ids (one logit per id,
masked to ids present in the graph). Node ids are re-drawn per graph, so a probe
cannot succeed by memorising identities; it must read an id-indexed code.

Label families (per graph, per present node v, excluding the start node):
  reach   : v reachable from s (any depth)
  within_j: dist(s, v) <= j
Controls:
  * pre-recurrence representation (h_0, output at the START token)
  * untrained model with identical architecture (random representation)
  * label shuffle across graphs (probe capacity / memorisation)
  * depth-matched decoys: AUC computed within strata of within-branch depth,
    comparing reachable-branch nodes with shape-matched decoy-branch nodes.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .data import to_torch
from .graphs import UNREACH, V, sample_graphs


@torch.no_grad()
def collect(model, g, K, device, bs=512):
    """Thoughts (N, K+1, d) for the delayed format (no target in context)."""
    model.eval()
    out = []
    for i in range(0, len(g), bs):
        sub = g.subset(np.arange(i, min(len(g), i + bs)))
        src, dst, start = to_torch(sub, device)
        out.append(model.run(src, dst, start, K, keep=True)["thoughts"].float())
    return torch.cat(out)


def labels(g, kind="reach"):
    d = g.dist()
    present = (d != -1) & (d != 0)
    if kind == "reach":
        y = (d >= 1) & (d < UNREACH)
    else:
        j = int(kind.split("_")[1])
        y = (d >= 1) & (d <= j)
    return torch.as_tensor(y, dtype=torch.float32), torch.as_tensor(present)


def fit_probe(H, Y, M, epochs=300, lr=3e-2, wd=1e-4, device="cuda"):
    """H (N,d), Y/M (N,V). Full-batch AdamW logistic regression, masked to present ids."""
    H, Y, M = H.to(device), Y.to(device), M.to(device).float()
    mu, sd = H.mean(0), H.std(0) + 1e-5
    W = torch.zeros(H.shape[1], V, device=device, requires_grad=True)
    b = torch.zeros(V, device=device, requires_grad=True)
    opt = torch.optim.AdamW([W, b], lr=lr, weight_decay=wd)
    Hn = (H - mu) / sd
    for _ in range(epochs):
        loss = (F.binary_cross_entropy_with_logits(Hn @ W + b, Y, reduction="none") * M).sum() / M.sum()
        opt.zero_grad(); loss.backward(); opt.step()
    return {"W": W.detach(), "b": b.detach(), "mu": mu, "sd": sd}


def apply_probe(p, H):
    H = H.to(p["W"].device)
    return ((H - p["mu"]) / p["sd"]) @ p["W"] + p["b"]


def probe_directions(p):
    """Directions in raw thought space whose inner product gives each id's logit: W / sd."""
    return p["W"] / p["sd"][:, None]          # (d, V)


def auc(pos, neg):
    """Mann-Whitney AUC."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    r = allv.argsort().argsort().astype(np.float64) + 1
    # average ties
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=r)
    r = (sums / cnt)[inv]
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def matched_auc(scores, g):
    """AUC(reachable-branch nodes vs decoy-branch nodes), averaged over within-branch depth strata.

    scores: (N, V) numpy logits. Returns (overall matched AUC, {depth: AUC}).
    """
    N = len(g)
    bi = np.arange(N)[:, None, None]
    s = scores[bi, g.nodes]                              # (N, NB, NPB)
    att = np.broadcast_to(g.attached[:, :, None], s.shape)
    per = {}
    for j in np.unique(g.bdepth):
        m = g.bdepth == j
        per[int(j)] = auc(s[m & att], s[m & ~att])
    return float(np.nanmean(list(per.values()))), per


def branch_auc(scores, g):
    """Per graph, per attached branch: mean score of the branch nodes vs matched decoys.

    Returns (N, NB) array of per-branch AUCs vs that graph's decoy nodes at the same
    depths (nan for unattached branches).
    """
    N = len(g)
    bi = np.arange(N)[:, None, None]
    s = scores[bi, g.nodes]
    out = np.full(g.attached.shape, np.nan)
    for n in range(N):
        dec = s[n][~g.attached[n]].ravel()
        for i in np.nonzero(g.attached[n])[0]:
            out[n, i] = auc(s[n, i], dec)
    return out
