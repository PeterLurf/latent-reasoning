"""Branch-level donor replacement on a shared latent state.

For a recipient graph g and one of its reachable branches i, the donor is the
single-token counterfactual g' = move_branch(g, i, to_start=False): identical
tokens except that branch i's hub edge now leaves s' instead of s.

At thought k (the vector that becomes the input of LAT_{k+1}):
    h_patched = h_rec + P (h_don - h_rec)
where P = U U^T for an orthonormal U. KV entries of positions <= k are kept;
positions k+1..K and every query are recomputed. All present nodes are then
queried as independent forks of that one saved state.
"""
from __future__ import annotations

import numpy as np
import torch

from .data import to_torch
from .graphs import NB, NPB, UNREACH, move_branch
from .probes import probe_directions


def all_queries(g):
    """(N, NB*NPB + 1) query ids: every branch node, then the decoy hub."""
    N = len(g)
    return np.concatenate([g.nodes.reshape(N, -1), g.decoy[:, None]], 1)


def query_meta(g):
    """branch index (-1 for decoy hub) and distance of each query in all_queries order."""
    N = len(g)
    br = np.concatenate([np.repeat(np.arange(NB), NPB)[None].repeat(N, 0), -np.ones((N, 1), int)], 1)
    d = np.take_along_axis(g.dist(), all_queries(g), 1)
    return br, d


def orth(A):
    """(N, d, r) -> orthonormal basis of the column space (drops ~zero columns)."""
    Q, R = torch.linalg.qr(A)
    return Q


def make_basis(kind, h_rec, h_don, g, branch, k, probe=None, rank=None, gen=None):
    N, d = h_rec.shape
    dev = h_rec.device
    if kind == "full":
        return None
    if kind == "probe":
        D = probe_directions(probe).to(dev)                          # (d, V)
        ids = torch.as_tensor(g.nodes[np.arange(N), branch], device=dev)   # (N, NPB)
        return orth(D.T[ids].transpose(1, 2))                         # (N, d, NPB)
    if kind == "random":
        r = rank or NPB
        A = torch.randn(N, d, r, device=dev, generator=gen)
        return orth(A)
    if kind == "pattern":
        # Haufe activation patterns of the reach probe, A = Cov(h_k) W_raw, for branch i's ids
        A = probe["pattern"].to(dev)                                  # (d, V)
        ids = torch.as_tensor(g.nodes[np.arange(N), branch], device=dev)
        return orth(A.T[ids].transpose(1, 2))
    if kind == "pca":
        # shared basis (d, r) fitted on discovery donor differences at this step
        B_ = probe["pca"][k][:, : (rank or probe["pca"][k].shape[1])].to(dev)
        return B_.expand(N, -1, -1)
    raise ValueError(kind)


def apply_patch(h_rec, h_don, U, kind, gen=None):
    diff = h_don - h_rec
    if kind == "full":
        return h_don.clone()
    proj = (U @ (U.transpose(1, 2) @ diff[:, :, None]))[:, :, 0]
    if kind in ("probe", "random", "pca", "pattern"):
        return h_rec + proj
    if kind == "normrand":        # isotropic perturbation with the norm of the full donor difference
        xi = torch.randn(h_rec.shape, device=h_rec.device, generator=gen)
        return h_rec + xi / xi.norm(dim=1, keepdim=True) * diff.norm(dim=1, keepdim=True)
    raise ValueError(kind)


@torch.no_grad()
def run_thoughts(model, g, K, target=None, device="cuda"):
    src, dst, start = to_torch(g, device)
    tgt = None if target is None else torch.as_tensor(target, device=device)
    return model.run(src, dst, start, K, target=tgt, keep=True)["thoughts"]


@torch.no_grad()
def patched_probs(model, g_rec, K, k, new_h, qnodes=None, target=None, device="cuda", keep=False):
    """Run recipient with thought k replaced by new_h; return P(reachable) per query."""
    src, dst, start = to_torch(g_rec, device)
    q = None if qnodes is None else torch.as_tensor(qnodes, device=device)
    tgt = None if target is None else torch.as_tensor(target, device=device)
    patch = lambda kk, h: new_h if kk == k else h
    o = model.run(src, dst, start, K, target=tgt, qnodes=q, patch=patch, keep=keep)
    p = torch.sigmoid(o["logit"].float())
    return (p, o["thoughts"]) if keep else p


@torch.no_grad()
def effect_matrix(model, g, K, k, kind="full", probe=None, direction="remove",
                  donor_mode="hub", device="cuda", seed=0, keep_thoughts=False, rank=None, pca=None):
    """Edit each reachable branch in turn from the same saved recipient state.

    direction='remove': recipient = g (all W branches reachable), donor detaches branch i.
    direction='insert': recipient = g with branch i detached, donor = g (reattached);
                        effects are then measured against that per-edit recipient.
    Returns dict with base probs (N, Qn), patched probs (N, W, Qn), metadata.
    Assumes all graphs in g share the same width W.
    """
    W = int(g.width[0])
    assert (g.width == W).all()
    N = len(g)
    gen = torch.Generator(device=device).manual_seed(seed)
    rng = np.random.default_rng(seed)
    att_idx = np.stack([np.nonzero(a)[0] for a in g.attached])       # (N, W)
    q = all_queries(g)
    base = patched_probs(model, g, K, -1, None, qnodes=q, device=device)  # k=-1 -> no patch
    out_p = torch.zeros(N, W, q.shape[1], device=device)
    out_base = torch.zeros(N, W, q.shape[1], device=device)
    thoughts = [] if keep_thoughts else None
    stats = {"diff_norm": [], "proj_norm": []}
    for w in range(W):
        br = att_idx[:, w]
        g_don = move_branch(g, br, to_start=False, mode=donor_mode, rng=rng)
        if direction == "remove":
            g_rec, g_src = g, g_don
            b_rec = base
        else:
            g_rec, g_src = g_don, g
            b_rec = patched_probs(model, g_rec, K, -1, None, qnodes=q, device=device)
        h_rec = run_thoughts(model, g_rec, K, device=device)[:, k]
        h_don = run_thoughts(model, g_src, K, device=device)[:, k]
        U = make_basis("probe" if kind == "probe" else ("random" if kind in ("random", "normrand") else kind),
                       h_rec, h_don, g, br, k, probe={"pca": pca} if kind == "pca" else probe,
                       rank=rank, gen=gen)
        new_h = apply_patch(h_rec, h_don, U, kind, gen=gen)
        stats["diff_norm"].append((h_don - h_rec).norm(dim=1).mean().item())
        stats["proj_norm"].append((new_h - h_rec).norm(dim=1).mean().item())
        r = patched_probs(model, g_rec, K, k, new_h, qnodes=q, device=device, keep=keep_thoughts)
        if keep_thoughts:
            r, th = r
            thoughts.append(th.cpu())
        out_p[:, w] = r
        out_base[:, w] = b_rec
    br_meta, d_meta = query_meta(g)
    res = {"W": W, "k": k, "kind": kind, "direction": direction, "att_idx": att_idx,
           "base": out_base.cpu().numpy(), "patched": out_p.cpu().numpy(),
           "q_branch": br_meta, "q_dist": d_meta, "bdepth": np.concatenate(
               [g.bdepth.reshape(N, -1), np.zeros((N, 1), int)], 1),
           "stats": {k_: float(np.mean(v)) for k_, v in stats.items()}}
    if keep_thoughts:
        res["thoughts"] = torch.stack(thoughts, 1).numpy()
    return res


@torch.no_grad()
def upfront_effects(model, g, K, k, kind="full", probe=None, device="cuda", seed=0, rank=None, pca=None):
    """Conventional format (target in context before the latents).

    Target = a node of one attached branch b_t (random depth). For every attached
    branch i, donor detaches i; patch thought k of the target-conditioned run.
    Diagonal (i == b_t) should flip the answer to 'unreachable'; off-diagonal not.
    Returns base P (N,), patched P (N, W), and which slot holds the target branch.
    """
    W = int(g.width[0])
    N = len(g)
    rng = np.random.default_rng(seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    att_idx = np.stack([np.nonzero(a)[0] for a in g.attached])
    slot_t = rng.integers(0, W, N)
    bt = att_idx[np.arange(N), slot_t]
    jt = rng.integers(0, NPB, N)
    target = g.nodes[np.arange(N), bt, jt]
    th_rec = run_thoughts(model, g, K, target=target, device=device)
    base = patched_probs(model, g, K, -1, None, target=target, device=device)[:, 0]
    pat = torch.zeros(N, W, device=device)
    for w in range(W):
        br = att_idx[:, w]
        g_don = move_branch(g, br, to_start=False, rng=rng)
        h_rec = th_rec[:, k]
        h_don = run_thoughts(model, g_don, K, target=target, device=device)[:, k]
        U = make_basis("probe" if kind == "probe" else ("random" if kind in ("random", "normrand") else kind),
                       h_rec, h_don, g, br, k, probe={"pca": pca} if kind == "pca" else probe,
                       rank=rank, gen=gen)
        new_h = apply_patch(h_rec, h_don, U, kind, gen=gen)
        pat[:, w] = patched_probs(model, g, K, k, new_h, target=target, device=device)[:, 0]
    return {"W": W, "k": k, "kind": kind, "base": base.cpu().numpy(), "patched": pat.cpu().numpy(),
            "slot_t": slot_t, "target_bdepth": jt}


@torch.no_grad()
def control_effects(model, g, K, k, control, device="cuda", seed=0):
    """Negative controls on the delayed format; returns mean |dP| over all queries per graph.

    identity    : donor = recipient (must be exactly 0)
    same_status : donor re-grafts one *decoy* branch onto another decoy branch
                  (one token changes, no node changes reachability)
    """
    N = len(g)
    rng = np.random.default_rng(seed)
    q = all_queries(g)
    base = patched_probs(model, g, K, -1, None, qnodes=q, device=device)
    h_rec = run_thoughts(model, g, K, device=device)[:, k]
    if control == "identity":
        h_don = h_rec.clone()
    elif control == "same_status":
        free = ~g.attached
        assert (free.sum(1) >= 2).all()
        order = np.argsort(np.where(free, rng.random(free.shape), 2.0), 1)
        b_move, b_host = order[:, 0], order[:, 1]
        src = g.src.copy()
        bi = np.arange(N)
        src[bi, g.hub_pos[bi, b_move]] = g.nodes[bi, b_host, rng.integers(0, NPB, N)]
        from dataclasses import replace
        g_don = replace(g, src=src)
        h_don = run_thoughts(model, g_don, K, device=device)[:, k]
    else:
        raise ValueError(control)
    pat = patched_probs(model, g, K, k, h_don, qnodes=q, device=device)
    return (pat - base).abs().mean(1).cpu().numpy(), (h_don - h_rec).norm(dim=1).mean().item()


def summarize(res, min_bdepth=0):
    """Targeted / collateral effects in probability units, per graph.

    Sign convention: effect toward the graph-defined counterfactual answer.
      remove: targeted = P_base - P_patched on edited-branch nodes (want > 0)
      insert: targeted = P_patched - P_base on edited-branch nodes (want > 0)
    collateral_* = mean |P_patched - P_base| on other reachable branches / decoy nodes.
    min_bdepth filters queried branch nodes by within-branch depth (0 = root).
    """
    base, pat = res["base"], res["patched"]
    N, W, Qn = pat.shape
    qb, bd = res["q_branch"], res["bdepth"]
    sign = 1.0 if res["direction"] == "remove" else -1.0
    tgt = np.full((N, W), np.nan)
    col_a = np.full((N, W), np.nan)
    col_s = np.full((N, W), np.nan)     # signed: >0 other branches pushed toward 'unreachable'
    col_d = np.full((N, W), np.nan)
    mat = np.full((N, W, W), np.nan)        # edited slot x queried slot
    dec_mask = qb == -1
    for n in range(N):
        att = res["att_idx"][n]
        attset = set(att.tolist())
        decoy_q = np.array([b not in attset for b in qb[n]])
        deep = bd[n] >= min_bdepth
        for w in range(W):
            delta = base[n, w] - pat[n, w]              # >0 : moved toward "unreachable"
            for v in range(W):
                m = (qb[n] == att[v]) & deep
                mat[n, w, v] = sign * delta[m].mean() if v == w else np.abs(delta[m]).mean()
            me = (qb[n] == att[w]) & deep
            tgt[n, w] = sign * delta[me].mean()
            others = np.isin(qb[n], att[att != att[w]]) & deep
            col_a[n, w] = np.abs(delta[others]).mean() if others.any() else np.nan
            col_s[n, w] = delta[others].mean() if others.any() else np.nan
            col_d[n, w] = np.abs(delta[decoy_q]).mean()
    return {"targeted": tgt, "collateral_attached": col_a, "collateral_signed": col_s,
            "collateral_decoy": col_d, "matrix": mat}


def bootstrap_ci(x, n_boot=2000, seed=0, axis_units=None):
    """Percentile CI of the mean over independent graphs (rows). x: (N,) or (N, ...) -> mean over rest."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim > 1:
        x = np.nanmean(x.reshape(len(x), -1), 1)
    x = x[~np.isnan(x)]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (n_boot, len(x)))
    bs = x[idx].mean(1)
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
