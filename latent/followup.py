"""Follow-up experiments (docs/NEXT_STEPS.md): record-level measurements.

Every function returns per-graph arrays so that effects, competence strata, recovery ratios
and uncertainty are computed afterwards (latent/followup_analysis.py) from the same records.
All queries of one graph are forks of one saved state; the graph is the sampling unit.
"""
from __future__ import annotations

import os
from dataclasses import replace

import numpy as np
import torch

from .competence import load
from .flex import START, cache_entries, forward
from .graphs import NB, NPB, move_branch
from .interventions import all_queries, query_meta
from .probes import apply_probe, probe_directions

# ------------------------------------------------------------------ assets / helpers


def load_assets(run, device):
    model, c = load(run, device)
    d = os.path.join(run, "discovery")
    probes = [{k: v.to(device) for k, v in p.items()} for p in torch.load(os.path.join(d, "probes_reach.pt"), weights_only=False)]
    pats = torch.load(os.path.join(d, "patterns.pt"))
    for i, p in enumerate(probes):
        p["pattern"] = pats[i].to(device)
    pca = torch.load(os.path.join(d, "pca_bases.pt"))
    import json
    k_star = json.load(open(os.path.join(d, "discovery.json")))["k_star"]
    return {"model": model, "K": c["K"], "probes": probes, "pca": pca, "k_star": k_star, "device": device, "mode": c["mode"]}


def T(a, dev):
    return torch.as_tensor(a, device=dev)


def run(assets, g, q, record=False, **kw):
    dev = assets["device"]
    return forward(assets["model"], T(g.src, dev), T(g.dst, dev), T(g.start, dev), assets["K"], T(q, dev),
                   record=record, **kw)


def orth_masked(A):
    """(N, d, m) -> orthonormal basis of the column space with rank-deficient columns zeroed."""
    U, S, _ = torch.linalg.svd(A, full_matrices=False)
    mask = (S > 1e-6 * S[:, :1]).to(U.dtype)
    return U * mask[:, None, :]


def basis(assets, kind, k, ids=None, N=None, rank=None, gen=None):
    """Edit subspace U (N, d, r) or None for the full space."""
    dev = assets["device"]
    d = assets["model"].d
    if kind == "full":
        return None
    if kind == "pattern":
        A = assets["probes"][k]["pattern"]                                    # (d, V)
        return orth_masked(A.T[T(ids, dev)].transpose(1, 2))
    if kind == "filter":
        A = probe_directions(assets["probes"][k])
        return orth_masked(A.T[T(ids, dev)].transpose(1, 2))
    if kind == "pca":
        return assets["pca"][k][:, :rank].to(dev).expand(N, -1, -1)
    if kind == "random":
        return orth_masked(torch.randn(N, d, rank, device=dev, generator=gen))
    raise ValueError(kind)


def project(U, diff):
    if U is None:
        return diff
    return (U @ (U.transpose(1, 2) @ diff[:, :, None]))[:, :, 0]


def branch_ids(g, branch):
    return g.nodes[np.arange(len(g)), branch]                                # (N, NPB)


# ------------------------------------------------------------------ Experiment 1: dose response


def dose_records(assets, g, k, kinds=("full", "pattern", "filter", "pca20", "random6"),
                 doses=(0.0, 0.1, 0.25, 0.5, 1.0, 1.5), direction="remove", seed=0):
    """For every reachable branch slot: clean recipient, clean donor, natural and projected
    differences, and patched logits at each dose for each edit kind."""
    N, W = len(g), int(g.width[0])
    gen = torch.Generator(device=assets["device"]).manual_seed(seed)
    att = np.stack([np.nonzero(a)[0] for a in g.attached])
    q = all_queries(g)
    Qn = q.shape[1]
    out = {"base": np.zeros((N, W, Qn), np.float32), "don": np.zeros((N, W, Qn), np.float32),
           "pat": np.zeros((len(kinds), len(doses), N, W, Qn), np.float32),
           "diff_norm": np.zeros((N, W), np.float32), "proj_norm": np.zeros((len(kinds), N, W), np.float32),
           "h_norm": np.zeros((N, W), np.float32)}
    for w in range(W):
        br = att[:, w]
        g_d = move_branch(g, br, to_start=False)
        g_rec, g_src = (g, g_d) if direction == "remove" else (g_d, g)
        r_rec = run(assets, g_rec, q, record=True)
        r_src = run(assets, g_src, q, record=True)
        out["base"][:, w] = r_rec["logit"].float().cpu().numpy()
        out["don"][:, w] = r_src["logit"].float().cpu().numpy()
        h, hd = r_rec["thoughts"][:, k], r_src["thoughts"][:, k]
        diff = hd - h
        out["diff_norm"][:, w] = diff.norm(dim=1).cpu().numpy()
        out["h_norm"][:, w] = h.norm(dim=1).cpu().numpy()
        for ki, kind in enumerate(kinds):
            base_kind = kind.rstrip("0123456789")
            rank = int(kind[len(base_kind):]) if kind != base_kind else None
            U = basis(assets, base_kind, k, ids=branch_ids(g, br), N=N, rank=rank, gen=gen)
            proj = project(U, diff)
            out["proj_norm"][ki, :, w] = proj.norm(dim=1).cpu().numpy()
            for di, lam in enumerate(doses):
                new = h + lam * proj
                out["pat"][ki, di, :, w] = run(assets, g_rec, q, thought_edit={k: (lambda _h, t=new: t)})["logit"].float().cpu().numpy()
    qb, qd = query_meta(g)
    out.update({"att_idx": att, "q_branch": qb, "q_dist": qd, "kinds": np.array(kinds), "doses": np.array(doses),
                "direction": direction, "k": k, "W": W})
    return out


# ------------------------------------------------------------------ Experiment 2: fixed-width swaps


def _regraft_decoy(g, rng):
    """Same-status control: one decoy branch re-parented onto another decoy branch's node."""
    N = len(g)
    free = ~g.attached
    order = np.argsort(np.where(free, rng.random(free.shape), 2.0), 1)
    bm, bh = order[:, 0], order[:, 1]
    src = g.src.copy()
    bi = np.arange(N)
    src[bi, g.hub_pos[bi, bm]] = g.nodes[bi, bh, rng.integers(0, NPB, N)]
    return replace(g, src=src), bm


def swap_records(assets, g, k, kinds=("full", "pattern", "pca20"), doses=(0.5, 1.0, 1.5), n_pairs=3, seed=0):
    """A (reachable) and B (shape-matched unreachable) exchange hub parents: width fixed.
    Compared on the same recipient with A-only removal, B-only insertion, and a same-status edit."""
    N, W = len(g), int(g.width[0])
    rng = np.random.default_rng(seed)
    gen = torch.Generator(device=assets["device"]).manual_seed(seed)
    q = all_queries(g)
    Qn = q.shape[1]
    base = run(assets, g, q, record=True)
    h = base["thoughts"][:, k]
    conds = ["swap", "remove_A", "insert_B", "same_status"]
    out = {"base": base["logit"].float().cpu().numpy(),
           "don": np.zeros((len(conds), n_pairs, N, Qn), np.float32),
           "pat": np.zeros((len(conds), len(kinds), len(doses), n_pairs, N, Qn), np.float32),
           "A": np.zeros((n_pairs, N), int), "B": np.zeros((n_pairs, N), int), "C": np.zeros((n_pairs, N), int),
           "diff_norm": np.zeros((len(conds), n_pairs, N), np.float32)}
    for pi in range(n_pairs):
        A = np.argmax(np.where(g.attached, rng.random(g.attached.shape), -1), 1)
        B = np.argmax(np.where(~g.attached, rng.random(g.attached.shape), -1), 1)
        out["A"][pi], out["B"][pi] = A, B
        g_swap = move_branch(move_branch(g, A, to_start=False), B, to_start=True)
        g_rem = move_branch(g, A, to_start=False)
        g_ins = move_branch(g, B, to_start=True)
        g_same, C = _regraft_decoy(g, rng)
        out["C"][pi] = C
        for ci, gd in enumerate([g_swap, g_rem, g_ins, g_same]):
            rd = run(assets, gd, q, record=True)
            out["don"][ci, pi] = rd["logit"].float().cpu().numpy()
            diff = rd["thoughts"][:, k] - h
            out["diff_norm"][ci, pi] = diff.norm(dim=1).cpu().numpy()
            ids = {"swap": np.concatenate([branch_ids(g, A), branch_ids(g, B)], 1), "remove_A": branch_ids(g, A),
                   "insert_B": branch_ids(g, B), "same_status": branch_ids(g, C)}[conds[ci]]
            for ki, kind in enumerate(kinds):
                bk = kind.rstrip("0123456789")
                rank = int(kind[len(bk):]) if kind != bk else None
                U = basis(assets, bk, k, ids=ids, N=N, rank=rank, gen=gen)
                proj = project(U, diff)
                for di, lam in enumerate(doses):
                    new = h + lam * proj
                    out["pat"][ci, ki, di, pi] = run(assets, g, q, thought_edit={k: (lambda _h, t=new: t)})["logit"].float().cpu().numpy()
    qb, qd = query_meta(g)
    out.update({"q_branch": qb, "q_dist": qd, "attached": g.attached, "conds": np.array(conds),
                "kinds": np.array(kinds), "doses": np.array(doses), "k": k, "W": W})
    return out


# ------------------------------------------------------------------ semantic continuation test


def _descendants(g, n, b, j):
    """Within-branch indices of node j and all its descendants in branch b of graph n."""
    kids = {i: [] for i in range(NPB)}
    for i in range(1, NPB):
        kids[int(g.parent[n, b, i])].append(i)
    out, stack = [], [j]
    while stack:
        u = stack.pop()
        out.append(u)
        stack.extend(kids[u])
    return out


def continuation_records(assets, g, k, j=2, kinds=("full", "pattern", "pca20"), seed=0):
    """Donor and recipient disagree about what lies below depth j of branch B.

    Base G: B attached, decoy D. Swap the parents of c in B (depth j) and c' in D (depth j).
      recipient R  = swapped graph, B detached
      donor     Dn = original G (B attached, c below x)
      natural   Nt = swapped graph, B attached (the recipient continuation, run cleanly)
      wrong     Wr = R with a different decoy branch attached (wrong-branch donor)
    Edit: insert B's state from Dn (or Wr) into R at thought k; query:
      C_rec = c' and descendants  (reachable under the recipient continuation)
      C_don = c  and descendants  (reachable only in the donor's future)
      B_sh  = B's nodes above the swap (reachable in Dn and Nt)
    """
    rng = np.random.default_rng(seed)
    N0 = len(g)
    keep, rows = [], []
    for n in range(N0):
        Bs = np.nonzero(g.attached[n])[0]
        Ds = np.nonzero(~g.attached[n])[0]
        rng.shuffle(Bs); rng.shuffle(Ds)
        found = None
        for b in Bs:
            cb = [i for i in range(NPB) if g.bdepth[n, b, i] == j]
            if not cb:
                continue
            for dd in Ds:
                cd = [i for i in range(NPB) if g.bdepth[n, dd, i] == j]
                others = [e for e in Ds if e != dd]
                if cd and others:
                    found = (b, dd, cb[rng.integers(len(cb))], cd[rng.integers(len(cd))], others[rng.integers(len(others))])
                    break
            if found:
                break
        if found:
            keep.append(n); rows.append(found)
    g = g.subset(np.array(keep))
    N = len(g)
    bi = np.arange(N)
    b, dd, jc, jc2, e = (np.array(x) for x in zip(*rows))
    c, c2 = g.nodes[bi, b, jc], g.nodes[bi, dd, jc2]
    x, y = g.nodes[bi, b, g.parent[bi, b, jc]], g.nodes[bi, dd, g.parent[bi, dd, jc2]]
    src = g.src.copy()
    pos_c = np.argmax(g.dst == c[:, None], 1)
    pos_c2 = np.argmax(g.dst == c2[:, None], 1)
    src[bi, pos_c] = y
    src[bi, pos_c2] = x
    g_sw = replace(g, src=src)
    g_R = move_branch(g_sw, b, to_start=False)
    g_N = g_sw
    g_Wr = move_branch(g_R, e, to_start=True)
    # query sets (ids)
    C_rec = [g.nodes[n, dd[n], _descendants(g, n, dd[n], jc2[n])] for n in range(N)]
    C_don = [g.nodes[n, b[n], _descendants(g, n, b[n], jc[n])] for n in range(N)]
    B_sh = [np.setdiff1d(g.nodes[n, b[n]], C_don[n]) for n in range(N)]
    # ground truth check with the exact solver on the constructed graphs
    from .graphs import UNREACH, bfs_dist
    for nm, gg, want_rec, want_don in (("R", g_R, False, False), ("Nt", g_N, True, False), ("Dn", g, False, True)):
        dd_ = bfs_dist(gg.src[:20], gg.dst[:20], gg.start[:20])
        for n in range(min(20, N)):
            r_rec = (dd_[n, C_rec[n]] < UNREACH).all() if want_rec else (dd_[n, C_rec[n]] == UNREACH).all()
            r_don = (dd_[n, C_don[n]] < UNREACH).all() if want_don else (dd_[n, C_don[n]] == UNREACH).all()
            assert r_rec and r_don, (nm, n)
    q = all_queries(g)
    member = lambda sets: np.stack([np.isin(q[n], sets[n]) for n in range(N)])
    masks = {"C_rec": member(C_rec), "C_don": member(C_don), "B_shared": member(B_sh)}
    other_att = np.zeros_like(masks["C_rec"])
    qb, _ = query_meta(g)
    for n in range(N):
        oa = [i for i in np.nonzero(g.attached[n])[0] if i != b[n]]
        other_att[n] = np.isin(qb[n], oa)
    masks["other_attached"] = other_att & ~masks["C_rec"] & ~masks["C_don"]
    rR = run(assets, g_R, q, record=True)
    rD = run(assets, g, q, record=True)
    rN = run(assets, g_N, q)
    rW = run(assets, g_Wr, q, record=True)
    h = rR["thoughts"][:, k]
    out = {"R": rR["logit"].float().cpu().numpy(), "Dn": rD["logit"].float().cpu().numpy(),
           "Nt": rN["logit"].float().cpu().numpy(), "Wr": rW["logit"].float().cpu().numpy()}
    gen = torch.Generator(device=assets["device"]).manual_seed(seed)
    ids_sh = np.stack([np.pad(s, (0, NPB - len(s)), mode="edge") for s in B_sh])      # (N, NPB) pad by repetition
    for kind in kinds:
        bk = kind.rstrip("0123456789")
        rank = int(kind[len(bk):]) if kind != bk else None
        U = basis(assets, bk, k, ids=ids_sh, N=N, rank=rank, gen=gen)
        for src_name, rs in (("Dn", rD), ("Wr", rW)):
            new = h + project(U, rs["thoughts"][:, k] - h)
            out[f"edit_{kind}_{src_name}"] = run(assets, g_R, q, thought_edit={k: (lambda _h, t=new: t)})["logit"].float().cpu().numpy()
    out.update({f"mask_{m}": v for m, v in masks.items()})
    out.update({"k": k, "j": j, "W": int(g.width[0]), "n": N})
    return out


def continuation_pair_records(assets, g, k, j=2, seed=0):
    """Composed edits: two branches B1, B2, each with a conflicting continuation (parent swap with
    its own decoy D1, D2 at depth j), both detached in the recipient; one donor state with both
    attached is transferred at thought k. Also single-branch edits for each, for additivity."""
    rng = np.random.default_rng(seed)
    keep, rows = [], []
    for n in range(len(g)):
        Bs = [b for b in np.nonzero(g.attached[n])[0] if (g.bdepth[n, b] == j).any()]
        Ds = [d_ for d_ in np.nonzero(~g.attached[n])[0] if (g.bdepth[n, d_] == j).any()]
        if len(Bs) >= 2 and len(Ds) >= 2:
            rng.shuffle(Bs); rng.shuffle(Ds)
            pick = lambda b: [i for i in range(NPB) if g.bdepth[n, b, i] == j][0]
            rows.append((Bs[0], Bs[1], Ds[0], Ds[1], pick(Bs[0]), pick(Bs[1]), pick(Ds[0]), pick(Ds[1])))
            keep.append(n)
    g = g.subset(np.array(keep))
    N = len(g)
    bi = np.arange(N)
    b1, b2, d1, d2, i1, i2, e1, e2 = (np.array(x) for x in zip(*rows))
    src = g.src.copy()
    sets = {}
    for tag, b, dd, ib, idd in (("1", b1, d1, i1, e1), ("2", b2, d2, i2, e2)):
        c, c2 = g.nodes[bi, b, ib], g.nodes[bi, dd, idd]
        x, y = g.nodes[bi, b, g.parent[bi, b, ib]], g.nodes[bi, dd, g.parent[bi, dd, idd]]
        src[bi, np.argmax(g.dst == c[:, None], 1)] = y
        src[bi, np.argmax(g.dst == c2[:, None], 1)] = x
        sets[f"C_rec{tag}"] = [g.nodes[n, dd[n], _descendants(g, n, dd[n], idd[n])] for n in range(N)]
        sets[f"C_don{tag}"] = [g.nodes[n, b[n], _descendants(g, n, b[n], ib[n])] for n in range(N)]
    g_sw = replace(g, src=src)
    g_R = move_branch(move_branch(g_sw, b1, False), b2, False)
    g_N = g_sw
    q = all_queries(g)
    masks = {s: np.stack([np.isin(q[n], v[n]) for n in range(N)]) for s, v in sets.items()}
    rR = run(assets, g_R, q, record=True)
    rD = run(assets, g, q, record=True)                                    # both attached, original subtrees
    rD1 = run(assets, move_branch(g, b2, False), q, record=True)           # only B1 attached
    rD2 = run(assets, move_branch(g, b1, False), q, record=True)           # only B2 attached
    rN = run(assets, g_N, q)
    h = rR["thoughts"][:, k]
    out = {"R": rR["logit"].float().cpu().numpy(), "Nt": rN["logit"].float().cpu().numpy()}
    for name, rs in (("both", rD), ("only1", rD1), ("only2", rD2)):
        new = rs["thoughts"][:, k]
        out[f"edit_{name}"] = run(assets, g_R, q, thought_edit={k: (lambda _h, t=new: t)})["logit"].float().cpu().numpy()
    out.update({f"mask_{m}": v for m, v in masks.items()})
    out.update({"k": k, "j": j, "W": int(g.width[0]), "n": N})
    return out


@torch.no_grad()
def amplify_records(assets, g, k, lams=(0.0, 0.5, 1.5, 2.0), n_slots=3, seed=0):
    """No donor: rescale one branch's own component of the saved thought, h' = h + (lam-1) U U^T h,
    with U = that branch's pattern subspace (6 ids); controls: a decoy branch's pattern subspace and
    a random rank-6 subspace."""
    N, W = len(g), int(g.width[0])
    gen = torch.Generator(device=assets["device"]).manual_seed(seed)
    rng = np.random.default_rng(seed)
    q = all_queries(g)
    base = run(assets, g, q, record=True)
    h = base["thoughts"][:, k]
    att = np.stack([np.nonzero(a)[0] for a in g.attached])
    dec = np.argmax(np.where(~g.attached, rng.random(g.attached.shape), -1), 1)
    ns = min(n_slots, W)
    out = {"base": base["logit"].float().cpu().numpy(),
           "att": np.zeros((len(lams), ns, N, q.shape[1]), np.float32),
           "dec": np.zeros((len(lams), N, q.shape[1]), np.float32),
           "rnd": np.zeros((len(lams), N, q.shape[1]), np.float32)}
    U_dec = basis(assets, "pattern", k, ids=branch_ids(g, dec), N=N)
    U_rnd = basis(assets, "random", k, N=N, rank=NPB, gen=gen)
    for li, lam in enumerate(lams):
        for si in range(ns):
            U = basis(assets, "pattern", k, ids=branch_ids(g, att[:, si]), N=N)
            new = h + (lam - 1.0) * project(U, h)
            out["att"][li, si] = run(assets, g, q, thought_edit={k: (lambda _h, t=new: t)})["logit"].float().cpu().numpy()
        for key, U in (("dec", U_dec), ("rnd", U_rnd)):
            new = h + (lam - 1.0) * project(U, h)
            out[key][li] = run(assets, g, q, thought_edit={k: (lambda _h, t=new: t)})["logit"].float().cpu().numpy()
    qb, qd = query_meta(g)
    out.update({"att_idx": att[:, :ns], "dec_idx": dec, "q_branch": qb, "q_dist": qd, "attached": g.attached,
                "lams": np.array(lams), "W": W, "k": k})
    return out


@torch.no_grad()
def upfront_cache_records(assets, g, k, seed=0):
    """Route decomposition in the conventional (target-upfront) format. Target = a node of one
    reachable branch b_t; donor detaches b_t (the answer flips). Routes as in cache_records, with
    positions shifted by the TARGET token (START=60, TARGET=61, LAT_j=61+j)."""
    from .graphs import E
    dev = assets["device"]
    N = len(g)
    rng = np.random.default_rng(seed)
    att = np.stack([np.nonzero(a)[0] for a in g.attached])
    bt = att[np.arange(N), rng.integers(0, att.shape[1], N)]
    jt = rng.integers(0, NPB, N)
    tgt = g.nodes[np.arange(N), bt, jt]
    gd = move_branch(g, bt, False)
    runT = lambda gg, **kw: forward(assets["model"], T(gg.src, dev), T(gg.dst, dev), T(gg.start, dev), assets["K"],
                                    None, target=T(tgt, dev), **kw)
    base, don = runT(g, record=True), runT(gd, record=True)
    ov = lambda r, pos: {p: cache_entries(r["cache"], p) for p in pos}
    Lpos, Gpos = range(START, START + 1 + k + 1), range(0, START)
    hk = don["thoughts"][:, k]
    te = {k: (lambda _h, t=hk: t)}
    spec = {"T": {"thought_edit": te}, "L": {"kv_override": ov(don, Lpos)},
            "TL": {"thought_edit": te, "kv_override": ov(don, Lpos)}, "Gq": {"query_kv_override": ov(don, Gpos)},
            "TLGq": {"thought_edit": te, "kv_override": ov(don, Lpos), "query_kv_override": ov(don, Gpos)},
            "clamp": {"thought_edit": {kk: (lambda _h, t=don["thoughts"][:, kk]: t) for kk in range(k, assets["K"])}}}
    out = {"base": base["logit"][:, 0].float().cpu().numpy(), "don": don["logit"][:, 0].float().cpu().numpy()}
    for c, kw in spec.items():
        out[c] = runT(g, **kw)["logit"][:, 0].float().cpu().numpy()
    out.update({"target_dist": jt + 1, "W": int(g.width[0]), "k": k})
    return out


@torch.no_grad()
def gain_records(assets, g, k):
    """Natural branch-signal amplitude: |h_k(donor) - h_k(recipient)| for single-branch removal,
    and layer-1 attention excess per reachable branch; one random reachable branch per graph."""
    N, W = len(g), int(g.width[0])
    q = all_queries(g)[:, :1]
    rng = np.random.default_rng(0)
    br = np.argmax(np.where(g.attached, rng.random(g.attached.shape), -1), 1)
    ra = run(assets, g, q, record=True)
    rd = run(assets, move_branch(g, br, False), q, record=True)
    return {"W": W, "diff_norm": (rd["thoughts"][:, k] - ra["thoughts"][:, k]).norm(dim=1).cpu().numpy(),
            "h_norm": ra["thoughts"][:, k].norm(dim=1).cpu().numpy()}


# ------------------------------------------------------------------ cache recovery factorial


def cache_records(assets, g, k, n_slots=3, seed=0):
    """Remove branch A. Routes: thought h_k (T), latent cache START..START+k (L), graph tokens (G).
    Conditions over {T, L, G} plus 'clamp' (donor thought fed at every step >= k) and
    'L_wrong' (latent cache from a donor that removed a different branch)."""
    N, W = len(g), int(g.width[0])
    rng = np.random.default_rng(seed)
    q = all_queries(g)
    recurrent = assets["mode"] == "coconut"
    conds = (["base", "L", "G", "LG", "Gq", "Gl"]
             + (["T", "TL", "TG", "TLG", "clamp", "clampL", "TLGq", "TLGl"] if recurrent else ["LGq"])
             + (["L_wrong"] if W > 1 else []))
    K = assets["K"]
    out = {c: np.zeros((min(n_slots, W), N, q.shape[1]), np.float32) for c in conds + ["don"]}
    margins = {c: np.zeros((min(n_slots, W), N, K + 1), np.float32) for c in conds + ["don"]}
    att = np.stack([np.nonzero(a)[0] for a in g.attached])
    base = run(assets, g, q, record=True)
    probes = assets["probes"]

    def margin(th, br):
        if probes is None:                      # checkpoints without probe assets (development study)
            return np.full((N, K + 1), np.nan, np.float32)
        ids = branch_ids(g, br)
        dec = np.where(~g.attached[:, :, None], g.nodes, -1).reshape(N, -1)
        m = np.zeros((N, K + 1), np.float32)
        for kk in range(K + 1):
            s = apply_probe(probes[kk], th[:, kk]).cpu().numpy()
            for n in range(N):
                m[n, kk] = s[n, ids[n]].mean() - s[n, dec[n][dec[n] >= 0]].mean()
        return m

    for si in range(min(n_slots, W)):
        A = att[:, si]
        g_d = move_branch(g, A, to_start=False)
        rd = run(assets, g_d, q, record=True)
        others = np.array([[x for x in row if x != a] for row, a in zip(att, A)]) if W > 1 else None
        if W > 1:
            g_w = move_branch(g, others[np.arange(N), rng.integers(0, W - 1, N)], to_start=False)
            rw = run(assets, g_w, q, record=True)
        Lpos = range(START, START + k + 1)
        Gpos = range(0, START)
        ov = lambda r, pos: {p: cache_entries(r["cache"], p) for p in pos}
        spec = {"base": {}, "L": {"kv_override": ov(rd, Lpos)}, "G": {"kv_override": ov(rd, Gpos)},
                "LG": {"kv_override": {**ov(rd, Gpos), **ov(rd, Lpos)}}}
        if recurrent:
            hk = rd["thoughts"][:, k]
            te = {k: (lambda _h, t=hk: t)}
            clamp = {kk: (lambda _h, t=rd["thoughts"][:, kk]: t) for kk in range(k, K)}
            spec.update({"T": {"thought_edit": te}, "TL": {"thought_edit": te, "kv_override": ov(rd, Lpos)},
                         "TG": {"thought_edit": te, "kv_override": ov(rd, Gpos)},
                         "TLG": {"thought_edit": te, "kv_override": {**ov(rd, Gpos), **ov(rd, Lpos)}},
                         "clamp": {"thought_edit": clamp}, "clampL": {"thought_edit": clamp, "kv_override": ov(rd, Lpos)}})
        # split the graph route by reader: queries only (Gq) vs later latents only (Gl)
        spec["Gq"] = {"query_kv_override": ov(rd, Gpos)}
        spec["Gl"] = {"kv_override": ov(rd, Gpos), "query_kv_override": ov(base, Gpos)}
        if recurrent:
            spec["TLGq"] = {"thought_edit": te, "kv_override": ov(rd, Lpos), "query_kv_override": ov(rd, Gpos)}
            spec["TLGl"] = {"thought_edit": te, "kv_override": {**ov(rd, Gpos), **ov(rd, Lpos)},
                            "query_kv_override": ov(base, Gpos)}
        else:
            spec["LGq"] = {"kv_override": ov(rd, Lpos), "query_kv_override": ov(rd, Gpos)}
        if W > 1:
            spec["L_wrong"] = {"kv_override": ov(rw, Lpos)}
        for cnd, kw in spec.items():
            r = run(assets, g, q, record=True, **kw)
            out[cnd][si] = r["logit"].float().cpu().numpy()
            margins[cnd][si] = margin(r["thoughts"], A)
        out["don"][si] = rd["logit"].float().cpu().numpy()
        margins["don"][si] = margin(rd["thoughts"], A)
        out.setdefault("A", np.zeros((min(n_slots, W), N), int))[si] = A
    qb, qd = query_meta(g)
    out.update({f"margin_{c}": v for c, v in margins.items()})
    out.update({"q_branch": qb, "q_dist": qd, "attached": g.attached, "k": k, "W": W, "conds": np.array(conds)})
    return out


# ------------------------------------------------------------------ attention budget


def branch_edges(g):
    """(N, NB, E) bool: which edge tokens belong to each branch (hub edge + internal edges)."""
    from .graphs import E
    N = len(g)
    m = np.zeros((N, NB, E), bool)
    bi = np.arange(N)
    for b in range(NB):
        m[bi, b, g.hub_pos[:, b]] = True
        m[:, b] |= (g.dst[:, :, None] == g.nodes[:, b, None, 1:]).any(2)
    return m


def edge_mask(g, branches):
    return branch_edges(g)[np.arange(len(g)), branches]

def key_bias_from_mask(mask, value, device, layers=(0,), T_max=None):
    from .graphs import E
    T_max = T_max or (E + 16)
    b = torch.zeros(mask.shape[0], T_max, device=device)
    b[:, :E] = torch.as_tensor(mask, device=device).float() * value
    return {li: b for li in layers}


def attention_records(assets, g, doses=(0.5, 1.0, 2.0), layers=(0,), n_slots=3, seed=0):
    """Boost layer-1 attention (START + latent rows) onto one branch's edges.
    Targets: an attached branch, a decoy branch, or all edges uniformly (control)."""
    N, W = len(g), int(g.width[0])
    rng = np.random.default_rng(seed)
    q = all_queries(g)
    base = run(assets, g, q)["logit"].float().cpu().numpy()
    att = np.stack([np.nonzero(a)[0] for a in g.attached])
    dec = np.argmax(np.where(~g.attached, rng.random(g.attached.shape), -1), 1)
    ns = min(n_slots, W)
    out = {"base": base, "att_boost": np.zeros((len(doses), ns, N, q.shape[1]), np.float32),
           "dec_boost": np.zeros((len(doses), N, q.shape[1]), np.float32),
           "all_boost": np.zeros((len(doses), N, q.shape[1]), np.float32)}
    from .graphs import E
    allm = np.ones((N, E), bool)
    for di, b in enumerate(doses):
        for si in range(ns):
            kb = key_bias_from_mask(edge_mask(g, att[:, si]), b, assets["device"], layers)
            out["att_boost"][di, si] = run(assets, g, q, key_bias=kb)["logit"].float().cpu().numpy()
        out["dec_boost"][di] = run(assets, g, q, key_bias=key_bias_from_mask(edge_mask(g, dec), b, assets["device"], layers))["logit"].float().cpu().numpy()
        out["all_boost"][di] = run(assets, g, q, key_bias=key_bias_from_mask(allm, b, assets["device"], layers))["logit"].float().cpu().numpy()
    qb, qd = query_meta(g)
    out.update({"att_idx": att[:, :ns], "dec": dec, "q_branch": qb, "q_dist": qd, "attached": g.attached,
                "doses": np.array(doses), "W": W})
    return out


def gain_restore_records(assets, g, k, W_ref=2, layers=(0,), n_slots=3):
    """Dilution test: add log(W / W_ref) to layer-1 logits on every *reachable* branch's edges
    (in recipient and donor runs alike), so each branch's attention share approaches its
    W_ref share; then repeat the full-vector single-branch removal at thought k."""
    N, W = len(g), int(g.width[0])
    q = all_queries(g)
    att = np.stack([np.nonzero(a)[0] for a in g.attached])
    ns = min(n_slots, W)
    from .graphs import E
    out = {}
    for tag, val in (("plain", 0.0), ("restored", float(np.log(W / W_ref)))):
        def bias_for(gg, val=val):
            m = (branch_edges(gg) & gg.attached[:, :, None]).any(1)
            return key_bias_from_mask(m, val, assets["device"], layers)
        rb = run(assets, g, q, record=True, key_bias=bias_for(g))
        base = rb["logit"].float().cpu().numpy()
        pats, dons = np.zeros((ns, N, q.shape[1]), np.float32), np.zeros((ns, N, q.shape[1]), np.float32)
        for si in range(ns):
            gd = move_branch(g, att[:, si], to_start=False)
            rd = run(assets, gd, q, record=True, key_bias=bias_for(gd))
            dons[si] = rd["logit"].float().cpu().numpy()
            hk = rd["thoughts"][:, k]
            pats[si] = run(assets, g, q, key_bias=bias_for(g), thought_edit={k: (lambda _h, t=hk: t)})["logit"].float().cpu().numpy()
        out[f"{tag}_base"], out[f"{tag}_pat"], out[f"{tag}_don"] = base, pats, dons
    qb, qd = query_meta(g)
    out.update({"att_idx": att[:, :ns], "q_branch": qb, "q_dist": qd, "attached": g.attached, "W": W, "k": k})
    return out


# ------------------------------------------------------------------ local controllability


def jacobian_records(assets, g, k, rank=20, n_slots=3):
    """J = d(logits of all forked queries)/d(h_k) projected onto the shared PCA basis, plus the
    observed natural-dose full and pca-projected donor edits for the same (graph, branch)."""
    m, dev, K = assets["model"], assets["device"], assets["K"]
    N, W = len(g), int(g.width[0])
    q = all_queries(g)
    Qn = q.shape[1]
    base = run(assets, g, q, record=True)
    h0 = base["thoughts"][:, k].detach()
    U = assets["pca"][k][:, :rank].to(dev)                                           # (d, r)
    src, dst, start, qq = T(g.src, dev), T(g.dst, dev), T(g.start, dev), T(q, dev)
    with torch.enable_grad():
        hv = h0.clone().requires_grad_(True)
        logit = m.run(src, dst, start, K, qnodes=qq, patch=lambda kk, hh: hv if kk == k else hh)["logit"]
        J = torch.zeros(N, Qn, h0.shape[1], device=dev)
        for qi in range(Qn):
            gr, = torch.autograd.grad(logit[:, qi].sum(), hv, retain_graph=qi < Qn - 1)
            J[:, qi] = gr
    A = (J @ U).detach()                                                              # (N, Qn, r)
    att = np.stack([np.nonzero(a)[0] for a in g.attached])
    ns = min(n_slots, W)
    obs_full = np.zeros((ns, N, Qn), np.float32)
    obs_pca = np.zeros((ns, N, Qn), np.float32)
    u = np.zeros((ns, N, rank), np.float32)
    dnorm = np.zeros((ns, N), np.float32)
    for si in range(ns):
        rd = run(assets, move_branch(g, att[:, si], to_start=False), q, record=True)
        diff = rd["thoughts"][:, k] - h0
        dnorm[si] = diff.norm(dim=1).cpu().numpy()
        uu = diff @ U
        u[si] = uu.cpu().numpy()
        obs_full[si] = run(assets, g, q, thought_edit={k: (lambda _h, t=h0 + diff: t)})["logit"].float().cpu().numpy()
        obs_pca[si] = run(assets, g, q, thought_edit={k: (lambda _h, t=h0 + uu @ U.T: t)})["logit"].float().cpu().numpy()
    qb, qd = query_meta(g)
    return {"A": A.cpu().numpy(), "Jnorm": J.norm(dim=2).cpu().numpy(), "base": base["logit"].float().cpu().numpy(),
            "obs_full": obs_full, "obs_pca": obs_pca, "u": u, "diff_norm": dnorm, "att_idx": att[:, :ns],
            "q_branch": qb, "q_dist": qd, "attached": g.attached, "W": W, "k": k, "rank": rank}
