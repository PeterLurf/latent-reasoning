"""Instrument validation: generator vs exact solver, splits, counterfactual edits,
and model code paths (cache vs full recompute, forked queries, identity patch).

    python -m latent.tests
"""
from __future__ import annotations

import numpy as np
import torch

from .data import balanced_queries
from .graphs import (ABSENT, E, NB, NPB, SPLITS, UNREACH, WMAX, bfs_dist, move_branch,
                     sample_graphs)
from .model import T_LAT, LatentReasoner


def check_generator():
    rng = np.random.default_rng(0)
    g = sample_graphs(rng, 3000, split=None)
    ref = bfs_dist(g.src, g.dst, g.start)
    assert (ref == g.dist()).all(), "construction labels disagree with BFS"
    assert g.src.shape == (3000, E)
    # every present node appears; counts fixed by budget
    present = (ref != ABSENT).sum(1)
    assert (present == 2 + NB * NPB).all()
    # width = number of reachable branch roots
    assert ((ref[np.arange(3000)[:, None], g.nodes[:, :, 0]] == 1).sum(1) == g.width).all()
    depth = ref[(ref >= 1) & (ref < UNREACH)]
    hist = np.bincount(depth)
    # renaming + reordering invariance of the family id
    g2 = sample_graphs(np.random.default_rng(1), 20000, split=None)
    for s, r in SPLITS.items():
        frac = np.isin(g2.bucket, list(r)).mean()
        print(f"  split {s:9s} fraction {frac:.3f}")
    fams = {s: set(g2.family[np.isin(g2.bucket, list(r))].tolist()) for s, r in SPLITS.items()}
    assert not (fams["train"] & fams["confirm"]) and not (fams["train"] & fams["discovery"])
    print(f"  generator == BFS on 3000 graphs; reachable depth histogram {hist[1:].tolist()}")
    print(f"  distinct families in 20000 draws: {len(set(g2.family.tolist()))}")

    # balanced queries
    q, qd = balanced_queries(rng, g, 16)
    y = (qd >= 1) & (qd < UNREACH)
    assert abs(y.mean() - 0.5) < 1e-9 and (qd != ABSENT).all() and (qd != 0).all()
    # unreachable queries are shape-matched: decoy-branch nodes have same within-branch depths
    print("  balanced queries OK (50.0% reachable)")


def check_counterfactuals():
    rng = np.random.default_rng(2)
    g = sample_graphs(rng, 2000, widths=range(2, WMAX + 1), split=None)
    b = np.argmax(np.where(g.attached, rng.random(g.attached.shape), -1), 1)
    for mode in ("hub", "graft"):
        h = move_branch(g, b, to_start=False, mode=mode, rng=rng)
        diff = (h.src != g.src).sum(1)
        assert (diff == 1).all() and (h.dst == g.dst).all() and (h.start == g.start).all()
        ref = bfs_dist(h.src, h.dst, h.start)
        assert (ref == h.dist()).all(), mode
        d0 = g.dist()
        moved = g.nodes[np.arange(2000), b]
        changed = (d0 != ref)
        assert changed[np.arange(2000)[:, None], moved].all()
        other = changed.copy()
        other[np.arange(2000)[:, None], moved] = False
        assert not other.any(), "edit changed reachability outside the edited branch"
    back = move_branch(h, b, to_start=True)
    assert (bfs_dist(back.src, back.dst, back.start) == back.dist()).all()
    print("  single-token counterfactuals change exactly the edited branch (hub, graft, reverse)")


@torch.no_grad()
def check_model(device):
    torch.manual_seed(0)
    rng = np.random.default_rng(3)
    for mode in ("coconut", "pause"):
        m = LatentReasoner(64, 2, 4, mode=mode, d_id=16).to(device).eval()
        # make weights non-trivial
        for p in m.parameters():
            p.add_(torch.randn_like(p) * 0.05)
        g = sample_graphs(rng, 8, split=None)
        src, dst, start = (torch.as_tensor(a, device=device) for a in (g.src, g.dst, g.start))
        q, _ = balanced_queries(rng, g, 6)
        q = torch.as_tensor(q, device=device)
        K = 5
        o = m.run(src, dst, start, K, qnodes=q, keep=True)
        # (1) full causal recompute over [prefix, latent inputs] reproduces incremental thoughts
        x = m.embed_prefix(src, dst, start)
        te = m.type_emb.weight
        th = o["thoughts"]
        lat = [(m.pause.expand_as(th[:, 0]) if mode == "pause" else th[:, k - 1]) + te[T_LAT] + m.step_emb.weight[k]
               for k in range(1, K + 1)]
        full = torch.cat([x, torch.stack(lat, 1)], 1)
        h, _, _ = m.encode(full)
        assert torch.allclose(h, th[:, K], atol=1e-4), (h - th[:, K]).abs().max()
        # (2) forked queries == each query appended alone to its own sequence
        for j in range(q.shape[1]):
            xq = m.node_emb(q[:, j]) + te[4]
            h1, _, _ = m.encode(torch.cat([full, xq[:, None]], 1))
            lj = m.head(h1)[:, 0]
            assert torch.allclose(lj, o["logit"][:, j], atol=1e-4)
        # (3) identity patch leaves everything unchanged
        o2 = m.run(src, dst, start, K, qnodes=q, patch=lambda k, h: h.clone())
        assert torch.equal(o2["logit"], o["logit"])
        # (4) patching thought k leaves thoughts <= k untouched and changes later ones (coconut)
        def zero2(k, h):
            return torch.zeros_like(h) if k == 2 else h
        o3 = m.run(src, dst, start, K, qnodes=q, patch=zero2, keep=True)
        assert torch.equal(o3["thoughts"][:, :2], o["thoughts"][:, :2])
        if mode == "coconut":
            assert not torch.allclose(o3["thoughts"][:, 3], o["thoughts"][:, 3])
        else:
            assert torch.allclose(o3["thoughts"][:, 3:], o["thoughts"][:, 3:], atol=1e-6)
        # (5) shared-prefix training path == separate runs for both formats
        tgt = q[:, 0]
        ld, lu, _, _ = m.run_both(src, dst, start, K, q, tgt)
        assert torch.allclose(ld, o["logit"], atol=1e-4)
        assert torch.allclose(lu, m.run(src, dst, start, K, target=tgt)["logit"][:, 0], atol=1e-4)
        # (6) batch composition does not change results
        o4 = m.run(src[:3], dst[:3], start[:3], K, qnodes=q[:3])
        assert torch.allclose(o4["logit"], o["logit"][:3], atol=1e-5)
    print("  cache==recompute, fork==separate sequences, identity patch exact, patch locality, batch invariance")


@torch.no_grad()
def check_flex(device):
    """flex.forward == model.run; full-route override reproduces the donor run exactly."""
    from .flex import START, cache_entries, forward
    from .graphs import move_branch
    torch.manual_seed(0)
    rng = np.random.default_rng(5)
    for mode in ("coconut", "pause"):
        m = LatentReasoner(64, 2, 4, mode=mode, d_id=16).to(device).eval()
        for p in m.parameters():
            p.add_(torch.randn_like(p) * 0.05)
        g = sample_graphs(rng, 16, widths=[4], split=None)
        T = lambda a: torch.as_tensor(a, device=device)
        q = T(balanced_queries(rng, g, 6)[0])
        K = 5
        ref = m.run(T(g.src), T(g.dst), T(g.start), K, qnodes=q)["logit"]
        out = forward(m, T(g.src), T(g.dst), T(g.start), K, q)["logit"]
        assert torch.allclose(ref, out, atol=1e-5), (ref - out).abs().max()
        # donor = one branch detached; overriding every cached position <= k and thought k
        att = np.stack([np.nonzero(a)[0] for a in g.attached])[:, 0]
        gd = move_branch(g, att, False)
        don = forward(m, T(gd.src), T(gd.dst), T(gd.start), K, q, record=True)
        k = 2
        # caches recorded at the end include all positions; entries at pos <= START+k are fixed then
        ov = {pos: cache_entries(don["cache"], pos) for pos in range(0, START + k + 1)}
        hk = don["thoughts"][:, k]
        mix = forward(m, T(g.src), T(g.dst), T(g.start), K, q, kv_override=ov,
                      thought_edit={k: lambda h: hk})["logit"]
        assert torch.allclose(mix, don["logit"], atol=1e-4), (mix - don["logit"]).abs().max()
        # zero key bias is a no-op
        zb = {0: torch.zeros(16, 80, device=device), 1: torch.zeros(16, 80, device=device)}
        assert torch.allclose(forward(m, T(g.src), T(g.dst), T(g.start), K, q, key_bias=zb)["logit"], out, atol=1e-6)
    print("  flex.forward == run; full-route override == donor run; zero key bias no-op")


def check_fork_generator():
    from .graphs import set_shapes, SHAPES
    ref_default = sample_graphs(np.random.default_rng(7), 500, split=None)
    set_shapes("fork", 0.5)
    try:
        g = sample_graphs(np.random.default_rng(0), 3000, split=None)
        ref = bfs_dist(g.src, g.dst, g.start)
        assert (ref == g.dist()).all(), "fork: construction labels disagree with BFS"
        assert g.src.shape == (3000, E) and ((ref != ABSENT).sum(1) == 2 + NB * NPB).all()
        kids = np.stack([(g.parent == j).sum(2) for j in range(NPB)], 2)       # (B, NB, NPB) child counts
        assert kids.max() <= 2 and ((kids == 2).sum(2) <= 1).all(), "fork: at most one split per branch"
        fk = (kids == 2).any(2)
        assert (fk.all(1) | ~fk.any(1)).all(), "fork: one shape per graph"
        assert (g.parent == g.parent[:, :1]).all(), "fork: all branches share the template"
        frac = fk[:, 0].mean()
        assert abs(frac - 0.5) < 0.03, frac
        # frontier = 2W on the first layer past the split, = W in chain graphs
        split_at = np.where(fk[:, 0], np.argmax(kids[:, 0] == 2, 1), -1)
        f2 = (ref == split_at[:, None] + 2).sum(1)                              # first layer past the split
        assert (f2[fk[:, 0]] == 2 * g.width[fk[:, 0]]).all() and ((ref == 3).sum(1)[~fk[:, 0]] == g.width[~fk[:, 0]]).all()
        # depth cap (curriculum) still holds and keeps the budget
        gc = sample_graphs(np.random.default_rng(1), 500, split=None, max_bdepth=2)
        assert gc.bdepth.max() <= 2 and (bfs_dist(gc.src, gc.dst, gc.start) == gc.dist()).all()
        # frontier size at distance 3 (2 hops into a branch) varies at fixed W
        f3 = ((ref == 3).sum(1))
        spread = [np.unique(f3[g.width == w]).size for w in range(1, 9)]
        print(f"  fork == BFS on 3000 graphs; fork-graph share {frac:.3f}; frontier 2W past the split; "
              f"distinct distance-3 frontier sizes per W {spread}")
    finally:
        set_shapes("default")
    assert SHAPES["mode"] == "default"
    again = sample_graphs(np.random.default_rng(7), 500, split=None)
    assert (again.src == ref_default.src).all() and (again.family == ref_default.family).all(), "default stream changed"
    print("  default mode unchanged after switching back")


if __name__ == "__main__":
    print("generator"); check_generator()
    print("fork generator"); check_fork_generator()
    print("counterfactuals"); check_counterfactuals()
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("model"); check_model(dev)
    print("flex"); check_flex(dev)
    print("ALL CHECKS PASSED")
