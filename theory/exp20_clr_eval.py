"""THEORY §45: evaluate checkpointed students (theory/train_clr.py). CPU, measurement only.

For each model (its W and sweep head from arch.json), with checkpoints written by the model itself (the W targets of
the edges its sweep head attends most at a segment's last step; top-1 last):
  * two_seg  : episode 1 from START + one restart from the written checkpoint; long-chain probe(512), bal-acc by
               distance (answers max-combined over segments)
  * chain    : S segments of K steps on §37's 24-hop probe (seed 7), each restarted from the checkpoint written by the
               previous one; bal-acc by distance 1..24, for S = 2, 3, 4
  * edit     : C7 on checkpoints (probe(512)): in episode 2's written checkpoint, replace every node that lies on the
               reachable chain by its look-alike (same index on the unreachable chain); IIA = share of graphs where the
               look-alike's successors at +2, +3 are answered reachable and the reachable ones at +2, +3 are not
               (answers of episode 2 alone, not max-combined)
    python theory/exp20_clr_eval.py theory/runs/pc3_s111 ... --out theory/results/exp20_clr_eval.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.set_num_threads(int(os.environ.get("THEORY_THREADS", "2")))
from deep_chain import deep_chain_graphs  # noqa: E402
from eval_arch import load_arch  # noqa: E402
from exp12_reanchor import bal_by_dist  # noqa: E402
from exp19_set_restart import set_episode  # noqa: E402
from rollout import probe, rollout  # noqa: E402

T = lambda a: torch.as_tensor(a)


def segment(m, K, g, starts, head, W, Q):
    """Answers (n, Q) of one segment from START set `starts`, and the checkpoint it writes (n, W)."""
    n = len(g); bi = np.arange(n)
    if starts.shape[1] == 1:
        o = rollout(m, K, T(g.src), T(g.dst), T(starts[:, 0]), K, qnodes=Q)
        ans = o["qlogit"].numpy()
        att = o["attn1"][K - 1][:, head, :o["P"] - 1]
    else:
        ans = set_episode(m, K, T(g.src), T(g.dst), starts, Q).numpy()
        # checkpoint of a set-started segment: run the same set episode through a manual loop that records attention
        att = set_attention(m, K, g, starts, head)
    order = torch.argsort(att, 1, descending=True)[:, :W].numpy()
    ck = np.stack([g.dst[bi, order[:, w]] for w in range(W)][::-1], 1)
    return ans, ck


@torch.no_grad()
def set_attention(m, K, g, starts, head):
    """Sweep-head layer-1 attention over edge tokens at the last latent step of a set-started episode."""
    from latent.model import T_LAT, T_START
    import math
    x = m.embed_prefix(T(g.src), T(g.dst), T(starts[:, -1]))
    te = m.type_emb.weight
    extra = [(m.node_emb(T(starts[:, w])) + te[T_START])[:, None] for w in range(starts.shape[1] - 1)]
    x = torch.cat([x[:, :-1]] + extra + [x[:, -1:]], 1)
    E = g.src.shape[1]
    h, cache, _ = m.encode(x)
    for k in range(1, K + 1):
        if k == K:
            blk = m.blocks[0]
            base = h
            xx = (base + te[T_LAT] + (m.step_emb.weight[k] if getattr(m, "use_step", True) else 0))[:, None]
            qh, kh, vh = blk.attn._split(blk.ln1(xx))
            Kc = torch.cat([cache[0][0], kh], 2)
            s = qh @ Kc.transpose(-1, -2) / math.sqrt(blk.attn.dh)
            a = torch.softmax(s, -1)[:, head, 0, :E]
        h, cache, _ = m.latent_step(h, k, cache)
    return a


@torch.no_grad()
def anchor_maxshare(m, K, g, starts):
    """§45 amendment CLR-1a (manipulation check): at the first latent step of a set-started episode, attention mass on
    each of the W START tokens summed over all layers and heads; per graph, the largest anchor's share of the W-token
    total; returns the mean over graphs (1/W = even use, 1 = collapse onto one anchor)."""
    from latent.model import T_START
    x = m.embed_prefix(T(g.src), T(g.dst), T(starts[:, -1]))
    te = m.type_emb.weight
    extra = [(m.node_emb(T(starts[:, w])) + te[T_START])[:, None] for w in range(starts.shape[1] - 1)]
    x = torch.cat([x[:, :-1]] + extra + [x[:, -1:]], 1)
    P = x.shape[1]; W = starts.shape[1]
    h, cache, _ = m.encode(x)
    for blk in m.blocks:
        blk.attn.record = []
    m.latent_step(h, 1, cache)
    mass = sum(blk.attn.record[0][:, :, P - W:P].sum(1) for blk in m.blocks)     # (n, W)
    for blk in m.blocks:
        blk.attn.record = None
    sh = mass / mass.sum(1, keepdim=True).clamp_min(1e-12)
    return float(sh.max(1).values.mean())


@torch.no_grad()
def run(m, K, head, W):
    out = {}
    g, br, _ = probe(512)
    n = len(g); bi = np.arange(n)
    q = np.concatenate([g.nodes[bi, br[c]] for c in ("A", "B", "C", "D")], 1)
    Q = T(q)
    a1, ck = segment(m, K, g, g.start[:, None], head, W, Q)
    a2, _ = segment(m, K, g, ck, head, W, Q)
    out["two_seg"] = bal_by_dist(np.maximum(a1, a2), n)
    if W > 1:
        out["anchor_maxshare"] = anchor_maxshare(m, K, g, ck)
        rng = np.random.default_rng(0); ckr = ck.copy()
        for i in range(n):                                      # non-seed anchors -> random off-chain nodes
            pool = [v for v in set(g.src[i].tolist()) | set(g.dst[i].tolist()) if v not in set(q[i].tolist())]
            ckr[i, :-1] = rng.choice(pool, W - 1, replace=len(pool) < W - 1)
        a2r, _ = segment(m, K, g, ckr, head, W, Q)
        out["two_seg_rand"] = bal_by_dist(np.maximum(a1, a2r), n)
    out["one_seg"] = bal_by_dist(a1, n)
    # C7 edit: swap written nodes that are on the reachable chain for their look-alikes
    reach = np.concatenate([g.nodes[bi, br["A"]], g.nodes[bi, br["B"]]], 1)
    decoy = np.concatenate([g.nodes[bi, br["D"]], g.nodes[bi, br["C"]]], 1)
    ed = ck.copy(); jmax = np.full(n, -1)
    for i in range(n):
        for w in range(ck.shape[1]):
            hit = np.where(reach[i] == ck[i, w])[0]
            if len(hit):
                ed[i, w] = decoy[i, hit[0]]; jmax[i] = max(jmax[i], hit[0])
    el = (jmax >= 0) & (jmax <= 8)
    a2e, _ = segment(m, K, g, ed, head, W, Q)
    p0, p1 = a2 > 0, a2e > 0
    # columns: A 0-5, B 6-11 (reach index), C 12-17 (decoy index 6-11), D 18-23 (decoy index 0-5)
    rcol = lambda j: j
    dcol = lambda j: np.where(j < 6, 18 + j, 12 + j - 6)
    jj = np.where(el, jmax, 0)
    ins = p1[bi, dcol(jj + 2)] & p1[bi, dcol(jj + 3)]
    dele = ~p1[bi, rcol(jj + 2)] & ~p1[bi, rcol(jj + 3)]
    out["edit"] = {"eligible": int(el.sum()), "IIA": float((ins & dele)[el].mean()) if el.any() else None,
                   "ins": float(np.mean([p1[bi, dcol(jj + o)][el].mean() - p0[bi, dcol(jj + o)][el].mean() for o in (2, 3)])) if el.any() else None,
                   "del": float(np.mean([p1[bi, rcol(jj + o)][el].mean() - p0[bi, rcol(jj + o)][el].mean() for o in (2, 3)])) if el.any() else None}
    # chaining on the 24-hop probe
    gd, sd = deep_chain_graphs(np.random.default_rng(7), 512, [2, 3], 4)
    Qd = T(np.concatenate([sd["reach"], sd["decoy"]], 1))
    L = sd["reach"].shape[1]
    starts = gd.start[:, None]; best = None
    for S in range(1, 5):
        ans, starts = segment(m, K, gd, starts, head, W, Qd)
        best = ans if best is None else np.maximum(best, ans)
        if S >= 2:
            p = best > 0
            out[f"chain{S}"] = [float(0.5 * (p[:, j].mean() + 1 - p[:, L + j].mean())) for j in range(L)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="theory/results/exp20_clr_eval.json")
    a = ap.parse_args()
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for path in a.runs:
        name = os.path.basename(path.rstrip("/\\"))
        if name in res or not os.path.exists(os.path.join(path, "model.pt")):
            continue
        m, c = load_arch(path, "cpu")
        A = json.load(open(os.path.join(path, "arch.json")))
        cl = A.get("clr", {})
        W, head = cl.get("W", 1), cl.get("head")
        o = run(m, c["K"], head, W)
        o.update(W=W, head=head)
        res[name] = o
        json.dump(res, open(a.out, "w", newline="\n"), indent=1)
        f = lambda v: "%.3f" % np.mean(v)
        print("%-12s W %d | two_seg far8-10 %s | chain4 d13-18 %s | edit IIA %s" % (
            name, W, f(o["two_seg"][7:10]), f(o["chain4"][12:18]), o["edit"]["IIA"]), flush=True)


if __name__ == "__main__":
    main()
