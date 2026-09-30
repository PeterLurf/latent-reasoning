"""THEORY §30 (PROPOSALS §9 F3a, F6a, F6b): training-free / readout-only repairs on the 59 P30 ∪ L models (CPU).

F3a renorm   : past K, every fed-back thought's deviation from the batch mean is rescaled to its example's mean in-range
               norm (thoughts h_1..h_{K-1}); with and without the layer-2 block from K (exp8 used a constant alpha).
F6a dec_set  : a frontier decoder (linear head h_K -> scores over node ids through the id table, trained per frozen model
               on training graphs, labels = all nodes at the BFS depth of the model's own step-K anchor) decodes the
               frontier set; episode 2 is run from each decoded node (p > 0.5, at most 6; top-1 if none) and max-combined
               with episode 1. dec_top1: episode 2 from the decoder's top node only.
F6b dec_soft : one episode 2 whose START embedding is the decoder-probability-weighted mean of the decoded nodes'
               embeddings (a superposed restart).
Metrics on the long-chain probe(512), as exp12: bal-acc by distance 1..12; decoder recall of the probed chain's node
at the anchor depth.
    python theory/exp13_decode_restart.py --out theory/results/exp13_decode_restart.json
"""
import argparse
import json
import os
import sys
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.set_num_threads(int(os.environ.get("THEORY_THREADS", "2")))
from exp12_reanchor import bal_by_dist  # noqa: E402
from latent.graphs import V, sample_graphs  # noqa: E402
from latent.model import T_START  # noqa: E402
from rollout import load, probe, rollout  # noqa: E402


def bfs_depth(src, dst, s):
    adj = {}
    for a, b in zip(src, dst):
        adj.setdefault(int(a), []).append(int(b))
    d = {int(s): 0}; q = deque([int(s)])
    while q:
        u = q.popleft()
        for v in adj.get(u, []):
            if v not in d:
                d[v] = d[u] + 1; q.append(v)
    return d


@torch.no_grad()
def anchor_and_thought(m, K, g, head):
    T = lambda a: torch.as_tensor(a)
    o = rollout(m, K, T(g.src), T(g.dst), T(g.start), K)
    a = o["attn1"][K - 1][:, head, :o["P"] - 1]
    anc = g.dst[np.arange(len(g)), a.argmax(1).numpy()]
    return anc, o["thoughts"][:, K]


def train_decoder(m, K, head, steps=250, bs=256, seed=0):
    rng = np.random.default_rng(seed)
    E = m.id_emb.weight.detach()
    W = torch.nn.Linear(m.d, E.shape[1]); b = torch.nn.Parameter(torch.zeros(()))
    opt = torch.optim.Adam(list(W.parameters()) + [b], lr=1e-2)
    for _ in range(steps):
        g = sample_graphs(rng, bs, split="train", max_bdepth=None)
        anc, h = anchor_and_thought(m, K, g, head)
        y = torch.zeros(bs, V); keep = np.zeros(bs, bool)
        for i in range(bs):
            D = bfs_depth(g.src[i], g.dst[i], g.start[i]); da = D.get(int(anc[i]))
            if da is None or da == 0:
                continue
            keep[i] = True
            for v, dv in D.items():
                if dv == da:
                    y[i, v] = 1.0
        if not keep.any():
            continue
        s = W(h) @ E.T + b
        loss = F.binary_cross_entropy_with_logits(s[keep], y[keep])
        opt.zero_grad(); loss.backward(); opt.step()
    return lambda h: torch.sigmoid(W(h) @ E.T + b)


@torch.no_grad()
def episode_from_embedding(m, K, src, dst, start_ids, start_vec, q):
    x = m.embed_prefix(src, dst, start_ids)
    if start_vec is not None:
        x = x.clone(); x[:, -1] = start_vec + m.type_emb.weight[T_START]
    h, cache, _ = m.encode(x)
    for k in range(1, K + 1):
        h, cache, _ = m.latent_step(h, k, cache)
    return m.query(cache, q)


def renorm_feed(K):
    def feed(k, h, hist):
        if k >= K + 1:
            devs = [(t - t.mean(0, keepdim=True)).norm(dim=1) for t in hist[1:K]]         # thoughts h_1..h_{K-1}
            target = torch.stack(devs, 0).mean(0)
            c = h.mean(0, keepdim=True)
            d = h - c
            return c + d * (target / d.norm(dim=1).clamp_min(1e-6))[:, None]
        return h
    return feed


@torch.no_grad()
def measure(m, K, head, dec, g, br):
    T = lambda a: torch.as_tensor(a)
    n = len(g); bi = np.arange(n)
    q = np.concatenate([g.nodes[bi, br[c]] for c in ("A", "B", "C", "D")], 1)
    chain = np.concatenate([g.nodes[bi, br["A"]], g.nodes[bi, br["B"]]], 1)
    src, dst, Q = T(g.src), T(g.dst), T(q)
    out = {}
    for tag, kw in (("renorm", {}), ("renorm_block", {"block_from": K, "block_layers": (1,)})):
        o = rollout(m, K, src, dst, T(g.start), 12, qnodes=Q, feed=renorm_feed(K), **kw)
        out[tag] = bal_by_dist(o["qlogit"].numpy(), n)
    o1 = rollout(m, K, src, dst, T(g.start), K, qnodes=Q)
    q1 = o1["qlogit"].numpy()
    a = o1["attn1"][K - 1][:, head, :o1["P"] - 1]
    anc = g.dst[bi, a.argmax(1).numpy()]
    p = dec(o1["thoughts"][:, K])                                                   # (n, V)
    D = [bfs_depth(g.src[i], g.dst[i], g.start[i]) for i in range(n)]
    sets = []
    for i in range(n):
        s = [v for v in torch.argsort(p[i], descending=True)[:6].tolist() if p[i, v] > 0.5] or [int(p[i].argmax())]
        sets.append(s)
    # recall of the probed chain's node at the anchor's depth
    hit = []
    for i in range(n):
        da = D[i].get(int(anc[i]))
        if da is not None and 1 <= da <= 12:
            hit.append(int(chain[i, da - 1]) in sets[i])
    out["decoder_chain_recall"] = float(np.mean(hit)) if hit else None
    out["decoded_set_size"] = float(np.mean([len(s) for s in sets]))
    out["attention_anchor_on_chain"] = float(np.mean([(chain[i] == anc[i]).any() for i in range(n)]))
    qset = q1.copy()
    for r in range(6):
        ids = np.array([s[r] if r < len(s) else s[0] for s in sets])
        qset = np.maximum(qset, episode_from_embedding(m, K, src, dst, T(ids), None, Q).numpy())
    out["dec_set"] = bal_by_dist(qset, n)
    top1 = np.array([s[0] for s in sets])
    out["dec_top1"] = bal_by_dist(np.maximum(q1, episode_from_embedding(m, K, src, dst, T(top1), None, Q).numpy()), n)
    E = m.node_emb.weight                                                              # (V, d)
    w = torch.zeros(n, V)
    for i, s in enumerate(sets):
        w[i, s] = p[i, s]
    w = w / w.sum(1, keepdim=True).clamp_min(1e-9)
    out["dec_soft"] = bal_by_dist(np.maximum(q1, episode_from_embedding(m, K, src, dst, T(top1), w @ E, Q).numpy()), n)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="theory/results/exp13_decode_restart.json")
    ap.add_argument("--runs", nargs="*")
    a = ap.parse_args()
    R = json.load(open("theory/results/exp12_reanchor.json"))
    names = a.runs or sorted(R, key=lambda r: (R[r]["extrap"], r))
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    g, br, _ = probe(512)
    for name in names:
        if name in res:
            continue
        path = name if os.path.isdir(name) else "runs/" + name
        m, c = load(path, "cpu")
        head = R[name]["head"] if name in R else None
        if head is None:
            from analyse_pilotA import ptr_traj
            from rollout import probe as _p
            gg, bb, EE = _p(512)
            with torch.no_grad():
                _, _, head = ptr_traj(m, c["K"], gg, bb, EE, {})
        dec = train_decoder(m, c["K"], head)
        o = measure(m, c["K"], head, dec, g, br)
        o.update(K=c["K"], extrap=R[name]["extrap"] if name in R else None)
        res[os.path.basename(name.rstrip("/\\"))] = o
        json.dump(res, open(a.out, "w", newline="\n"), indent=1)
        f = lambda v: " ".join("%.2f" % x for x in v[6:10])
        print("%-10s %s | renorm %s | renorm+blk %s | dec_set %s | top1 %s | soft %s | recall %.2f size %.1f" % (
            name, "EXT" if o["extrap"] else "   ", f(o["renorm"]), f(o["renorm_block"]), f(o["dec_set"]), f(o["dec_top1"]),
            f(o["dec_soft"]), o["decoder_chain_recall"] or -1, o["decoded_set_size"]), flush=True)


if __name__ == "__main__":
    main()
