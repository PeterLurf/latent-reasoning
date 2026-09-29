"""PROTOCOL7 T7: independent re-implementation of P7 (code audit). Written 2026-09-28 ~23:57 from the definition, without
reusing scratch/far_sweep*.py or latent/frontier_amp.py:

  * probe graphs: latent.length_gen.long_chain_graphs (seed 2033, 1024 graphs, widths 2/4/6);
  * each chain edge is located by matching the id of the node it leads into against the edge list (dst ids);
  * first-block attention of each latent step is recomputed from that block's own qkv weights: the step's layer-norm
    input, its query, and the keys of every earlier position (edges, START, earlier latent steps) plus itself;
  * excess = mean over graphs of (attention on reachable edge) - (attention on its look-alike), heads averaged;
    P7 = max over 12 steps of the distance-7 excess / max over steps 1..K and distances 1..6 of the in-range excess.
Pass (PROTOCOL7 T7): Spearman(P7_indep, P7_frozen) >= 0.99 and max |difference| <= 0.02 over P30 and L competent runs.
    PYTHONPATH=. python scratch/p7_indep.py --device cuda:0
"""
import argparse
import json
import math
import os

import numpy as np
import torch

from latent.competence import load
from latent.length_gen import long_chain_graphs
from latent.model import T_LAT

OUT = "results/p7_indep.json"


def edges_into(g, chain_nodes):
    """chain_nodes: (N, 6) node ids -> (N, 6) index of the edge whose dst is that node."""
    N = len(g)
    out = np.full(chain_nodes.shape, -1)
    for j in range(chain_nodes.shape[1]):
        hit = g.dst == chain_nodes[:, j][:, None]                  # (N, n_edges)
        assert (hit.sum(1) == 1).all(), "each node must have exactly one incoming edge"
        out[:, j] = hit.argmax(1)
    return out


@torch.no_grad()
def first_block_attention(m, K, g, Kp, device):
    """(Kp, N, n_edges) attention of each latent step onto the edge tokens in the first block, heads averaged."""
    T = lambda a: torch.as_tensor(a, device=device)
    blk = m.blocks[0]
    H = blk.attn.h
    dh = m.d // H
    Wqkv, bqkv = blk.attn.qkv.weight, blk.attn.qkv.bias
    n_edges = g.src.shape[1]
    out = []
    for i in range(0, len(g), 256):
        s = slice(i, min(len(g), i + 256))
        x0 = m.embed_prefix(T(g.src[s]), T(g.dst[s]), T(g.start[s]))            # prefix embeddings
        keys = (blk.ln1(x0) @ Wqkv.T + bqkv)[..., m.d:2 * m.d]                 # (B, T0, d) keys of the prefix
        h, cache, _ = m.encode(x0)
        use_step = getattr(m, "use_step", True)
        rows = []
        for k in range(1, Kp + 1):
            base = m.pause.expand_as(h) if m.mode == "pause" else h
            xin = base + m.type_emb.weight[T_LAT] + (m.step_emb.weight[min(k, K)] if use_step else 0)   # (B, d)
            qkv = blk.ln1(xin) @ Wqkv.T + bqkv
            q, kk = qkv[:, :m.d], qkv[:, m.d:2 * m.d]
            keys = torch.cat([keys, kk[:, None]], 1)                               # this step attends to itself too
            qh = q.view(-1, H, dh)
            kh = keys.view(keys.shape[0], keys.shape[1], H, dh)
            logits = torch.einsum("bhd,bthd->bht", qh, kh) / math.sqrt(dh)
            att = torch.softmax(logits, -1).mean(1)                                 # (B, T)
            rows.append(att[:, :n_edges].cpu().numpy())
            h, cache, _ = m.latent_step(h, min(k, K), cache)                        # advance with the model's own step
        out.append(np.stack(rows))
    return np.concatenate(out, 1)


def p7(run, device, seed=2033, n=1024, Kp=12):
    m, c = load(run, device)
    K = c["K"]
    g, br = long_chain_graphs(np.random.default_rng(seed), n, [2, 4, 6])
    bi = np.arange(n)
    idx = {ch: edges_into(g, g.nodes[bi, br[ch]]) for ch in "ABCD"}
    A = first_block_attention(m, K, g, Kp, device)                                 # (Kp, n, n_edges)
    at = lambda ch, j: A[:, bi, idx[ch][:, j]].mean(1)                             # (Kp,)
    near = np.stack([at("A", j) - at("D", j) for j in range(6)], 1)                # (Kp, 6)
    e7 = float((at("B", 0) - at("C", 0)).max())
    in_peak = float(near[:K].max())
    return {"P7": e7 / in_peak, "in_peak": in_peak, "e7": e7}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    frozen = {}
    for f in ("results/p30.json", "results/p_l.json"):
        d = json.load(open(f))
        for r in d["primary"]:
            frozen[f"runs/{r}"] = d["rows"][r]["P7"]
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for r in frozen:
        if r not in res:
            res[r] = p7(r, a.device)
            json.dump(res, open(OUT, "w"), indent=1)
    x = np.array([frozen[r] for r in frozen]); y = np.array([res[r]["P7"] for r in frozen])
    rank = lambda v: np.argsort(np.argsort(v, kind="stable"), kind="stable").astype(float)
    rho, dmax = float(np.corrcoef(rank(x), rank(y))[0, 1]), float(np.abs(x - y).max())
    ok = rho >= 0.99 and dmax <= 0.02
    summ = {"n": len(x), "spearman": rho, "max_abs_diff": dmax, "T7": "pass" if ok else "FAIL"}
    json.dump({"rows": res, "summary": summ}, open(OUT, "w"), indent=1)
    print(summ)


if __name__ == "__main__":
    main()
