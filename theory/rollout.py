"""Theory stream (exploratory): instrumented latent rollout with interventions on the recurrence.

Mirrors LatentReasoner.encode / latent_step / query exactly (checked by --selftest), but lets each latent
step see a *modified* context:
  * window w      : latent step k attends to the prefix + only the last w latent positions (older latent KV
                    hidden from the recurrence); queries still read the full cache.  w=None -> normal.
  * window_from   : apply the window only from this step on (e.g. K+1 = only untrained steps).
  * feed          : optional fn(k, h, hist) -> h fed into step k (hist = list of earlier thoughts).
Records per step: layer-1 attention of the latent position over all keys (edges, START, latent positions).

Long-chain probe graphs from latent/length_gen.py (chain A attached, B hung from A's leaf, decoys C/D).
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from latent.competence import load  # noqa: E402
from latent.graphs import NPB  # noqa: E402
from latent.length_gen import long_chain_graphs  # noqa: E402
from latent.model import T_LAT  # noqa: E402

torch.set_num_threads(int(os.environ.get("THEORY_THREADS", "4")))


def _step_attn(attn, x, kv, keep_idx=None):
    """One latent position through an Attn module, attending to cache[keep_idx] + itself.
    Returns output, new full cache, attention row (B,H,Tvisible+1)."""
    q, k, v = attn._split(x)
    Kc, Vc = kv
    if keep_idx is not None:
        Kc, Vc = Kc[:, :, keep_idx], Vc[:, :, keep_idx]
    Kall = torch.cat([Kc, k], 2)
    Vall = torch.cat([Vc, v], 2)
    s = q @ Kall.transpose(-1, -2) / math.sqrt(attn.dh)
    a = torch.softmax(s, -1)
    y = attn._merge(a @ Vall)
    return y, (torch.cat([kv[0], k], 2), torch.cat([kv[1], v], 2)), a[:, :, 0], s[:, :, 0]


@torch.no_grad()
def rollout(m, K, src, dst, start, Kp, window=None, window_from=1, feed=None, qnodes=None,
            block_from=None, block_layers=(1,)):
    """Returns dict with thoughts (B,Kp+1,d), attn1 list of (B,H,T_k) [full-width, hidden keys = 0],
    logits1 list, and query logits (B,Q) if qnodes given."""
    x = m.embed_prefix(src, dst, start)
    P = x.shape[1]                                   # prefix length (edges + START)
    h, cache, _ = m.encode(x)
    thoughts, attn1, logit1 = [h], [], []
    te = m.type_emb.weight
    for k in range(1, Kp + 1):
        if feed is not None:
            h = feed(k, h, thoughts)
        kk = min(k, K)
        base = m.pause.expand_as(h) if m.mode == "pause" else h
        xx = (base + te[T_LAT] + (m.step_emb.weight[kk] if getattr(m, "use_step", True) else 0))[:, None]
        nlat = k - 1                                  # latent positions already in the cache
        keep = None
        if window is not None and k >= window_from and nlat > window:
            keep = torch.as_tensor(list(range(P)) + list(range(P + nlat - window, P + nlat)), device=src.device)
        new = []
        for li, (blk, kv) in enumerate(zip(m.blocks, cache)):
            kp = keep
            if block_from is not None and k >= block_from and li in block_layers and nlat > 0:
                base_idx = keep if keep is not None else torch.arange(P + nlat, device=src.device)
                kp = base_idx[base_idx < P]                   # prefix only: no earlier latent positions
            a_out, kv2, arow, srow = _step_attn(blk.attn, blk.ln1(xx), kv, kp)
            xx = xx + a_out
            xx = xx + blk.mlp(blk.ln2(xx))
            new.append(kv2)
            if li == 0:
                full = torch.zeros(arow.shape[0], arow.shape[1], P + nlat + 1, device=arow.device)
                fs = torch.full_like(full, float("nan"))
                idx = kp if kp is not None else torch.arange(P + nlat, device=src.device)
                full[:, :, idx] = arow[:, :, :-1]; full[:, :, -1] = arow[:, :, -1]
                fs[:, :, idx] = srow[:, :, :-1]; fs[:, :, -1] = srow[:, :, -1]
                attn1.append(full); logit1.append(fs)
        cache = new
        h = m._thought(xx[:, 0])
        thoughts.append(h)
    out = {"thoughts": torch.stack(thoughts, 1), "attn1": attn1, "logit1": logit1, "P": P}
    if qnodes is not None:
        out["qlogit"] = m.query(cache, qnodes)
    return out


def chain_edges(g, br):
    """Edge-token index of each chain's edge j (edge into the chain's node j) -> dict ch -> (N, NPB)."""
    n = len(g)
    bi = np.arange(n)
    E = {}
    for ch in ("A", "B", "C", "D"):
        cols = []
        for j in range(NPB):
            cols.append(g.hub_pos[bi, br[ch]] if j == 0 else np.argmax(g.dst == g.nodes[bi, br[ch], j][:, None], 1))
        E[ch] = np.stack(cols, 1)
    return E


def probe(n=512, seed=4247, widths=(2, 4, 6)):
    g, br = long_chain_graphs(np.random.default_rng(seed), n, list(widths))
    return g, br, chain_edges(g, br)


def front_profile(attn1, g, br, E):
    """Per step: excess attention (heads averaged) on the 12 chain-AB edges minus matched C/D edges, per graph.
    Returns (Kp, N, 12) array: edge e -> distance e+1."""
    n = len(g)
    bi = np.arange(n)[:, None]
    rows = []
    for a in attn1:
        am = a.mean(1).cpu().numpy()
        ab = np.concatenate([am[bi, E["A"]] - am[bi, E["D"]], am[bi, E["B"]] - am[bi, E["C"]]], 1)
        rows.append(ab)
    return np.stack(rows)


def query_eval(m, K, g, br, Kp, device, **kw):
    """TPR on A/B nodes and FPR on C/D nodes by chain position at budget Kp (with the rollout's interventions)."""
    n = len(g)
    bi = np.arange(n)
    q = np.concatenate([g.nodes[bi, br[c]] for c in ("A", "B", "C", "D")], 1)
    T = lambda a: torch.as_tensor(a, device=device)
    out = rollout(m, K, T(g.src), T(g.dst), T(g.start), Kp, qnodes=T(q), **kw)
    p = (out["qlogit"] > 0).cpu().numpy().reshape(n, 4, NPB)
    return {"A": p[:, 0].mean(0), "B": p[:, 1].mean(0), "C": p[:, 2].mean(0), "D": p[:, 3].mean(0)}, out


def selftest(run, device="cpu"):
    m, c = load(run, device)
    g, br, E = probe(128)
    T = lambda a: torch.as_tensor(a, device=device)
    bi = np.arange(len(g))
    q = T(np.concatenate([g.nodes[bi, br[ch]] for ch in ("A", "B", "C", "D")], 1))
    ref = m.run(T(g.src), T(g.dst), T(g.start), c["K"], qnodes=q)["logit"]
    mine = rollout(m, c["K"], T(g.src), T(g.dst), T(g.start), c["K"], qnodes=q)["qlogit"]
    print("selftest max|diff|", float((ref - mine).abs().max()))


if __name__ == "__main__":
    selftest(sys.argv[1] if len(sys.argv) > 1 else "runs/p30_s105")
