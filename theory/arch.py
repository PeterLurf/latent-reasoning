"""Theory stream: architecture variants of the latent reasoner (subclasses; latent/ is not modified).

  markov     : every layer of the latent position attends to the prefix (edges, START[, TARGET]) and itself only —
               never to earlier latent positions. Queries still read every position's keys/values. Removes the
               retrieval route and the implicit step clock (docs/THEORY.md §2) by construction.
  markov_l2  : only layer 2 is Markovian.
State-dict format is the LatentReasoner's, so the analysis tools load these checkpoints unchanged (the rollout
harness reproduces a Markov model with block_from=1, block_layers=markov_layers).
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from latent.model import T_LAT, T_TARGET, LatentReasoner  # noqa: E402


class MarkovReasoner(LatentReasoner):
    def __init__(self, *a, markov_layers=(0, 1), **kw):
        super().__init__(*a, **kw)
        self.markov_layers = tuple(markov_layers)
        self._P = None

    # the prefix length is recorded whenever a prefix cache is built
    def encode(self, x, keep_resid=False):
        h, cache, resid = super().encode(x, keep_resid)
        self._P = cache[0][0].shape[2]
        return h, cache, resid

    def extend(self, x, cache):
        h, new = super().extend(x, cache)
        self._P = new[0][0].shape[2]
        return h, new

    def latent_step(self, h, k, cache, keep_resid=False):
        P = self._P
        te = self.type_emb.weight
        base = self.pause.expand_as(h) if self.mode == "pause" else h
        x = (base + te[T_LAT] + (self.step_emb.weight[k] if self.use_step else 0))[:, None]
        new = []
        for li, (blk, kv) in enumerate(zip(self.blocks, cache)):
            vis = (kv[0][:, :, :P], kv[1][:, :, :P]) if li in self.markov_layers else kv
            x_new, kv2 = blk.step(x, vis)
            new.append((torch.cat([kv[0], kv2[0][:, :, -1:]], 2), torch.cat([kv[1], kv2[1][:, :, -1:]], 2)))
            x = x_new
        return self._thought(x[:, 0]), new, None

    def run_both(self, src, dst, start, K, qnodes, target, pnodes=None, cnodes=None):
        h, cache, _ = self.encode(self.embed_prefix(src, dst, start))
        P0 = self._P
        te = self.type_emb.weight
        h_u, cache_u = self.extend(self.node_emb(target) + te[T_TARGET], cache)
        pl = self.parent_logits(cache, pnodes) if pnodes is not None else None
        cl = self.children_logits(cache, cnodes) if cnodes is not None else None
        outs = []
        for hh, cc, qq, P in ((h, cache, qnodes, P0), (h_u, cache_u, None, P0 + 1)):
            self._P = P
            for k in range(1, K + 1):
                hh, cc, _ = self.latent_step(hh, k, cc)
            outs.append(self.query(cc, qq, B=src.shape[0]))
        return outs[0], outs[1][:, 0], pl, cl


def build(kind, **kw):
    if kind == "coconut":
        return LatentReasoner(**kw)
    if kind == "markov":
        return MarkovReasoner(markov_layers=(0, 1), **kw)
    if kind == "markov_l2":
        return MarkovReasoner(markov_layers=(1,), **kw)
    raise ValueError(kind)
