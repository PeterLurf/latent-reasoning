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

import math

import torch
import torch.nn as nn

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


class VariantReasoner(MarkovReasoner):
    """Pilot B options on top of MarkovReasoner (markov_layers=() gives Coconut attention):
      restore : gain-restoring pointer (A2). After each latent step the thought is scored against every node id
                (shared id table), passed through a per-id steep sigmoid with learned slope/threshold (set membership,
                non-competitive) and the restored set is written back: h <- h + W_r(sigmoid(beta*(s - tau)) @ E_id).
      sigread : non-competitive frontier read (A3). Layer-1 attention of latent positions uses independent sigmoid
                weights sigmoid(score + b_head) instead of a softmax (keys never compete for a fixed budget).
    """
    def __init__(self, *a, markov_layers=(), restore=False, sigread=False, **kw):
        super().__init__(*a, markov_layers=markov_layers, **kw)
        self.restore, self.sigread = restore, sigread
        d, d_id, H = self.d, self.id_emb.weight.shape[1], self.blocks[0].attn.h
        if restore:
            self.r_read = nn.Linear(d, d_id)
            self.r_write = nn.Linear(d_id, d, bias=False)
            for m_ in (self.r_read, self.r_write):
                nn.init.normal_(m_.weight, std=0.02)
            nn.init.zeros_(self.r_read.bias)
            self.r_beta = nn.Parameter(torch.tensor(4.0))
            self.r_tau = nn.Parameter(torch.tensor(0.5))
        if sigread:
            self.sig_b = nn.Parameter(torch.full((H,), -4.0))

    def _restore(self, h):
        E = self.id_emb.weight                                   # (V, d_id)
        s = self.r_read(h) @ E.T                                 # (B, V)
        m = torch.sigmoid(self.r_beta * (s - self.r_tau))
        return h + self.r_write(m @ E)

    def _sig_block(self, blk, x, vis, kv):
        at = blk.attn
        q, k, v = at._split(blk.ln1(x))
        Kc = torch.cat([vis[0], k], 2); Vc = torch.cat([vis[1], v], 2)
        sc = q @ Kc.transpose(-1, -2) / math.sqrt(at.dh)
        a = torch.sigmoid(sc + self.sig_b[None, :, None, None])
        x = x + at._merge(a @ Vc)
        x = x + blk.mlp(blk.ln2(x))
        return x, (torch.cat([kv[0], k], 2), torch.cat([kv[1], v], 2))

    def latent_step(self, h, k, cache, keep_resid=False):
        P = self._P
        te = self.type_emb.weight
        base = self.pause.expand_as(h) if self.mode == "pause" else h
        x = (base + te[T_LAT] + (self.step_emb.weight[k] if self.use_step else 0))[:, None]
        new = []
        for li, (blk, kv) in enumerate(zip(self.blocks, cache)):
            vis = (kv[0][:, :, :P], kv[1][:, :, :P]) if li in self.markov_layers else kv
            if li == 0 and self.sigread:
                x, kvn = self._sig_block(blk, x, vis, kv)
                new.append(kvn)
                continue
            x_new, kv2 = blk.step(x, vis)
            new.append((torch.cat([kv[0], kv2[0][:, :, -1:]], 2), torch.cat([kv[1], kv2[1][:, :, -1:]], 2)))
            x = x_new
        h2 = self._thought(x[:, 0])
        return (self._restore(h2) if self.restore else h2), new, None


def run_both_anytime(model, src, dst, start, K, qnodes, target, pnodes=None, cnodes=None):
    """As run_both, but returns delayed and upfront logits after every latent step: (K, B, Q) and (K, B).
    Works for LatentReasoner and the Markov/Variant subclasses (sets the prefix length they need)."""
    from latent.model import T_TARGET as _TT
    h, cache, _ = model.encode(model.embed_prefix(src, dst, start))
    P0 = cache[0][0].shape[2]
    te = model.type_emb.weight
    h_u, cache_u = model.extend(model.node_emb(target) + te[_TT], cache)
    pl = model.parent_logits(cache, pnodes) if pnodes is not None else None
    cl = model.children_logits(cache, cnodes) if cnodes is not None else None
    outs = []
    for hh, cc, qq, P in ((h, cache, qnodes, P0), (h_u, cache_u, None, P0 + 1)):
        if hasattr(model, "_P"):
            model._P = P
        per = []
        for k in range(1, K + 1):
            hh, cc, _ = model.latent_step(hh, k, cc)
            per.append(model.query(cc, qq, B=src.shape[0]))
        outs.append(torch.stack(per, 0))
    return outs[0], outs[1][..., 0], pl, cl



def add_jitter(model, r, p=0.5):
    """Proposal T6 (THEORY §22): amplitude jitter. While training, the thought fed into every latent step is rescaled
    about the batch mean, h <- c + s (h - c), with s ~ LogUniform[1/r, r] per example, applied with probability p
    (the exp9 transfer-curve perturbation used as augmentation). Evaluation (model.eval()) is untouched and the
    state dict is unchanged, so every analysis tool loads these checkpoints as they are."""
    base, logr = model.latent_step, math.log(r)

    def latent_step(h, k, cache, keep_resid=False):
        if model.training:
            B = h.shape[0]
            s = torch.exp((2 * torch.rand(B, 1, device=h.device) - 1) * logr)
            s = torch.where(torch.rand(B, 1, device=h.device) < p, s, torch.ones_like(s)).to(h.dtype)
            c = h.mean(0, keepdim=True).detach()
            h = c + s * (h - c)
        return base(h, k, cache, keep_resid)
    model.latent_step = latent_step
    model.jitter = (r, p)
    return model


def build(kind, **kw):
    if kind == "coconut":
        return LatentReasoner(**kw)
    if kind == "markov":
        return MarkovReasoner(markov_layers=(0, 1), **kw)
    if kind == "markov_l2":
        return MarkovReasoner(markov_layers=(1,), **kw)
    # Pilot B variants
    opts = {"restore": dict(restore=True), "markov_restore": dict(markov_layers=(0, 1), restore=True),
            "markov_sigread": dict(markov_layers=(0, 1), sigread=True),
            "markov_restore_sigread": dict(markov_layers=(0, 1), restore=True, sigread=True)}
    if kind in opts:
        return VariantReasoner(**opts[kind], **kw)
    raise ValueError(kind)
