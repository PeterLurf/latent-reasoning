"""Flexible forward pass for the follow-up interventions.

Positions: edges 0..E-1, START = E, LAT_j = E + j. Equivalent to LatentReasoner.run
(START is appended as a cached step, which tests.check_flex verifies), plus:

  thought_edit : {k: fn(h) -> h}   edit thought k before it feeds LAT_{k+1} (any set of k)
  kv_override  : {pos: [(k, v) per layer]}  after position `pos` is cached, overwrite its
                 cached keys/values (each (B, H, dh)) — what LATER positions and queries read.
                 Edge positions are overwritten right after the edge prefix is encoded.
  key_bias     : {layer: (B, T_max) tensor}  additive attention-logit bias on keys, applied to
                 the START row and every latent row (never to the edge prefix or to queries).
  record       : collect thoughts (B, K+1, d) and per-position caches.
"""
from __future__ import annotations

import math

import torch

from .graphs import E
from .model import T_EDGE, T_LAT, T_START

START = E


def _step(blk, x, kv, bias=None):
    """Block.step with an optional additive key bias (B, T) for this single query row."""
    attn = blk.attn
    h = blk.ln1(x)
    q, k, v = attn._split(h)
    K = torch.cat([kv[0], k], 2)
    Vv = torch.cat([kv[1], v], 2)
    s = q @ K.transpose(-1, -2) / math.sqrt(attn.dh)                  # (B, H, 1, T)
    if bias is not None:
        s = s + bias[:, None, None, : s.shape[-1]]
    a = torch.softmax(s, -1)
    x = x + attn._merge(a @ Vv)
    return x + blk.mlp(blk.ln2(x)), (K, Vv)


def _extend(model, x, cache, key_bias):
    x = x[:, None]
    new = []
    for li, (blk, kv) in enumerate(zip(model.blocks, cache)):
        x, kv = _step(blk, x, kv, None if key_bias is None else key_bias.get(li))
        new.append(kv)
    return model._thought(x[:, 0]), new


def _override(cache, pos, entries):
    out = []
    for (Kc, Vc), (k, v) in zip(cache, entries):
        Kc = Kc.clone(); Vc = Vc.clone()
        Kc[:, :, pos] = k
        Vc[:, :, pos] = v
        out.append((Kc, Vc))
    return out


def cache_entries(cache, pos):
    """[(k, v)] per layer at absolute position pos, each (B, H, dh)."""
    return [(Kc[:, :, pos].clone(), Vc[:, :, pos].clone()) for Kc, Vc in cache]


@torch.no_grad()
def forward(model, src, dst, start, K, qnodes, thought_edit=None, kv_override=None, key_bias=None,
            record=False, query_kv_override=None, target=None):
    """target (B,) : upfront format — TARGET token at position START+1, latents shift by one,
    thought 0 is the TARGET output, and qnodes=None asks the single ANSWER query."""
    te = model.type_emb.weight
    thought_edit = thought_edit or {}
    kv_override = kv_override or {}
    x = model.src_emb(src) + model.dst_emb(dst) + te[T_EDGE]
    _, cache, _ = model.encode(x)
    for pos in [p for p in kv_override if p < START]:
        cache = _override(cache, pos, kv_override[pos])
    h, cache = _extend(model, model.node_emb(start) + te[T_START], cache, key_bias)
    if START in kv_override:
        cache = _override(cache, START, kv_override[START])
    off = START
    if target is not None:
        from .model import T_TARGET
        h, cache = _extend(model, model.node_emb(target) + te[T_TARGET], cache, key_bias)
        off = START + 1
        if off in kv_override:
            cache = _override(cache, off, kv_override[off])
    thoughts, caches = [h], [cache]
    for k in range(1, K + 1):
        if (k - 1) in thought_edit:
            h = thought_edit[k - 1](h)
            thoughts[-1] = h
        base = model.pause.expand_as(h) if model.mode == "pause" else h
        xin = base + te[T_LAT] + model.step_emb.weight[k]
        h, cache = _extend(model, xin, cache, key_bias)
        if off + k in kv_override:
            cache = _override(cache, off + k, kv_override[off + k])
        thoughts.append(h)
        caches.append(cache)
    for pos, entries in (query_kv_override or {}).items():     # what only the queries read
        cache = _override(cache, pos, entries)
    logit = model.query(cache, qnodes, B=src.shape[0])
    out = {"logit": logit}
    if record:
        out["thoughts"] = torch.stack(thoughts, 1)
        out["cache"] = cache
    return out
