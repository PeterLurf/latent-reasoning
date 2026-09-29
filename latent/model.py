"""Small decoder-only transformer with Coconut-style continuous-thought recurrence.

Sequence layout (format "delayed", the shared-state diagnostic):
    [edge_1 ... edge_E] [START(s)] [LAT_1 ... LAT_K] | QUERY(t_1) ... QUERY(t_Q)
Sequence layout (format "upfront", conventional reachability):
    [edge_1 ... edge_E] [START(s)] [TARGET(t)] [LAT_1 ... LAT_K] | ANSWER

Thought h_k is the final-LayerNorm output at the k-th recurrent position
(h_0 = output at the last prefix token). In ``mode='coconut'`` the input
embedding of LAT_{k+1} is h_k (+ a step embedding, analogous to GPT-2 adding
position embeddings to inputs_embeds). In ``mode='pause'`` LAT inputs are a
learned constant (extra positions, no hidden-state feedback).

Queries after '|' each attend to the full cached prefix and to themselves
only, so Q queries against one saved prefix are Q independent forks of the
identical state. This is exactly equivalent to running Q separate sequences
(checked in tests.py).

Intervention interface (``patch`` callback): h_k -> h_k' is applied to the
thought *before* it becomes the input of LAT_{k+1}. KV-cache entries of
positions <= k are untouched; positions k+1..K and all queries are recomputed.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .graphs import V

T_EDGE, T_START, T_TARGET, T_LAT, T_QUERY, T_ANSWER, T_PARENT, T_CHILDREN = range(8)


class Attn(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h, self.dh = h, d // h
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)

    def _split(self, x):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        return q, k, v  # (B, H, T, dh)

    def _merge(self, y):
        B, H, T, dh = y.shape
        return self.out(y.transpose(1, 2).reshape(B, T, H * dh))

    # explicit attention: faster than SDPA kernels at these tiny sequence lengths
    def prefix(self, x):
        q, k, v = self._split(x)
        T = x.shape[1]
        s = q @ k.transpose(-1, -2) / math.sqrt(self.dh)
        s = s.masked_fill(torch.ones(T, T, dtype=torch.bool, device=x.device).triu(1), float("-inf"))
        return self._merge(torch.softmax(s, -1) @ v), (k, v)

    def step(self, x, kv):
        """x: (B,1,d); attends to cache + itself; returns updated cache."""
        q, k, v = self._split(x)
        K = torch.cat([kv[0], k], 2)
        Vv = torch.cat([kv[1], v], 2)
        s = q @ K.transpose(-1, -2) / math.sqrt(self.dh)
        a = torch.softmax(s, -1)
        if getattr(self, "record", None) is not None:      # analysis only: attention of this position
            self.record.append(a[:, :, 0].detach())          # (B, H, T)
        return self._merge(a @ Vv), (K, Vv)

    def fork(self, x, kv, return_attn=False):
        """x: (B,Q,d) independent queries; each attends to cache + itself only."""
        q, k, v = self._split(x)
        s_c = q @ kv[0].transpose(-1, -2) / math.sqrt(self.dh)      # (B,H,Q,T)
        s_s = (q * k).sum(-1, keepdim=True) / math.sqrt(self.dh)     # (B,H,Q,1)
        a = torch.softmax(torch.cat([s_c, s_s], -1), -1)
        y = a[..., :-1] @ kv[1] + a[..., -1:] * v
        return self._merge(y), (a if return_attn else None)


class Block(nn.Module):
    def __init__(self, d, h, mlp_mult=4):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = Attn(d, h)
        self.mlp = nn.Sequential(nn.Linear(d, mlp_mult * d), nn.GELU(), nn.Linear(mlp_mult * d, d))

    def prefix(self, x):
        a, kv = self.attn.prefix(self.ln1(x))
        x = x + a
        return x + self.mlp(self.ln2(x)), kv

    def step(self, x, kv):
        a, kv = self.attn.step(self.ln1(x), kv)
        x = x + a
        return x + self.mlp(self.ln2(x)), kv

    def fork(self, x, kv, return_attn=False):
        a, att = self.attn.fork(self.ln1(x), kv, return_attn)
        x = x + a
        return x + self.mlp(self.ln2(x)), att


class _SlotEmb(nn.Module):
    """One shared identity table placed into a role slot of the d-dim input.

    Slots: 0 = edge source, 1 = edge target, 2 = node token (start/target/query).
    Sharing the table makes id matching across roles an id-independent linear
    map (a slot permutation), instead of an alignment between independent tables.
    """
    def __init__(self, table, slot, d):
        super().__init__()
        self.table, self.slot, self.d = table, slot, d

    def forward(self, ids):
        e = self.table(ids)
        w = e.shape[-1]
        return F.pad(e, (self.slot * w, self.d - (self.slot + 1) * w))

    @property
    def weight(self):   # (V, d) effective embedding, for analysis
        return self.forward(torch.arange(self.table.num_embeddings, device=self.table.weight.device))


class LatentReasoner(nn.Module):
    def __init__(self, d=256, n_layers=2, n_heads=8, k_max=8, mode="coconut", emb="slots", d_id=64,
                 init="gpt2", feedback="post_ln", step_emb=True):
        super().__init__()
        assert mode in ("coconut", "pause") and emb in ("slots", "untied")
        self.mode, self.d, self.feedback = mode, d, feedback
        if emb == "untied":
            self.src_emb = nn.Embedding(V, d)
            self.dst_emb = nn.Embedding(V, d)
            self.node_emb = nn.Embedding(V, d)
        else:
            assert 3 * d_id <= d
            self.id_emb = nn.Embedding(V, d_id)
            self.src_emb = _SlotEmb(self.id_emb, 0, d)
            self.dst_emb = _SlotEmb(self.id_emb, 1, d)
            self.node_emb = _SlotEmb(self.id_emb, 2, d)
            self.id_out = nn.Linear(d, d_id)
            self.id_out2 = nn.Linear(d, d_id)
        self.type_emb = nn.Embedding(8, d)
        self.step_emb = nn.Embedding(k_max + 1, d)
        self.pause = nn.Parameter(torch.zeros(d))
        self.blocks = nn.ModuleList([Block(d, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, 1)
        if init == "gpt2":
            self.apply(self._init)
        # init == "torch": PyTorch defaults (kaiming-uniform Linear, N(0,1) embeddings)
        self.use_step = step_emb
        if not step_emb:            # ablation: no step embedding (table zeroed; it gets no gradient, so it stays 0)
            nn.init.zeros_(self.step_emb.weight)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def _thought(self, x):
        """Recurrent output fed to the next latent position: final-LN output (Coconut) or, for the
        raw-feedback ablation, the pre-LN residual."""
        return self.ln_f(x) if getattr(self, "feedback", "post_ln") == "post_ln" else x

    # ------------------------------------------------------------ pieces
    def embed_prefix(self, src, dst, start, target=None):
        te = self.type_emb.weight
        x = self.src_emb(src) + self.dst_emb(dst) + te[T_EDGE]
        s = (self.node_emb(start) + te[T_START])[:, None]
        parts = [x, s]
        if target is not None:
            parts.append((self.node_emb(target) + te[T_TARGET])[:, None])
        return torch.cat(parts, 1)

    def encode(self, x, keep_resid=False):
        cache, resid = [], [x] if keep_resid else None
        for blk in self.blocks:
            x, kv = blk.prefix(x)
            cache.append(kv)
            if keep_resid:
                resid.append(x)
        return self._thought(x[:, -1]), cache, resid

    def latent_step(self, h, k, cache, keep_resid=False):
        te = self.type_emb.weight
        base = self.pause.expand_as(h) if self.mode == "pause" else h
        x = (base + te[T_LAT] + (self.step_emb.weight[k] if getattr(self, "use_step", True) else 0))[:, None]
        new, resid = [], [x[:, 0]] if keep_resid else None
        for blk, kv in zip(self.blocks, cache):
            x, kv = blk.step(x, kv)
            new.append(kv)
            if keep_resid:
                resid.append(x[:, 0])
        return self._thought(x[:, 0]), new, resid

    def query(self, cache, qnodes=None, B=None, return_attn=False):
        """qnodes: (B,Q) node ids -> delayed-format queries; None -> single ANSWER token."""
        te = self.type_emb.weight
        if qnodes is None:
            x = te[T_ANSWER].expand(B, 1, self.d)
        else:
            x = self.node_emb(qnodes) + te[T_QUERY]
        atts = []
        for blk, kv in zip(self.blocks, cache):
            x, a = blk.fork(x, kv, return_attn)
            atts.append(a)
        logit = self.head(self.ln_f(x))[..., 0]
        return (logit, atts) if return_attn else logit

    def extend(self, x, cache):
        """Append one ordinary token embedding x (B,d) to the cache (causal-equivalent)."""
        x = x[:, None]
        new = []
        for blk, kv in zip(self.blocks, cache):
            x, kv = blk.step(x, kv)
            new.append(kv)
        return self._thought(x[:, 0]), new

    def parent_logits(self, cache, pnodes):
        """Auxiliary lookup (training scaffold only): PARENT(v) -> logits over ids of v's parent.

        Forks against the *prefix* cache (edges + START, no latent positions), so it
        supervises input lookup only, never the latent thoughts.
        """
        x = self.node_emb(pnodes) + self.type_emb.weight[T_PARENT]
        for blk, kv in zip(self.blocks, cache):
            x, _ = blk.fork(x, kv)
        return self.id_out(self.ln_f(x)) @ self.id_emb.weight.T

    def children_logits(self, cache, cnodes):
        """Auxiliary lookup (training scaffold only): CHILDREN(v) -> multi-label logits over ids.

        Same prefix-only fork as PARENT. Never asked for the start node.
        """
        x = self.node_emb(cnodes) + self.type_emb.weight[T_CHILDREN]
        for blk, kv in zip(self.blocks, cache):
            x, _ = blk.fork(x, kv)
        return self.id_out2(self.ln_f(x)) @ self.id_emb.weight.T

    def run_both(self, src, dst, start, K, qnodes, target, pnodes=None, cnodes=None):
        """Training helper: delayed and upfront formats sharing one edge+start prefix."""
        h, cache, _ = self.encode(self.embed_prefix(src, dst, start))
        te = self.type_emb.weight
        h_u, cache_u = self.extend(self.node_emb(target) + te[T_TARGET], cache)
        pl = self.parent_logits(cache, pnodes) if pnodes is not None else None
        cl = self.children_logits(cache, cnodes) if cnodes is not None else None
        outs = []
        for hh, cc, qq in ((h, cache, qnodes), (h_u, cache_u, None)):
            for k in range(1, K + 1):
                hh, cc, _ = self.latent_step(hh, k, cc)
            outs.append(self.query(cc, qq, B=src.shape[0]))
        return outs[0], outs[1][:, 0], pl, cl

    # ------------------------------------------------------------ full run
    def run(self, src, dst, start, K, target=None, qnodes=None, patch=None,
            keep=False, keep_resid=False):
        """Returns dict(logit, thoughts[, resid]). ``patch(k, h) -> h`` edits thought k."""
        x = self.embed_prefix(src, dst, start, target)
        h, cache, r0 = self.encode(x, keep_resid)
        thoughts, resid = [h], ([r0] if keep_resid else None)
        for k in range(1, K + 1):
            if patch is not None:
                h = patch(k - 1, h)
                thoughts[-1] = h
            h, cache, r = self.latent_step(h, k, cache, keep_resid)
            thoughts.append(h)
            if keep_resid:
                resid.append(r)
        logit = self.query(cache, qnodes, B=src.shape[0])
        out = {"logit": logit}
        if keep:
            out["thoughts"] = torch.stack(thoughts, 1)   # (B, K+1, d)
            out["cache"] = cache
        if keep_resid:
            out["resid"] = resid
        return out
