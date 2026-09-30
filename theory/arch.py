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


class SetMemReasoner(LatentReasoner):
    """A6 (THEORY §22.4): monotone latent set memory next to the continuous thought.

    After every latent step the output thought is scored against every node id (shared id table), passed through a
    per-id sigmoid (non-competitive membership) and merged into the set by an elementwise max:
        m_k = max(m_{k-1}, sigmoid(beta * (r(h_k) @ E_id.T - tau)))      (m_0 = 0; non-decreasing in k)
    The set is fed back into the next latent input, h_k + gamma * W(m_k @ E_id), with gamma initialised to 0 (the
    restore module's learnability failure, §19, came from an untrained write at full strength). A found node can
    therefore never fade from the transition's input (Prop. 6a); the queries still read every latent position.
    """
    def __init__(self, *a, set_beta=8.0, set_tau=0.5, **kw):
        super().__init__(*a, **kw)
        d, d_id = self.d, self.id_emb.weight.shape[1]
        self.s_read = nn.Linear(d, d_id)
        self.s_write = nn.Linear(d_id, d, bias=False)
        for m_ in (self.s_read, self.s_write):
            nn.init.normal_(m_.weight, std=0.02)
        nn.init.zeros_(self.s_read.bias)
        self.s_beta = nn.Parameter(torch.tensor(float(set_beta)))
        self.s_tau = nn.Parameter(torch.tensor(float(set_tau)))
        self.s_gate = nn.Parameter(torch.zeros(()))
        self.register_buffer("set_on", torch.ones(()), persistent=False)   # 0 = set write disabled (--set_from_stage)
        self._m = None

    def set_scores(self, h):
        E = self.id_emb.weight
        return torch.sigmoid(self.s_beta * (self.s_read(h) @ E.T - self.s_tau))

    def latent_step(self, h, k, cache, keep_resid=False):
        if k == 1 or self._m is None or self._m.shape[0] != h.shape[0]:
            self._m = torch.zeros(h.shape[0], self.id_emb.weight.shape[0], device=h.device, dtype=h.dtype)
        x = h + (self.set_on * self.s_gate) * self.s_write(self._m.to(h.dtype) @ self.id_emb.weight.to(h.dtype))
        h2, new, resid = super().latent_step(x, k, cache, keep_resid)
        self._m = torch.maximum(self._m.to(h2.dtype), self.set_scores(h2).to(h2.dtype))
        return h2, new, resid


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



class ChunkReasoner(MarkovReasoner):
    """A8 / Pilot J (THEORY §29): chunk-reset latent attention. Latent step k attends to the prefix and only to the
    latent positions of its own chunk of L steps (steps ((k-1)//L)*L + 1 .. k-1); at every chunk boundary the latent
    history is cut, so the transition sees at most L-1 earlier thoughts and its count restarts — a finite, periodic
    history window (the continuous-thought-machine sliding window; a clock that cannot run out). The thought itself
    carries state across chunks; the queries still read every latent position."""
    def __init__(self, *a, chunk=3, **kw):
        super().__init__(*a, markov_layers=(), **kw)
        self.chunk = int(chunk)

    def latent_step(self, h, k, cache, keep_resid=False):
        P = self._P
        cs = ((k - 1) // self.chunk) * self.chunk + 1            # first step of this chunk
        te = self.type_emb.weight
        base = self.pause.expand_as(h) if self.mode == "pause" else h
        x = (base + te[T_LAT] + (self.step_emb.weight[k] if self.use_step else 0))[:, None]
        new = []
        for blk, kv in zip(self.blocks, cache):
            lo = P + cs - 1                                        # cache index of the chunk's first latent position
            vis = (torch.cat([kv[0][:, :, :P], kv[0][:, :, lo:]], 2), torch.cat([kv[1][:, :, :P], kv[1][:, :, lo:]], 2))
            x_new, kv2 = blk.step(x, vis)
            new.append((torch.cat([kv[0], kv2[0][:, :, -1:]], 2), torch.cat([kv[1], kv2[1][:, :, -1:]], 2)))
            x = x_new
        return self._thought(x[:, 0]), new, None


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


def add_step_adversary(model, lam, n_classes, hidden=128):
    """Proposal T12 (PROPOSALS §10, THEORY §39): step-adversarial thoughts, confusion form. While training, every
    thought produced by a latent step (h_k, k = 1..K, delayed and upfront chains) is recorded. A 2-layer probe is
    trained to predict k from the detached thoughts (cross-entropy); the model is trained, with weight lam, to make the
    probe's output uniform over 1..K (cross-entropy to the uniform target through a frozen copy of the probe) -- the
    domain-confusion objective, which unlike plain gradient reversal cannot be met by making the probe confidently
    wrong. The probe is not a registered submodule: the state dict is unchanged and evaluation is untouched. Returns
    the probe (its parameters go to the optimizer). model.adv_last holds the probe's last CE (device tensor, updated by
    CUDA-graph replays; ln K = chance)."""
    import torch.nn.functional as F
    dev = next(model.parameters()).device
    probe = nn.Sequential(nn.Linear(model.d, hidden), nn.GELU(), nn.Linear(hidden, n_classes)).to(dev)
    buf = []
    object.__setattr__(model, "_step_probe", probe)
    object.__setattr__(model, "_adv_buf", buf)
    object.__setattr__(model, "adv_last", torch.zeros((), device=dev))
    base = model.latent_step

    def latent_step(h, k, cache, keep_resid=False):
        out = base(h, k, cache, keep_resid)
        if model.training:
            buf.append((out[0], k))
        return out

    def step_adv_loss():
        hs = torch.cat([h.float() for h, _ in buf])
        ys = torch.cat([torch.full((h.shape[0],), k, dtype=torch.long, device=h.device) for h, k in buf])
        ks = sorted({k for _, k in buf})
        buf.clear()
        ce = F.cross_entropy(probe(hs.detach()), ys)                        # trains the probe only
        W1, b1, W2, b2 = (t.detach() for t in (probe[0].weight, probe[0].bias, probe[2].weight, probe[2].bias))
        logp = F.log_softmax(F.linear(F.gelu(F.linear(hs, W1, b1)), W2, b2), -1)
        assert ks == list(range(ks[0], ks[-1] + 1))
        # a slice, not a list index: list indexing copies an index tensor host -> device, which a CUDA-graph capture
        # forbids (the §39 runs crashed at capture_end; amendment T12-1)
        conf = -logp[:, ks[0]:ks[-1] + 1].mean()                             # CE to uniform over the used steps
        model.adv_last.copy_(ce.detach())
        return ce + lam * conf

    model.latent_step = latent_step
    object.__setattr__(model, "step_adv_loss", step_adv_loss)
    model.step_adv = lam
    return probe


def add_periodic_step(model, period):
    """Literature-session idea #2 (THEORY §28): a clock that never runs out. The latent step k uses step embedding
    ((k - 1) mod period) + 1 instead of k (the table is trained; use_step is switched on). Applied at training and at
    evaluation (theory/eval_arch.load_arch reads period from arch.json; theory/rollout.py follows model.periodic_step)."""
    base = model.latent_step
    model.use_step = True
    model.periodic_step = int(period)

    def latent_step(h, k, cache, keep_resid=False):
        return base(h, (k - 1) % model.periodic_step + 1, cache, keep_resid)
    model.latent_step = latent_step
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
    if kind == "setmem":
        return SetMemReasoner(**kw)
    if kind.startswith("chunk"):                  # chunk2, chunk3, ...
        return ChunkReasoner(chunk=int(kind[5:]), **kw)
    raise ValueError(kind)
