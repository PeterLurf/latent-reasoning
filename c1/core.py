"""THEORY §67 (C1 Design 1): a six-code categorical latent with GPT-2 as the transition cell, trained by exact
marginalisation (final answers only), executed hard at test.

S3: elements are the 6 permutations of (0, 1, 2), index 0 = identity. State update s_t = a_t o s_{t-1}
((a o s)[i] = a[s[i]]); the answer is the index of s_N. Operations are uniform random S3 elements.

Cell (arm d1): original GPT-2 small. Input embeddings [code_emb[c], wte(op token), wte(marker)], positions 0-2 at every
call (no KV cache, nothing else crosses calls). A 6-way head on the last position gives the next code. Decoder call:
[code_emb[c], wte(terminal marker)] -> 6-way answer head. Initial code fixed (0).
TAB: a learned (6 ops, 6 codes, 6 codes) transition table and a (6, 6) decoder table, same marginalisation.
RESET: the d1 cell, but every round starts from the initial code (training and inference).
FP: full-prompt GPT-2, no latent: [wte(op tokens) ..., wte(terminal marker)] -> 6-way head at the last position.
"""
import itertools
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PERMS = sorted(itertools.permutations(range(3)))           # index 0 = (0, 1, 2) = identity
COMP = np.array([[PERMS.index(tuple(a[s[i]] for i in range(3))) for s in PERMS] for a in PERMS])  # COMP[a, s] = a o s
M = 6
# GPT-2 tokens: op symbols " A".." F", call marker " step", terminal marker " answer" (single tokens in GPT-2's BPE)
OP_TOK = [317, 347, 327, 360, 412, 376]
STEP_TOK, ANS_TOK = 2239, 3280


def truth(ops):
    s = 0
    for a in ops:
        s = COMP[a, s]
    return int(s)


def sample(rng, B, N):
    ops = rng.integers(0, 6, size=(B, N))
    y = np.array([truth(r) for r in ops])
    return ops, y


class Cell(nn.Module):
    """arm d1 / reset: GPT-2 transition cell + decoder call over learned code embeddings."""
    def __init__(self, gpt):
        super().__init__()
        self.g = gpt
        self.code = nn.Parameter(gpt.embed(torch.tensor(OP_TOK)).detach().clone().mean(0, keepdim=True).repeat(M, 1)
                                 + 0.02 * torch.randn(M, 768))
        self.next_head = nn.Linear(768, M); self.ans_head = nn.Linear(768, 6)

    def step_logits(self, codes, ops):
        """codes, ops: (B,) long -> (B, M) next-code logits (one GPT-2 call per row, positions 0-2)."""
        dev = self.code.device
        x = torch.stack([self.code[codes], self.g.embed(torch.tensor(OP_TOK, device=dev)[ops]),
                         self.g.embed(torch.full_like(ops, STEP_TOK))], 1)
        h, _ = self.g(x)
        return self.next_head(h[:, -1])

    def dec_logits(self, codes):
        x = torch.stack([self.code[codes], self.g.embed(torch.full_like(codes, ANS_TOK))], 1)
        h, _ = self.g(x)
        return self.ans_head(h[:, -1])

    def tables(self):
        """log T (6 ops, M, M) over all 36 (code, op) pairs and log D (M, 6): 42 GPT-2 calls."""
        dev = self.code.device
        c = torch.arange(M, device=dev).repeat(6); a = torch.arange(6, device=dev).repeat_interleave(M)
        logT = self.step_logits(c, a).log_softmax(-1).view(6, M, M)
        logD = self.dec_logits(torch.arange(M, device=dev)).log_softmax(-1)
        return logT, logD


class Tab(nn.Module):
    def __init__(self):
        super().__init__()
        self.T = nn.Parameter(0.01 * torch.randn(6, M, M)); self.D = nn.Parameter(0.01 * torch.randn(M, 6))

    def tables(self):
        return self.T.log_softmax(-1), self.D.log_softmax(-1)

    def step_logits(self, codes, ops):
        return self.T[ops, codes]

    def dec_logits(self, codes):
        return self.D[codes]


def marginal_loss(logT, logD, ops, y, reset=False):
    """Exact forward marginalisation: -log sum_i alpha_N(i) D(i, y), alpha_0 = onehot(0); reset: alpha_{t-1} = onehot(0)."""
    B, N = ops.shape; dev = logT.device
    la = torch.full((B, M), -math.inf, device=dev); la[:, 0] = 0.0
    for t in range(N):
        lt = logT[ops[:, t]]                                       # (B, M, M)
        if reset:
            la = lt[:, 0]
        else:
            la = torch.logsumexp(la[:, :, None] + lt, 1)
    return -torch.logsumexp(la + logD[:, y].T, 1).mean()


@torch.no_grad()
def hard_rollout(model, ops, reset=False, start=None, from_step=0, upto=None):
    """Real cell calls, argmax code after each op. ops: (B, N) long tensor. start: (B,) codes to start from (default
    initial code 0) at step from_step; runs ops[:, from_step:upto]. Returns final codes, decoded answers, list of codes."""
    B, N = ops.shape; dev = ops.device
    c = torch.zeros(B, dtype=torch.long, device=dev) if start is None else start.clone()
    traj = [c]
    for t in range(from_step, N if upto is None else upto):
        c = model.step_logits(torch.zeros_like(c) if reset else c, ops[:, t]).argmax(-1); traj.append(c)
    return c, model.dec_logits(c).argmax(-1), traj


class FP(nn.Module):
    """Full-prompt GPT-2 (no latent): op tokens then the terminal marker, 6-way head at the last position."""
    def __init__(self, gpt):
        super().__init__()
        self.g = gpt; self.head = nn.Linear(768, 6)

    def forward(self, ops):
        dev = ops.device
        ids = torch.cat([torch.tensor(OP_TOK, device=dev)[ops], torch.full((ops.shape[0], 1), ANS_TOK, device=dev)], 1)
        h, _ = self.g(self.g.embed(ids))
        return self.head(h[:, -1])


@torch.no_grad()
def audit(model):
    """Decode the 6 hard codes; bijection to S3; all 36 hard transitions vs group composition; initial code = identity.
    Returns a dict with errors and argmax margins."""
    dev = next(model.parameters()).device
    codes = torch.arange(M, device=dev)
    dl = model.dec_logits(codes); dec = dl.argmax(-1).tolist()
    top2 = dl.topk(2, -1).values; dmarg = (top2[:, 0] - top2[:, 1]).tolist()
    bij = sorted(dec) == list(range(6))
    errs, tmarg = [], []
    c = codes.repeat(6); a = torch.arange(6, device=dev).repeat_interleave(M)
    tl = model.step_logits(c, a); nxt = tl.argmax(-1).tolist()
    t2 = tl.topk(2, -1).values; tmarg = (t2[:, 0] - t2[:, 1]).tolist()
    for ci, ai, ni in zip(c.tolist(), a.tolist(), nxt):
        if dec[ni] != COMP[ai, dec[ci]]:
            errs.append((ci, ai, ni))
    return {"decoded": dec, "bijection": bij, "init_is_identity": dec[0] == 0, "transition_errors": len(errs),
            "error_list": errs[:36], "min_dec_margin": float(min(dmarg)), "min_trans_margin": float(min(tmarg)),
            "certified": bij and dec[0] == 0 and not errs}


def wilson_upper(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    return float((p + z * z / (2 * n) + z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d)
