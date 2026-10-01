"""THEORY §66: GPT-2 state tracking (4 boxes, swap operations; S4) with arms NC, TC, TC2, CL, WC(k), WC0(k).

Text format (GPT-2 BPE; every op and every arrangement has a fixed token length, so batches of equal N and question
type need no padding):
  prompt   "start: B D A C. ops: swap 1 3, swap 2 4. q: all. a:"      answer " B D A C"   (contents of boxes 1..4)
           "... q: where C. a:"                                       answer " 3"         (the box holding C)
  TC       prompt + " cot:" + " > D B A C" per op (TC2: every 2nd op and the last) + " a:" + answer  (step-labelled)
  CL       prompt with "think:" before "a:": K' = N continuous thoughts (Coconut: the last hidden state is fed back as the
           next input embedding), then " done. a:" + answer; final answers only
  WC(k)    short-problem solver: "start: ... ops: <k' <= k ops>. think:" + k' thoughts + " done. a:" + end arrangement;
           at test the long problem is cut every k ops, each chunk's arrangement is decoded greedily and fed as the next
           start (latent history cut); the answer is read from the last arrangement
  WC0(k)   the same solver with no thoughts ("... ops: ... a:")
Compute per problem (KV cache used everywhere: each position is processed once): tokens processed (incl. latent
positions), FLOPs = sum over processed positions of (2 * non-embedding params + 4 * n_layer * d * context) plus
2 * d * vocab for every position whose logits are read, sequential forward passes, written (generated) tokens.
"""
import math
import random

import numpy as np
import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F

ENC = tiktoken.get_encoding("gpt2")
LET = "ABCD"


# ---------------------------------------------------------------- task
def make_problem(rng, n, qtype=None):
    start = list(LET); rng.shuffle(start)
    ops = []
    for _ in range(n):
        i, j = sorted(rng.sample(range(4), 2)); ops.append((i, j))
    qtype = qtype or rng.choice(["all", "where"])
    target = rng.choice(LET)
    return {"start": "".join(start), "ops": ops, "qtype": qtype, "target": target}


def apply(arr, ops):
    a = list(arr)
    for i, j in ops:
        a[i], a[j] = a[j], a[i]
    return "".join(a)


def answer_text(arr, qtype, target):
    return " " + " ".join(arr) if qtype == "all" else " " + str(arr.index(target) + 1)


def truth(p):
    return answer_text(apply(p["start"], p["ops"]), p["qtype"], p["target"])


def s_arr(arr):
    return " ".join(arr)


def s_ops(ops):
    return ", ".join(f"swap {i + 1} {j + 1}" for i, j in ops)


def s_q(p):
    return "q: all." if p["qtype"] == "all" else f"q: where {p['target']}."


def prompt(p, think=False, start=None, ops=None, q=True):
    start = p["start"] if start is None else start; ops = p["ops"] if ops is None else ops
    s = f"start: {s_arr(start)}. ops: {s_ops(ops)}."
    if q:
        s += " " + s_q(p)
    return s + (" think:" if think else " a:")


def tc_text(p, every=1):
    a = list(p["start"]); out = []
    for t, (i, j) in enumerate(p["ops"]):
        a[i], a[j] = a[j], a[i]
        if (t + 1) % every == 0 or t == len(p["ops"]) - 1:
            out.append(" > " + s_arr(a))
    return " cot:" + "".join(out) + " a:"


def tok(s):
    return ENC.encode(s)


# ---------------------------------------------------------------- model (GPT-2 small, KV cache, trainable)
class GPT2KV(nn.Module):
    def __init__(self, sd):
        super().__init__()
        self.L = sum(1 for k in sd if k.startswith("h.") and k.endswith(".ln_1.weight")); self.H = 12; self.D = 768
        self.p = nn.ParameterDict({k.replace(".", "_"): nn.Parameter(v.float().clone()) for k, v in sd.items()
                                   if not k.endswith(".attn.bias") and not k.endswith(".attn.masked_bias")})
        self.n_nonemb = sum(v.numel() for k, v in self.p.items() if k.startswith("h_") or k.startswith("ln_f"))

    def w(self, k):
        return self.p[k.replace(".", "_")]

    def embed(self, ids):
        return self.w("wte.weight")[ids]

    def forward(self, x, past=None):
        """x: (B, T, D) input embeddings appended after `past` (list per layer of (k, v)). Returns final hidden (after
        ln_f), new past."""
        B, T, D = x.shape; P0 = 0 if past is None else past[0][0].shape[2]; hd = D // self.H
        h = x + self.w("wpe.weight")[P0:P0 + T][None]
        new = []
        for i in range(self.L):
            pre = f"h.{i}."
            a = F.layer_norm(h, (D,), self.w(pre + "ln_1.weight"), self.w(pre + "ln_1.bias"))
            q, k, v = (a @ self.w(pre + "attn.c_attn.weight") + self.w(pre + "attn.c_attn.bias")).split(D, -1)
            q, k, v = (t.view(B, T, self.H, hd).transpose(1, 2) for t in (q, k, v))
            if past is not None:
                k = torch.cat([past[i][0], k], 2); v = torch.cat([past[i][1], v], 2)
            new.append((k, v))
            if P0 == 0:
                o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            else:
                mask = torch.ones(T, P0 + T, dtype=torch.bool, device=x.device).tril(P0)
                o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            h = h + o.transpose(1, 2).reshape(B, T, D) @ self.w(pre + "attn.c_proj.weight") + self.w(pre + "attn.c_proj.bias")
            a = F.layer_norm(h, (D,), self.w(pre + "ln_2.weight"), self.w(pre + "ln_2.bias"))
            m = F.gelu(a @ self.w(pre + "mlp.c_fc.weight") + self.w(pre + "mlp.c_fc.bias"), approximate="tanh")
            h = h + m @ self.w(pre + "mlp.c_proj.weight") + self.w(pre + "mlp.c_proj.bias")
        return F.layer_norm(h, (D,), self.w("ln_f.weight"), self.w("ln_f.bias")), new

    def logits(self, h):
        return h @ self.w("wte.weight").T


class Meter:
    """Per-problem compute (averaged over the batch rows, which are identical in shape)."""
    def __init__(self, m):
        self.m = m; self.tokens = 0; self.flops = 0.0; self.passes = 0; self.written = 0

    def step(self, T, ctx_before, logit_rows):
        for t in range(T):
            self.flops += 2 * self.m.n_nonemb + 4 * self.m.L * self.m.D * (ctx_before + t + 1)
        self.flops += 2 * self.m.D * 50257 * logit_rows
        self.tokens += T; self.passes += 1

    def as_dict(self):
        return {"tokens_processed": self.tokens, "flops": self.flops, "seq_passes": self.passes, "written_tokens": self.written,
                "latent_passes": 0, "gen_tokens": self.written}


def run_prefix(m, ids, past=None, meter=None, need_logits=False):
    x = m.embed(ids); ctx = 0 if past is None else past[0][0].shape[2]
    h, past = m(x, past)
    if meter:
        meter.step(ids.shape[1], ctx, 1 if need_logits else 0)
    return h, past


def latents(m, h_last, past, K, meter=None):
    """K Coconut thoughts: feed the last hidden state back as the next input embedding."""
    for _ in range(K):
        ctx = past[0][0].shape[2]
        h, past = m(h_last[:, None, :], past)
        if meter:
            meter.step(1, ctx, 0); meter.lat = getattr(meter, "lat", 0) + 1
        h_last = h[:, -1]
    return h_last, past


def greedy(m, h_last, past, n, meter=None, allowed=None):
    """Greedy-decode n tokens (allowed: optional list of token ids to restrict to)."""
    out = []
    for t in range(n):
        lg = m.logits(h_last)
        if allowed is not None:
            mask = torch.full_like(lg, float("-inf")); mask[:, allowed] = 0; lg = lg + mask
        nxt = lg.argmax(-1); out.append(nxt)
        if meter:
            meter.written += 1
            meter.flops += 2 * m.D * 50257
        if t < n - 1:
            ctx = past[0][0].shape[2]
            h, past = m(m.embed(nxt)[:, None, :], past)
            if meter:
                meter.step(1, ctx, 0)
            h_last = h[:, -1]
    return torch.stack(out, 1)


def gen_feed(m, h_last, past, n, meter=None, allowed=None):
    """Greedy-decode n tokens and feed every one of them into the cache. Returns (ids, last hidden, past)."""
    out = []
    for _ in range(n):
        lg = m.logits(h_last)
        if allowed is not None:
            mask = torch.full_like(lg, float("-inf")); mask[:, allowed] = 0; lg = lg + mask
        nxt = lg.argmax(-1); out.append(nxt)
        ctx = past[0][0].shape[2]; h, past = m(m.embed(nxt)[:, None, :], past)
        if meter:
            meter.written += 1; meter.step(1, ctx, 1)
        h_last = h[:, -1]
    return torch.stack(out, 1), h_last, past


LET_IDS = [tok(" " + c)[0] for c in LET]
DIG_IDS = [tok(" " + str(d))[0] for d in range(1, 5)]


def dec_arr(ids):
    return "".join(ENC.decode([int(i)]).strip() for i in ids)


# ---------------------------------------------------------------- training losses (teacher forcing on the answer)
def tensor(rows, dev):
    return torch.tensor(rows, dtype=torch.long, device=dev)


def loss_batch(m, arm, k, batch, dev):
    """batch: list of problems with equal shapes. Returns CE loss on the answer (+ CoT text for TC)."""
    if arm in ("nc", "tc", "tc2"):
        pre = [tok(prompt(p)) for p in batch] if arm == "nc" else \
              [tok(prompt(p)[:-3] + tc_text(p, 1 if arm == "tc" else 2)) for p in batch]
        tgt = [tok(truth(p)) for p in batch]
        if arm != "nc":                                          # CoT tokens are supervised too (step-labelled)
            base = [tok(prompt(p)[:-3]) for p in batch]
            ids = tensor([a + b for a, b in zip(pre, tgt)], dev)
            h, _ = m(m.embed(ids[:, :-1]))
            lg = m.logits(h); y = ids[:, 1:].clone()
            y[:, : len(base[0]) - 1] = -100
            return F.cross_entropy(lg.reshape(-1, lg.shape[-1]), y.reshape(-1), ignore_index=-100)
        ids = tensor([a + b for a, b in zip(pre, tgt)], dev)
        h, _ = m(m.embed(ids[:, :-1])); lg = m.logits(h[:, len(pre[0]) - 1:])
        return F.cross_entropy(lg.reshape(-1, lg.shape[-1]), ids[:, len(pre[0]):].reshape(-1))
    # latent arms: cl (K' = N, question included), wc / wc0 (solver: end arrangement, no question)
    q = arm == "cl"
    if arm == "cl":
        K = len(batch[0]["ops"])
    elif arm == "wc":
        K = len(batch[0]["ops"])                                 # k' thoughts for a k'-op chunk
    else:
        K = 0
    pre = tensor([tok(prompt(p, think=K > 0, q=q)) for p in batch], dev)
    h, past = m(m.embed(pre))
    hl = h[:, -1]
    if K:
        hl, past = latents(m, hl, past, K)
        mid = tensor([tok(" done. a:")] * len(batch), dev)
        h2, past = m(m.embed(mid), past); hl = h2[:, -1]
    tgt = tensor([tok(truth(p) if q else answer_text(apply(p["start"], p["ops"]), "all", "A")) for p in batch], dev)
    x = m.embed(tgt[:, :-1])
    h3, _ = m(x, past) if tgt.shape[1] > 1 else (None, None)
    hs = torch.cat([hl[:, None]] + ([h3] if h3 is not None else []), 1)
    lg = m.logits(hs)
    return F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1))


# ---------------------------------------------------------------- inference (batch of problems with equal N, qtype)
@torch.no_grad()
def infer(m, arm, k, batch, dev, edit=None):
    """Returns (list of answer strings, compute dict per problem). edit: for WC, {"chunk": c, "arr": [...]} replaces the
    written arrangement after chunk c by the given arrangements (per row)."""
    meter = Meter(m); B = len(batch); N = len(batch[0]["ops"]); qt = batch[0]["qtype"]
    nans = 4 if qt == "all" else 1
    allowed = LET_IDS if qt == "all" else DIG_IDS
    if arm in ("nc", "cl", "tc", "tc2"):
        if arm == "cl":
            pre = tensor([tok(prompt(p, think=True)) for p in batch], dev)
            h, past = run_prefix(m, pre, meter=meter); hl, past = latents(m, h[:, -1], past, N, meter)
            mid = tensor([tok(" done. a:")] * B, dev); h, past = run_prefix(m, mid, past, meter, True); hl = h[:, -1]
        elif arm == "nc":
            pre = tensor([tok(prompt(p)) for p in batch], dev); h, past = run_prefix(m, pre, meter=meter, need_logits=True); hl = h[:, -1]
        else:
            pre = tensor([tok(prompt(p)[:-3] + " cot:") for p in batch], dev); h, past = run_prefix(m, pre, meter=meter, need_logits=True)
            nstates = N if arm == "tc" else math.ceil(N / 2)
            hl = h[:, -1]
            gt = tok(" >")
            for s_ in range(nstates):                            # each state: " >" (forced) then 4 greedy letters, fed back
                ctx = past[0][0].shape[2]; h, past = m(m.embed(tensor([gt] * B, dev)), past); meter.step(1, ctx, 1)
                meter.written += 1; hl = h[:, -1]
                _, hl, past = gen_feed(m, hl, past, 4, meter, LET_IDS)
            ctx = past[0][0].shape[2]; h, past = m(m.embed(tensor([tok(" a:")] * B, dev)), past)
            meter.step(len(tok(" a:")), ctx, 1); hl = h[:, -1]
        ans = greedy(m, hl, past, nans, meter, allowed)
        texts = [" " + " ".join(ENC.decode([int(t)]).strip() for t in row) for row in ans]
        c = meter.as_dict(); c["latent_passes"] = getattr(meter, "lat", 0)
        return texts, c
    # WC / WC0: chunked solver
    arrs = [p["start"] for p in batch]
    chunks = [list(range(s, min(s + k, N))) for s in range(0, N, k)]
    for ci, ch in enumerate(chunks):
        sub = [dict(p, start=a, ops=[p["ops"][t] for t in ch]) for p, a in zip(batch, arrs)]
        K = len(ch) if arm == "wc" else 0
        pre = tensor([tok(prompt(sp, think=K > 0, q=False)) for sp in sub], dev)
        h, past = run_prefix(m, pre, meter=meter, need_logits=K == 0); hl = h[:, -1]
        if K:
            hl, past = latents(m, hl, past, K, meter)
            mid = tensor([tok(" done. a:")] * B, dev); h, past = run_prefix(m, mid, past, meter, True); hl = h[:, -1]
        out = greedy(m, hl, past, 4, meter, LET_IDS)
        arrs = [dec_arr(r) for r in out]
        if edit is not None and ci == edit["chunk"]:
            arrs = list(edit["arr"])
    texts = [answer_text(a if len(set(a)) == 4 else "ABCD", qt, p["target"]) if qt == "where" else " " + " ".join(a)
             for a, p in zip(arrs, batch)]
    c = meter.as_dict(); c["latent_passes"] = getattr(meter, "lat", 0)
    return texts, c
