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
        self.hn = nn.Identity()                                    # head-input norm (pilot option: ln)

    def ln_heads(self):
        """Implementation option (pilot; Audit): a learned LayerNorm on the cell output before both heads, addressing
        GPT-2's outlier dimension directly."""
        self.hn = nn.LayerNorm(768).to(self.code.device)

    def small_heads(self, logit_std=0.5, hidden_norm=200.0):
        """Implementation option (pilot): heads scaled so initial logits have std ~logit_std (GPT-2 hidden norm ~200)."""
        for h in (self.next_head, self.ans_head):
            nn.init.normal_(h.weight, std=logit_std / hidden_norm); nn.init.zeros_(h.bias)

    def distinct_codes(self, std=0.14):
        """Implementation option (pilot): codes drawn i.i.d. N(0, std^2) per coordinate (GPT-2 wte coordinate std ~0.14)
        instead of a shared mean + 0.02 noise, which leaves all codes nearly identical (a symmetric saddle)."""
        with torch.no_grad():
            self.code.copy_(std * torch.randn_like(self.code))

    def zero_heads(self):
        """Implementation option (pilot, logged in c1/PILOT.md): zero-initialised heads. GPT-2's last hidden state has
        norm ~200 with one outlier dimension, so default-initialised heads start with logit std ~5 and saturate."""
        for h in (self.next_head, self.ans_head):
            nn.init.zeros_(h.weight); nn.init.zeros_(h.bias)

    def bil_readout(self, hid=256):
        """§67b (+ amendment 67b-2): next-code logits = W ReLU((U LN(h_code)) * (V LN(h_op))), h_code / h_op = GPT-2's final hidden states at the
        code-embedding and op-symbol positions of the same call."""
        dev = self.code.device
        self.ln_c, self.ln_o = nn.LayerNorm(768).to(dev), nn.LayerNorm(768).to(dev)
        self.U, self.V, self.W = nn.Linear(768, hid).to(dev), nn.Linear(768, hid).to(dev), nn.Linear(hid, M).to(dev)
        self.readout = "bil"

    feat_norm = "ln"          # §72: "ln" (learned LayerNorm, §67b) or "tablestd" (per-coordinate standardisation across
                              # the rows of the full table; no affine; stats from the full table, identical at test)

    def _raw_step(self, codes, ops):
        dev = self.code.device
        x = torch.stack([self.code[codes], self.g.embed(torch.tensor(OP_TOK, device=dev)[ops]),
                         self.g.embed(torch.full_like(ops, STEP_TOK))], 1)
        h, _ = self.g(x)
        return h

    def _raw_dec(self, codes):
        x = torch.stack([self.code[codes], self.g.embed(torch.full_like(codes, ANS_TOK))], 1)
        h, _ = self.g(x)
        return h[:, -1]

    def _full_stats(self):
        """Full-table statistics (no grad) for tablestd at test / single-batch calls."""
        dev = self.code.device
        with torch.no_grad():
            c = torch.arange(M, device=dev).repeat(6); a = torch.arange(6, device=dev).repeat_interleave(M)
            h = self._raw_step(c, a); hd = self._raw_dec(torch.arange(M, device=dev))
            self._stats = (h[:, 0].mean(0), h[:, 0].std(0) + 1e-5, h[:, 1].mean(0), h[:, 1].std(0) + 1e-5,
                           hd.mean(0), hd.std(0) + 1e-5)
        return self._stats

    def head_inputs(self, codes, ops, full=False):
        """(h_code, h_op, raw h) as fed to U, V after the feature normalisation."""
        h = self._raw_step(codes, ops)
        if self.feat_norm == "tablestd":
            if full:
                mc, sc, mo, so = h[:, 0].mean(0), h[:, 0].std(0) + 1e-5, h[:, 1].mean(0), h[:, 1].std(0) + 1e-5
            else:
                st = getattr(self, "_stats", None) or self._full_stats(); mc, sc, mo, so = st[:4]
            return (h[:, 0] - mc) / sc, (h[:, 1] - mo) / so, h
        return self.ln_c(h[:, 0]), self.ln_o(h[:, 1]), h

    def step_logits(self, codes, ops, full=False):
        """codes, ops: (B,) long -> (B, M) next-code logits (one GPT-2 call per row, positions 0-2)."""
        if getattr(self, "readout", "linear") == "bil":
            hc, ho, _ = self.head_inputs(codes, ops, full)
            return self.W(torch.relu(self.U(hc) * self.V(ho)))       # 67b-2: ReLU
        h = self._raw_step(codes, ops)
        return self.next_head(self.hn(h[:, -1]))

    def dec_logits(self, codes, full=False):
        h = self._raw_dec(codes)
        if self.feat_norm == "tablestd":
            if full:
                md, sd = h.mean(0), h.std(0) + 1e-5
            else:
                st = getattr(self, "_stats", None) or self._full_stats(); md, sd = st[4], st[5]
            return self.ans_head((h - md) / sd)
        return self.ans_head(self.hn(h))

    def tables(self, tau=1.0):
        """log T (6 ops, M, M) over all M x 6 (code, op) pairs and log D (M, 6); T = softmax(logits / tau) (§67b)."""
        dev = self.code.device
        c = torch.arange(M, device=dev).repeat(6); a = torch.arange(6, device=dev).repeat_interleave(M)
        logT = (self.step_logits(c, a, full=True) / tau).log_softmax(-1).view(6, M, M)
        logD = self.dec_logits(torch.arange(M, device=dev), full=True).log_softmax(-1)
        self._stats = None                                            # stale after this update; recomputed on demand
        return logT, logD

    @torch.no_grad()
    def distinctness(self):
        """§72: mean pairwise cosine across codes of the head input h_code (per code averaged over the 6 ops), after the
        feature normalisation ("normed") and the GPT-2 final hidden state itself ("raw")."""
        dev = self.code.device
        c = torch.arange(M, device=dev).repeat(6); a = torch.arange(6, device=dev).repeat_interleave(M)
        self._stats = None
        hc, _, h = self.head_inputs(c, a, full=True)
        out = {}
        for name, v in (("normed", hc), ("raw", h[:, 0])):
            per = F.normalize(v.float().view(6, M, -1).mean(0), dim=-1); cos = per @ per.T
            out[f"mean_pairwise_cos_{name}"] = float((cos.sum() - cos.diag().sum()) / (M * (M - 1)))
        return out


class Tab(nn.Module):
    def __init__(self):
        super().__init__()
        self.T = nn.Parameter(0.01 * torch.randn(6, M, M)); self.D = nn.Parameter(0.01 * torch.randn(M, 6))

    def tables(self, tau=1.0):
        return (self.T / tau).log_softmax(-1), self.D.log_softmax(-1)

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
def hard_rollout(model, ops, reset=False, start=None, from_step=0, upto=None, chunk=2048):
    """Chunked over rows (memory only; rows are independent). See _hard_rollout."""
    if ops.shape[0] <= chunk:
        return _hard_rollout(model, ops, reset, start, from_step, upto)
    parts = [_hard_rollout(model, ops[i:i + chunk], reset, None if start is None else start[i:i + chunk], from_step, upto)
             for i in range(0, ops.shape[0], chunk)]
    return (torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts]),
            [torch.cat([p[2][t] for p in parts]) for t in range(len(parts[0][2]))])


@torch.no_grad()
def _hard_rollout(model, ops, reset=False, start=None, from_step=0, upto=None):
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
        self.hn = nn.Identity()

    def ln_heads(self):
        self.hn = nn.LayerNorm(768).to(self.head.weight.device)

    def small_heads(self, logit_std=0.5, hidden_norm=200.0):
        nn.init.normal_(self.head.weight, std=logit_std / hidden_norm); nn.init.zeros_(self.head.bias)

    def zero_heads(self):
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, ops):
        dev = ops.device
        ids = torch.cat([torch.tensor(OP_TOK, device=dev)[ops], torch.full((ops.shape[0], 1), ANS_TOK, device=dev)], 1)
        h, _ = self.g(self.g.embed(ids))
        return self.head(self.hn(h[:, -1]))


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


# ---------------------------------------------------------------- amendment 67-1 cells (same objective, same interface)
class _Tables:
    def tables(self, tau=1.0):
        dev = next(self.parameters()).device
        c = torch.arange(M, device=dev).repeat(6); a = torch.arange(6, device=dev).repeat_interleave(M)
        logT = (self.step_logits(c, a) / tau).log_softmax(-1).view(6, M, M)
        logD = self.dec_logits(torch.arange(M, device=dev)).log_softmax(-1)
        return logT, logD


class MLPCell(_Tables, nn.Module):
    """one-hot(code) + one-hot(op) -> 256 -> 6 next-code logits; decoder: one-hot(code) -> 6."""
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(M + 6, 256), nn.ReLU(), nn.Linear(256, M)); self.d = nn.Linear(M, 6)

    def step_logits(self, codes, ops):
        return self.f(torch.cat([F.one_hot(codes, M).float(), F.one_hot(ops, 6).float()], -1))

    def dec_logits(self, codes):
        return self.d(F.one_hot(codes, M).float())


class TinyTF(_Tables, nn.Module):
    """2-layer d=256 causal transformer from scratch, d1 input format: [code emb, op-symbol emb, marker emb] (positions
    0-2) -> next-code head; [code emb, terminal-marker emb] -> answer head. LayerNorm before the heads."""
    def __init__(self, d=256, L=2, H=4):
        super().__init__()
        self.code = nn.Parameter(0.5 * torch.randn(M, d)); self.op = nn.Embedding(6, d); self.mark = nn.Embedding(2, d)
        self.pos = nn.Parameter(0.02 * torch.randn(3, d))
        layer = nn.TransformerEncoderLayer(d, H, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, L, enable_nested_tensor=False); self.ln = nn.LayerNorm(d)
        self.next_head = nn.Linear(d, M); self.ans_head = nn.Linear(d, 6)

    def _run(self, x):
        T = x.shape[1]
        mask = torch.full((T, T), float("-inf"), device=x.device).triu(1)
        return self.ln(self.tf(x + self.pos[:T], mask=mask, is_causal=True)[:, -1])

    def step_logits(self, codes, ops):
        x = torch.stack([self.code[codes], self.op(ops), self.mark(torch.zeros_like(ops))], 1)
        return self.next_head(self._run(x))

    def dec_logits(self, codes):
        x = torch.stack([self.code[codes], self.mark(torch.ones_like(codes))], 1)
        return self.ans_head(self._run(x))


def gpt2_random_sd(seed=0, L=12, D=768, V=50257, P=1024):
    """GPT-2 small architecture with GPT-2's own init (normal 0.02; residual projections 0.02 / sqrt(2L))."""
    g = torch.Generator().manual_seed(seed); n = lambda *s, std=0.02: torch.randn(*s, generator=g) * std
    sd = {"wte.weight": n(V, D), "wpe.weight": n(P, D, std=0.01), "ln_f.weight": torch.ones(D), "ln_f.bias": torch.zeros(D)}
    for i in range(L):
        p = f"h.{i}."
        sd.update({p + "ln_1.weight": torch.ones(D), p + "ln_1.bias": torch.zeros(D), p + "ln_2.weight": torch.ones(D),
                   p + "ln_2.bias": torch.zeros(D), p + "attn.c_attn.weight": n(D, 3 * D), p + "attn.c_attn.bias": torch.zeros(3 * D),
                   p + "attn.c_proj.weight": n(D, D, std=0.02 / math.sqrt(2 * L)), p + "attn.c_proj.bias": torch.zeros(D),
                   p + "mlp.c_fc.weight": n(D, 4 * D), p + "mlp.c_fc.bias": torch.zeros(4 * D),
                   p + "mlp.c_proj.weight": n(4 * D, D, std=0.02 / math.sqrt(2 * L)), p + "mlp.c_proj.bias": torch.zeros(D)})
    return sd


@torch.no_grad()
def certificate(model):
    """Theory's product-automaton certificate (theory/c1_diag.py), on real hard cell calls: search the (code, true S3
    state) pairs reachable from (initial code 0, identity) under all 6 ops; certified iff every pair reached after >= 1
    op decodes (decoder argmax) to its true state. Exact for every length >= 1 given stable argmaxes; ignores unreachable
    codes and the start code's own label; valid for any M (several codes may share a decoded state)."""
    dev = next(model.parameters()).device
    codes = torch.arange(M, device=dev)
    Dh = model.dec_logits(codes).argmax(-1).tolist()
    Th = model.step_logits(codes.repeat_interleave(6), torch.arange(6, device=dev).repeat(M)).argmax(-1).view(M, 6).tolist()
    seen, frontier, bad = set(), [(0, 0)], 0
    while frontier:
        c, st = frontier.pop()
        for op in range(6):
            nxt = (Th[c][op], int(COMP[op, st]))
            if nxt not in seen:
                seen.add(nxt); frontier.append(nxt); bad += int(Dh[nxt[0]] != nxt[1])
    reach = {c for c, _ in seen}
    ambiguous = sum(len({st for c2, st in seen if c2 == c}) > 1 for c in reach)
    return {"certified": bad == 0, "bad_pairs": bad, "reachable_codes": len(reach), "ambiguous_codes": ambiguous,
            "reachable_pairs": len(seen)}


@torch.no_grad()
def counter_check(model, n=4000, N=24, seed=7):
    """§67b descriptive (theory/c1_diag.py counter_check's definition, on real hard cell calls): H(code | state) and
    MI(code; length mod q | state), q = 2, 3, 4, over n random sequences observed at every step 1..N."""
    dev = next(model.parameters()).device
    rng = np.random.default_rng(seed); ops = torch.tensor(rng.integers(0, 6, (n, N)), device=dev)
    _, _, traj = hard_rollout(model, ops)
    s = np.zeros(n, int); rows = []
    for t in range(1, N + 1):
        s = COMP[ops[:, t - 1].cpu().numpy(), s]
        rows.append(np.stack([traj[t].cpu().numpy(), s, np.full(n, t)], 1))
    R = np.concatenate(rows)

    def H(x):
        _, cnt = np.unique(x, axis=0, return_counts=True); p = cnt / cnt.sum(); return float(-(p * np.log2(p)).sum())
    out = {"H_code_given_state": H(R[:, :2]) - H(R[:, 1:2])}
    for q in (2, 3, 4):
        lm = (R[:, 2] % q)[:, None]
        out[f"MI_code_lenmod{q}_given_state"] = (H(R[:, :2]) + H(np.hstack([lm, R[:, 1:2]])) - H(np.hstack([R[:, :2], lm]))
                                                 - H(R[:, 1:2]))
    return out


@torch.no_grad()
def certificate_reach(model):
    """The set of codes reachable from the initial code 0 under all ops (hard argmax), excluding nothing."""
    dev = next(model.parameters()).device
    codes = torch.arange(M, device=dev)
    Th = model.step_logits(codes.repeat_interleave(6), torch.arange(6, device=dev).repeat(M)).argmax(-1).view(M, 6).tolist()
    seen, frontier = {0}, [0]
    while frontier:
        c = frontier.pop()
        for op in range(6):
            if Th[c][op] not in seen:
                seen.add(Th[c][op]); frontier.append(Th[c][op])
    return sorted(seen)


@torch.no_grad()
def layer_code_variance(model):
    """§72 (PT): per GPT-2 layer, at the code position of the 6M transition calls: between-code variance (of the code
    means over ops) vs within-code variance (across ops), summed over coordinates."""
    g = model.g; dev = model.code.device
    c = torch.arange(M, device=dev).repeat(6); a = torch.arange(6, device=dev).repeat_interleave(M)
    x = torch.stack([model.code[c], g.embed(torch.tensor(OP_TOK, device=dev)[a]), g.embed(torch.full_like(a, STEP_TOK))], 1)
    B, T, D = x.shape; hd = D // g.H; h = x + g.w("wpe.weight")[:T][None]; out = []
    for i in range(g.L):
        p = f"h.{i}."
        z = F.layer_norm(h, (D,), g.w(p + "ln_1.weight"), g.w(p + "ln_1.bias"))
        q, k, v = (z @ g.w(p + "attn.c_attn.weight") + g.w(p + "attn.c_attn.bias")).split(D, -1)
        q, k, v = (t.view(B, T, g.H, hd).transpose(1, 2) for t in (q, k, v))
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        h = h + o.transpose(1, 2).reshape(B, T, D) @ g.w(p + "attn.c_proj.weight") + g.w(p + "attn.c_proj.bias")
        z = F.layer_norm(h, (D,), g.w(p + "ln_2.weight"), g.w(p + "ln_2.bias"))
        h = h + F.gelu(z @ g.w(p + "mlp.c_fc.weight") + g.w(p + "mlp.c_fc.bias"), approximate="tanh") @ g.w(p + "mlp.c_proj.weight") \
            + g.w(p + "mlp.c_proj.bias")
        v0 = h[:, 0].float().view(6, M, D)                            # (op, code, D) at the code position
        between = float(v0.mean(0).var(0, unbiased=False).sum()); within = float(v0.var(0, unbiased=False).mean(0).sum())
        out.append({"layer": i + 1, "between_code": between, "within_code": within, "ratio": between / max(within, 1e-12)})
    return out
