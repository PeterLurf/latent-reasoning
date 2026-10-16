"""VERSION-BANK (VB) + READY-RECODE on nested arithmetic (docs/THEORY_VERSIONBANK_DRAFT_2026-10-15.md + docs/research_passuse_2026-10-15/; Stage 0).
A NEW module: imports boxes.nested170 / boxes.deltamem read-only and changes no existing interface (§205 s1 / s2 import those).
Part 1 (this commit): the per-slot recode targets E_t and their re-render gate.
  Slots: x = [EOT] + render(root) + [" ="]; class per slot (20): VOID, EOT, " =", " (", " )", 4 ops, 11 values.
  E_t = the slot-aligned expression after t Feng reductions (nested170's leftmost-ready handle): the handle's span (its two number operands, its operator
  and its own brackets - only the reduced node's brackets can vanish, an ancestor's depend on the ancestor's operator) goes VOID and the span's LAST
  slot takes the value; E_t = E_n for t > n (HOLD).
  Gate (draft §9 Stage 0 item 5): dropping VOIDs from E_t re-renders exactly cot_steps(root)[t] inside [EOT .. " ="], for every t, on 100k items."""
import math

import numpy as np
import torch
import torch.nn.functional as F

import boxes.nested170 as N

CLS = ["VOID", "EOT", " =", " (", " )"] + [" " + o for o in N.OPS] + [" %d" % v for v in range(N.P)]
CID = {c: i for i, c in enumerate(CLS)}; VOID = 0; NCLS = len(CLS)
assert NCLS == 20


def _render_owned(nd, parent=None, right=False):
    """nested170.render with ownership: -> [(symbol, node)] (brackets and the operator owned by the operator node, a number by its leaf)."""
    if nd.op is None:
        return [(" %d" % nd.v, nd)]
    s = _render_owned(nd.l, nd.op, False) + [(" " + nd.op, nd)] + _render_owned(nd.r, nd.op, True)
    if parent is not None and (N.PREC[nd.op] < N.PREC[parent] or (right and N.PREC[nd.op] == N.PREC[parent])):
        s = [(" (", nd)] + s + [(" )", nd)]
    return s


def recode_targets(root, t_max=None):
    """-> list E_0 .. E_{t_max} (t_max default n), each a list of class ids over the slots of [EOT] + render(root) + [" ="]; HOLD past n."""
    nd = N._copy(root); slots = [CID["EOT"]] + [CID[s] for s in N.render(nd)] + [CID[" ="]]; E = [list(slots)]
    while nd.op is not None:
        live = [j for j in range(1, len(slots) - 1) if slots[j] != VOID]; own = _render_owned(nd); assert len(own) == len(live), "VB: live slots != render"
        path = N._leftmost_ready(nd); h = nd
        for p_ in path:
            h = getattr(h, p_)
        span = [live[k] for k, (_, o) in enumerate(own) if o is h or o is h.l or o is h.r]
        assert span == list(range(span[0], span[-1] + 1)) or all(slots[j] == VOID for j in range(span[0], span[-1] + 1) if j not in span), "VB: the handle span is not contiguous over live slots"
        v = N.apply(h.op, h.l.v, h.r.v); h.v = v; h.op = h.l = h.r = None
        for j in span:
            slots[j] = VOID
        slots[span[-1]] = CID[" %d" % v]; E.append(list(slots))
    t_max = len(E) - 1 if t_max is None else t_max
    return [E[min(t, len(E) - 1)] for t in range(t_max + 1)]


def rerender(E_t):
    return [CLS[c] for c in E_t[1:-1] if c != VOID]


def gate_rerender(n_items=100_000, seed=20900, n_max=16):
    """every E_t (VOIDs dropped) == cot_steps(root)[t]; the frame slots stay; the slot count is constant; E_n's single live slot is the answer."""
    rng = np.random.default_rng(seed); bad = 0; n_steps = 0
    for _ in range(n_items):
        root, y = N.generate(rng, int(rng.integers(1, n_max + 1))); E = recode_targets(root); st = N.cot_steps(root); n_steps += len(E) - 1
        ok = len(E) == len(st) and all(rerender(e) == s for e, s in zip(E, st)) and all(len(e) == len(E[0]) and e[0] == CID["EOT"] and e[-1] == CID[" ="] for e in E)
        ok = ok and rerender(E[-1]) == [" %d" % y]
        bad += not ok
    return {"items": n_items, "steps": n_steps, "failures": bad, "pass": bad == 0}


# ------------------------------------------------------------------ Part 2: the float-mask bank block (deltamem._block is causal-only, REDTEAM F9c)
def bank_kv(g, i, V):
    """block i's own ln_1 + c_attn on stored versions V (B, T, D) -> (k, v) each (B, H, T, hd); WITH gradient (the bank cache must stay live, F9a)."""
    B, T, D = V.shape; hd = D // g.H; pre = f"h.{i}."
    a = F.layer_norm(V, (D,), g.w(pre + "ln_1.weight"), g.w(pre + "ln_1.bias"))
    _, k, v = (a @ g.w(pre + "attn.c_attn.weight") + g.w(pre + "attn.c_attn.bias")).split(D, -1)
    return k.view(B, T, g.H, hd).transpose(1, 2), v.view(B, T, g.H, hd).transpose(1, 2)


def block_bank(g, i, h, bk, bv, bbias):
    """deltamem._block with a bank: the current version's queries attend over [bank keys (bk, bv: (B, H, Tb, hd)) | the current version, causal];
    bbias (B, H, T, Tb) float: the bank logit bias (beta + gamma * age), -inf on masked entries (padding, versions >= t). bk None = no bank.
    Bank-free when every bank entry is -inf (G1: equal to deltamem._block within fp64 rounding)."""
    B, T, D = h.shape; hd = D // g.H; pre = f"h.{i}."
    a = F.layer_norm(h, (D,), g.w(pre + "ln_1.weight"), g.w(pre + "ln_1.bias"))
    q, k, v = (a @ g.w(pre + "attn.c_attn.weight") + g.w(pre + "attn.c_attn.bias")).split(D, -1)
    q, k, v = (t_.view(B, T, g.H, hd).transpose(1, 2) for t_ in (q, k, v))
    causal = torch.full((T, T), float("-inf"), device=h.device, dtype=q.dtype).triu(1)
    if bk is None:
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=causal)
    else:
        mask = torch.cat([bbias.to(q.dtype), causal.expand(B, g.H, T, T)], -1)
        o = F.scaled_dot_product_attention(q, torch.cat([bk.to(q.dtype), k], 2), torch.cat([bv.to(q.dtype), v], 2), attn_mask=mask)
    h = h + o.transpose(1, 2).reshape(B, T, D) @ g.w(pre + "attn.c_proj.weight") + g.w(pre + "attn.c_proj.bias")
    a = F.layer_norm(h, (D,), g.w(pre + "ln_2.weight"), g.w(pre + "ln_2.bias"))
    m = F.gelu(a @ g.w(pre + "mlp.c_fc.weight") + g.w(pre + "mlp.c_fc.bias"), approximate="tanh")
    return h + m @ g.w(pre + "mlp.c_proj.weight") + g.w(pre + "mlp.c_proj.bias")


def gate_g1(dev="cpu", B=4, T=13, Tb=26, seed=20901):
    """G1 (draft §9 item 2): fp64. (a) block_bank with every bank entry at -inf == deltamem._block (<= 1e-9 rel); (b) no bank == deltamem._block;
    (c) a finite bias moves the output (the bank is live); (d) d out / d bank K, V != 0 under a finite bias (F9a's dead-bank check at block level)."""
    import boxes.deltamem as DM
    from boxes import s5
    g = s5.load_gpt2(dev).double(); torch.manual_seed(seed); D = 768; out = {}
    h = torch.randn(B, T, D, dtype=torch.float64, device=dev); Vb = torch.randn(B, Tb, D, dtype=torch.float64, device=dev, requires_grad=True)
    ref = DM._block(g, 6, h, D // g.H); bk, bv = bank_kv(g, 6, Vb)
    ninf = torch.full((B, g.H, T, Tb), float("-inf"), dtype=torch.float64, device=dev)
    o_a = block_bank(g, 6, h, bk, bv, ninf); o_b = block_bank(g, 6, h, None, None, None)
    rel = lambda x, y: float((x - y).abs().max() / y.abs().max())  # noqa: E731
    out["masked_bank_eq_causal_rel"] = rel(o_a, ref); out["no_bank_eq_causal_rel"] = rel(o_b, ref)
    o_c = block_bank(g, 6, h, bk, bv, torch.full_like(ninf, -2.0)); out["finite_bias_moves_rel"] = rel(o_c, ref)
    o_c.sum().backward(); out["grad_bank_V_norm"] = float(Vb.grad.norm())
    out["pass"] = out["masked_bank_eq_causal_rel"] <= 1e-9 and out["no_bank_eq_causal_rel"] <= 1e-9 and out["finite_bias_moves_rel"] > 1e-6 and out["grad_bank_V_norm"] > 0
    return out


# ------------------------------------------------------------------ Part 3: the VBankN model (draft §3; note 211-1 pins 1-5)
CORE = (6, 7, 8, 9); CODA = (10, 11); LAYERS = CORE + CODA; KMAX = 12; NEG = float("-inf")   # bank params row li = LAYERS.index(i)


def alibi_init(H=12, n_alibi=6):
    """pin 2: heads 0..5 get the ALiBi slope m_h = 2^(-8(h+1)/6) as the bias -m_h * age (gamma init -m_h); heads 6..11 start at 0."""
    return torch.tensor([-(2.0 ** (-8.0 * (h + 1) / n_alibi)) if h < n_alibi else 0.0 for h in range(H)])


def bias_row(b, b0, gm, t, nver, pad, off_v0=False, off=False):
    """the bank logit bias over stored versions 0..nver-1, read at pass (or coda) index t: beta0 (version 0, no age term) or beta + gamma * (t - 1 - s);
    -inf on padded slots (pad (B, T) bool, True = real), on version 0 when off_v0, everywhere when off. b, b0, gm: (H,) -> (B, H, 1, nver * T)."""
    B, T = pad.shape; s = torch.arange(nver, device=pad.device); age = (t - 1 - s).to(b.dtype)
    row = torch.where(s[None, :] == 0, b0[:, None], b[:, None] + gm[:, None] * age[None, :])            # (H, nver)
    row = row[:, :, None].expand(-1, nver, T).reshape(len(b), nver * T)
    valid = pad[:, None, :].expand(B, nver, T).reshape(B, nver * T)
    if off_v0:
        valid = valid & (s.repeat_interleave(T) != 0)[None]
    if off:
        valid = torch.zeros_like(valid)
    return torch.where(valid[:, None, None, :], row[None, :, None, :], torch.full((), NEG, dtype=row.dtype, device=row.device))


class VBankN(torch.nn.Module):
    """VERSION-BANK on GPT-2 small (draft §3): prelude blocks 0-5 -> V_0; core blocks 6-9 looped (pass t's input U_t: CARRY = N(V_{t-1}), or the
    PURE-ANNEAL gather over the bank); each pass appends V_t to the bank; every core layer and the coda (blocks 10-11 + ln_f + LM head) attend over
    [bank versions 0..t-1 | the current version, causal] with the bias beta0 (V_0) / beta + gamma * age. R = LN + Linear(768 -> 20), the recode head.
    Pins (note 211-1): 1 gather = 12 heads x 64, each over its own 64-d slice of N(V); own W_q / W_k / LN copied from block 6 (c_attn q / k, ln_1);
    the content-free query wpe[j] @ W_q + b_q; gamma_q per head init 0; c = the c_anneal buffer (set by the trainer). N(V_s)[i] = gain * V_s[i] *
    rms(V_0[i]) / rms(V_s[i]). 2 ALiBi gamma, no age term on V_0. 4 the coda reads V_0..V_{t-1} + causal V_t with its own beta / beta0 / gamma rows.
    5 gain: a 768-vector, init 1. Bank K / V are computed once per (version, layer) WITH gradient and concatenated per pass (Experiments -> Theory
    23:2x: an in-place preallocated bank fails autograd's saved-tensor version check)."""
    def __init__(self, g, mode="pa"):
        super().__init__(); assert mode in ("pa", "carry"), mode
        self.g = g; self.mode = mode; D, H = g.D, g.H; self.D, self.H, self.hd = D, H, D // H; dev = g.w("wte.weight").device
        self.gain = torch.nn.Parameter(torch.ones(D, device=dev))
        self.beta = torch.nn.Parameter(torch.full((len(LAYERS), H), -5.0, device=dev))     # calibrated at step 0 (pin 3); -5 = the pre-calibration value
        self.beta0 = torch.nn.Parameter(torch.full((len(LAYERS), H), -5.0, device=dev))
        self.gam = torch.nn.Parameter(alibi_init(H).repeat(len(LAYERS), 1).to(dev))
        self.rec_ln = torch.nn.LayerNorm(D).to(dev); self.rec = torch.nn.Linear(D, NCLS).to(dev)
        if mode == "pa":
            W, b = g.w("h.6.attn.c_attn.weight").detach(), g.w("h.6.attn.c_attn.bias").detach()
            self.gq_w = torch.nn.Parameter(W[:, :D].clone()); self.gq_b = torch.nn.Parameter(b[:D].clone())
            self.gk_w = torch.nn.Parameter(W[:, D:2 * D].clone()); self.gk_b = torch.nn.Parameter(b[D:2 * D].clone())
            self.gln_w = torch.nn.Parameter(g.w("h.6.ln_1.weight").detach().clone()); self.gln_b = torch.nn.Parameter(g.w("h.6.ln_1.bias").detach().clone())
            self.gamma_q = torch.nn.Parameter(torch.zeros(H, device=dev))
        self.register_buffer("c_anneal", torch.tensor(1e4, device=dev))
        self.bank_off = False                                               # True: every bank logit -inf (G1 / the calibration rollout = exact CARRY)
        self.cap = None                                                     # calibration: a dict -> {("core" | "coda", layer, t): the layer's input h}
        self.gather_log = None                                              # a list -> per pass (t, (B,) mean self-mass on (t-1, j) over heads / real slots)

    def nrm(self, V, rms0):
        return self.gain * (V * (rms0 / V.pow(2).mean(-1, keepdim=True).sqrt()))

    def gkey(self, V):
        B, T, D = V.shape; a = F.layer_norm(V, (D,), self.gln_w, self.gln_b)
        return (a @ self.gk_w + self.gk_b).view(B, T, self.H, self.hd).transpose(1, 2)

    def gather(self, t, gk, NV, pad, off_v0=False):
        """PURE-ANNEAL (draft §3): U[j] = sum_{s < t, i} softmax(l_{j,(s,i)} - c * 1[(s, i) != (t-1, j)]) N(V_s)[i], per head on its 64-d slice; fp32 (fp64 in
        the gates), autocast off: at c = 1e4 the weights are exactly one-hot, so U == N(V_{t-1}) bitwise (gate_carry_bitwise)."""
        B, T = pad.shape; H, hd = self.H, self.hd; dev = pad.device; dt = torch.float64 if NV[0].dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=dev.type, enabled=False):
            q = (self.g.w("wpe.weight")[:T].to(dt) @ self.gq_w.to(dt) + self.gq_b.to(dt)).view(T, H, hd).transpose(0, 1)   # (H, T, hd), content-free
            K_ = torch.cat(gk[:t], 2).to(dt); lg = torch.einsum("htd,bhsd->bhts", q, K_) / math.sqrt(hd)                    # (B, H, T, tT)
            s_idx = torch.arange(t, device=dev).repeat_interleave(T); i_idx = torch.arange(T, device=dev).repeat(t)
            lg = lg + self.gamma_q.to(dt)[None, :, None, None] * (t - 1 - s_idx).to(dt)[None, None, None, :]
            own = (s_idx[None, :] == t - 1) & (i_idx[None, :] == torch.arange(T, device=dev)[:, None])                   # (T, tT): the (t-1, j) entry
            lg = lg - self.c_anneal.to(dt) * (~own).to(dt)[None, None]
            valid = pad[:, None, :].expand(B, t, T).reshape(B, t * T)
            if off_v0:
                valid = valid & (s_idx != 0)[None]
            w = torch.softmax(lg.masked_fill(~valid[:, None, None, :], NEG), -1)
            U = (w @ torch.cat(NV[:t], 1).to(dt).view(B, t * T, H, hd).transpose(1, 2)).transpose(1, 2).reshape(B, T, H * hd)
            if self.gather_log is not None:
                sm = w[:, :, torch.arange(T, device=dev), (t - 1) * T + torch.arange(T, device=dev)].detach()                # (B, H, T)
                self.gather_log.append((t, (sm.mean(1) * pad).sum(1) / pad.sum(1)))
        return U.to(NV[0].dtype)

    def coda(self, t, Vt, kvs, pad, off_v0=False, src=None, bp=None):
        """blocks 10-11 over [bank versions 0..t-1 (kvs {layer: [(k, v)] * t}) | V_t causal] + ln_f -> (b, T, D). src: a weight source (.w / .H; default
        the trunk) and bp: (beta, beta0, gam) rows for the 2 coda layers (default this model's) - the PROBE(t) coda copy passes its own."""
        src = src or self.g; D = self.D; h = Vt; bp = bp or (self.beta[4:], self.beta0[4:], self.gam[4:])
        for r, i in enumerate(CODA):
            if t == 0:
                h = block_bank(src, i, h, None, None, None)
            else:
                bk = torch.cat([k for k, _ in kvs[i][:t]], 2); bv = torch.cat([v for _, v in kvs[i][:t]], 2)
                bb = bias_row(bp[0][r], bp[1][r], bp[2][r], t, t, pad, off_v0, self.bank_off).expand(-1, -1, h.shape[1], -1)
                if self.cap is not None:
                    self.cap[("coda", i, t)] = h.detach()
                h = block_bank(src, i, h, bk, bv, bb)
        return F.layer_norm(h, (D,), src.w("ln_f.weight"), src.w("ln_f.bias"))

    def rollout(self, x, pad, K, coda_rows=None, edit=None, v0_off_after=None):
        """x (B, T) token ids [EOT, expr, ' ='] right-padded; pad (B, T) real-slot mask; K passes.
        coda_rows {t: LongTensor rows}: the coda after pass t (t = 0: on V_0) on those rows -> out["coda"][t] = (rows, h (b, T, D)).
        edit(t, V_t) -> the stored entry V_t' (bank, and the carry), or (bank_t, carry_t) under CARRY (VALUE-SWAP iii); None = V_t (interventions, MCT).
        v0_off_after: V_0's entry masked (core, gather and coda) for passes > it and the coda after t > it (VALUE-SWAP iv).
        -> {"V": computed V_0..V_K, "S": stored entries, "coda": {...}, "kv": the bank K / V}."""
        g = self.g; B, T = x.shape; V0 = DM_().gpt_lower(g, g.embed(x), CORE[0]); rms0 = V0.pow(2).mean(-1, keepdim=True).sqrt()
        Vc, St, kv, gk, NV, out = [V0], [V0], {i: [] for i in LAYERS}, [], [], {}

        def push(Vs):
            for i in LAYERS:
                kv[i].append(bank_kv(g, i, Vs))
            NV.append(self.nrm(Vs, rms0))
            if self.mode == "pa":
                gk.append(self.gkey(Vs))

        def coda_at(t, Vt):
            if coda_rows and t in coda_rows:
                r_ = coda_rows[t]; off_ = v0_off_after is not None and t > v0_off_after
                out[t] = (r_, self.coda(t, Vt[r_], {i: [(k[r_], v[r_]) for k, v in kv[i]] for i in CODA}, pad[r_], off_))
        coda_at(0, V0); push(V0); carry = V0
        for t in range(1, K + 1):
            off_ = v0_off_after is not None and t > v0_off_after
            h = self.nrm(carry, rms0) if self.mode == "carry" else self.gather(t, gk, NV, pad, off_)
            for li, i in enumerate(CORE):
                bk = torch.cat([k for k, _ in kv[i]], 2); bv = torch.cat([v for _, v in kv[i]], 2)
                bb = bias_row(self.beta[li], self.beta0[li], self.gam[li], t, t, pad, off_, self.bank_off).expand(-1, -1, T, -1)
                if self.cap is not None:
                    self.cap[("core", i, t)] = h.detach()
                h = block_bank(g, i, h, bk, bv, bb)
            Vc.append(h); coda_at(t, h)
            st = edit(t, h) if edit is not None else None
            st = h if st is None else st
            if isinstance(st, tuple):
                assert self.mode == "carry", "a separate carry exists under CARRY only"
                st, carry = st
            else:
                carry = st
            St.append(st); push(st)
        return {"V": Vc, "S": St, "coda": out, "kv": kv}

    def answer_logits(self, h, lens):
        """(b, T, D) coda output -> (b, V) full-vocabulary logits at each item's ' =' slot (len - 1)."""
        lg = self.g.logits(h[torch.arange(h.shape[0], device=h.device), lens - 1]); return lg if lg.dtype == torch.float64 else lg.float()   # fp64 kept (gates)

    def rec_logits(self, V):
        lg = self.rec(self.rec_ln(V)); return lg if lg.dtype == torch.float64 else lg.float()

    @torch.no_grad()
    def mct(self, V, E, valid, mu=1.0, max_iter=3):
        """R3 (draft §4; pin 7): the minimum-norm edit in R's row space, in LN coordinates, so that R decodes E (B, T) at every valid slot with margin
        mu nats over the runner-up: d_y = max(0, max_{c != y} l_c + mu - l_y), dz = pinv(W) d, mapped back to the residual as sd * dz / ln_w (mean
        removed); repeated (<= max_iter) while a slot misses the margin. -> (V', {"steps": per-slot step count, "rel": [||dV|| / ||V|| per step]})."""
        W = self.rec.weight.float(); b = self.rec.bias.float(); Wp = torch.linalg.pinv(W); lw = self.rec_ln.weight.float(); lb = self.rec_ln.bias.float()
        lw_safe = torch.where(lw.abs() < 1e-3, torch.full_like(lw, 1e-3) * torch.sign(lw + 1e-12), lw)
        V2 = V.float().clone(); E_ = E.clamp_min(0)[..., None]; steps = torch.zeros(V.shape[:2], dtype=torch.long, device=V.device); rel = []
        for _ in range(max_iter):
            mu_ = V2.mean(-1, keepdim=True); sd = (V2.var(-1, unbiased=False, keepdim=True) + self.rec_ln.eps).sqrt()
            lg = ((V2 - mu_) / sd * lw + lb) @ W.T + b; ly = lg.gather(-1, E_); lo = lg.scatter(-1, E_, NEG).max(-1, keepdim=True).values
            need = (lo + mu - ly).clamp_min(0) * valid[..., None]
            if not bool((need > 0).any()):
                break
            dv = (torch.zeros_like(lg).scatter(-1, E_, need) @ Wp.T) / lw_safe * sd; dv = dv - dv.mean(-1, keepdim=True)
            rel.append(float(dv.norm() / V2.norm())); V2 = V2 + dv; steps += (need[..., 0] > 0).long()
        return V2.to(V.dtype), {"steps": steps, "rel": rel}

    @torch.no_grad()
    def calibrate(self, x, pad, lens, K=KMAX, mu=0.02, q=0.99):
        """pin 3: a CARRY rollout with the bank masked (exact CARRY) to V_0..V_K, every layer's input captured; per layer (core 6-9 and coda 10-11) and
        head, the bank mass a query would give is sigmoid(beta + log A - log C) (A = the bank entries' exp-score sum, C = the rest), so beta0 is set
        first (bank = V_0; every t >= 1) and then beta (V_1..V_{t-1}, beta0 fixed; t >= 2) to logit(mu) - quantile_q(log A - log C) over (items, real
        query slots, t): the q-quantile mass is then mu (re-measured after setting). -> a report dict (the ratio test: ratio_test())."""
        g = self.g; D, H, hd = self.D, self.H, self.hd; mode0, off0 = self.mode, self.bank_off; self.bank_off = True; self.cap = {}
        try:
            r = self.rollout(x, pad, K, coda_rows={t: torch.arange(x.shape[0], device=x.device) for t in range(1, K + 1)})
            cap = self.cap
        finally:
            self.cap = None; self.bank_off = off0
        rep = {}; lmu = math.log(mu / (1 - mu)) - 0.01                                  # 0.01 nats inside the target (m99 ~ 0.0198)
        for li, i in enumerate(LAYERS):
            kind = "core" if i in CORE else "coda"; parts = {"v0": [], "rest": [], "cur": [], "t": []}
            for t in range(1, K + 1):
                h = cap[(kind, i, t)]; B, T, _ = h.shape; pre = f"h.{i}."
                a = F.layer_norm(h, (D,), g.w(pre + "ln_1.weight"), g.w(pre + "ln_1.bias"))
                qh, kh, _ = (a @ g.w(pre + "attn.c_attn.weight") + g.w(pre + "attn.c_attn.bias")).split(D, -1)
                qh, kh = (u.view(B, T, H, hd).transpose(1, 2).float() for u in (qh, kh))
                sc_cur = (qh @ kh.transpose(-1, -2) / math.sqrt(hd)).masked_fill(torch.ones(T, T, dtype=torch.bool, device=x.device).triu(1), NEG)
                kb = torch.cat([k for k, _ in r["kv"][i][:t]], 2).float(); sc_b = qh @ kb.transpose(-1, -2) / math.sqrt(hd)     # (B, H, T, tT)
                sc_b = sc_b.masked_fill(~pad.repeat(1, t)[:, None, None, :], NEG)
                age = (t - 1 - torch.arange(t, device=x.device).repeat_interleave(T)).float()
                parts["v0"].append(torch.logsumexp(sc_b[..., :T], -1)); parts["cur"].append(torch.logsumexp(sc_cur, -1)); parts["t"].append(t)
                parts["rest"].append(torch.logsumexp(sc_b[..., T:] + self.gam[li].float()[None, :, None, None] * age[T:][None, None, None, :], -1) if t >= 2 else None)
            sel = pad[:, None, :].expand(-1, H, -1)                       # real query slots
            x0 = torch.stack([(a0 - c0) for a0, c0 in zip(parts["v0"], parts["cur"])], 0)                    # (K, B, H, T): log A - log C for beta0
            b0 = torch.stack([lmu - torch.quantile(x0[:, :, h_, :][:, sel[:, h_, :]].flatten(), q) for h_ in range(H)])
            self.beta0.data[li] = b0.to(self.beta0.dtype)
            m0 = torch.sigmoid(x0 + b0[None, None, :, None]); m0q = [float(torch.quantile(m0[:, :, h_, :][:, sel[:, h_, :]].flatten(), q)) for h_ in range(H)]
            x1 = torch.stack([ar - torch.logaddexp(c0, a0 + b0[None, :, None]) for ar, a0, c0, t in zip(parts["rest"], parts["v0"], parts["cur"], parts["t"]) if t >= 2], 0)
            b1 = torch.stack([lmu - torch.quantile(x1[:, :, h_, :][:, sel[:, h_, :]].flatten(), q) for h_ in range(H)])
            self.beta.data[li] = b1.to(self.beta.dtype)
            m1 = torch.sigmoid(x1 + b1[None, None, :, None]); m1q = [float(torch.quantile(m1[:, :, h_, :][:, sel[:, h_, :]].flatten(), q)) for h_ in range(H)]
            rep[str(i)] = {"beta0": [round(float(v), 4) for v in b0], "beta": [round(float(v), 4) for v in b1], "m99_v0": m0q, "m99_rest": m1q}
        rep["max_m99"] = max(max(v["m99_v0"] + v["m99_rest"]) for k, v in rep.items() if k != "max_m99"); rep["mu"] = mu; rep["q"] = q
        return rep

    @torch.no_grad()
    def ratio_test(self, x, pad, lens, K=KMAX, shift=5.0):
        """pin 3: dz(beta) / dz(beta - 5) in [0.5, 2] e^5, dz = the mean L2 change of the coda's full-vocabulary answer logits after K passes, bank on vs
        masked (every beta and beta0 shifted together)."""
        rows = {K: torch.arange(x.shape[0], device=x.device)}

        def z():
            r = self.rollout(x, pad, K, coda_rows=rows); return self.answer_logits(r["coda"][K][1], lens)
        off0 = self.bank_off; self.bank_off = True; z0 = z(); self.bank_off = off0
        z1 = z(); self.beta.data -= shift; self.beta0.data -= shift; z2 = z(); self.beta.data += shift; self.beta0.data += shift
        d1, d2 = float((z1 - z0).norm(dim=-1).mean()), float((z2 - z0).norm(dim=-1).mean())
        rr = d1 / d2 / math.exp(shift) if d2 > 0 else None
        return {"dz_beta": d1, "dz_beta_minus5": d2, "z_masked_norm": float(z0.norm(dim=-1).mean()), "ratio_over_e5": rr, "pass": rr is not None and 0.5 <= rr <= 2.0}


def DM_():
    import boxes.deltamem as DM
    return DM


def batch_x(items, dev):
    """[EOT] + expr + [' ='] right-padded with EOT -> (x (B, T), pad (B, T) bool, lens (B,))."""
    seqs = [[N.EOT] + N.ids(it["syms"]) + [N.TID[" ="]] for it in items]; ids_, m_ = N.pad(seqs)
    return torch.tensor(ids_, device=dev), torch.tensor(m_, device=dev), torch.tensor([len(q_) for q_ in seqs], device=dev)


if __name__ == "__main__":
    import json
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "g1":
        print(json.dumps(gate_g1()))
    else:
        print(json.dumps(gate_rerender()))
