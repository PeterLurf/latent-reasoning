"""VERSION-BANK (VB) + READY-RECODE on nested arithmetic (docs/THEORY_VERSIONBANK_DRAFT_2026-10-15.md + docs/research_passuse_2026-10-15/; Stage 0).
A NEW module: imports boxes.nested170 / boxes.deltamem read-only and changes no existing interface (§205 s1 / s2 import those).
Part 1 (this commit): the per-slot recode targets E_t and their re-render gate.
  Slots: x = [EOT] + render(root) + [" ="]; class per slot (20): VOID, EOT, " =", " (", " )", 4 ops, 11 values.
  E_t = the slot-aligned expression after t Feng reductions (nested170's leftmost-ready handle): the handle's span (its two number operands, its operator
  and its own brackets - only the reduced node's brackets can vanish, an ancestor's depend on the ancestor's operator) goes VOID and the span's LAST
  slot takes the value; E_t = E_n for t > n (HOLD).
  Gate (draft §9 Stage 0 item 5): dropping VOIDs from E_t re-renders exactly cot_steps(root)[t] inside [EOT .. " ="], for every t, on 100k items."""
import hashlib
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


def block_bank(g, i, h, bk, bv, bbias, kmask=None):
    """deltamem._block with a bank: the current version's queries attend over [bank keys (bk, bv: (B, H, Tb, hd)) | the current version, causal];
    bbias (B, H, T, Tb) float: the bank logit bias (beta + gamma * age), -inf on masked entries (padding, versions >= t). bk None = no bank.
    Bank-free when every bank entry is -inf (G1: equal to deltamem._block within fp64 rounding).
    kmask (B, T) bool, eval-only (note 212-15 MASKVOID / GAPPED): the current version's keys at True slots are removed (-inf); None = unchanged."""
    B, T, D = h.shape; hd = D // g.H; pre = f"h.{i}."
    a = F.layer_norm(h, (D,), g.w(pre + "ln_1.weight"), g.w(pre + "ln_1.bias"))
    q, k, v = (a @ g.w(pre + "attn.c_attn.weight") + g.w(pre + "attn.c_attn.bias")).split(D, -1)
    q, k, v = (t_.view(B, T, g.H, hd).transpose(1, 2) for t_ in (q, k, v))
    causal = torch.full((T, T), float("-inf"), device=h.device, dtype=q.dtype).triu(1)
    if kmask is not None:                                                   # eval-only (212-15): per-item removed key slots in the current version
        causal = causal.expand(B, g.H, T, T).masked_fill(kmask[:, None, None, :], float("-inf"))
    if bk is None:
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=causal)
    else:
        mask = torch.cat([bbias.to(q.dtype), causal.expand(B, g.H, T, T)], -1)   # (expand is a no-op on the kmask (B, H, T, T) form)
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


def bias_row(b, b0, gm, t, nver, pad, off_v0=False, off=False, keep=None, floor=None):
    """the bank logit bias over stored versions 0..nver-1, read at pass (or coda) index t: beta0 (version 0, no age term) or beta + gamma * (t - 1 - s);
    -inf on padded slots (pad (B, T) bool, True = real), on version 0 when off_v0, everywhere when off. b, b0, gm: (H,) -> (B, H, 1, nver * T)."""
    B, T = pad.shape; s = torch.arange(nver, device=pad.device); age = (t - 1 - s).to(b.dtype)
    agev = gm[:, None] * age[None, :]
    if floor is not None:                                                   # eval-only (note 212-3): age term -> min(gamma * age, -floor * age) on versions 1..t-2
        ap_ = (s >= 1) & (s <= t - 2); agev = torch.where(ap_[None, :], torch.minimum(agev, -floor * age[None, :]), agev)
    row = torch.where(s[None, :] == 0, b0[:, None], b[:, None] + agev)                                      # (H, nver)
    row = row[:, :, None].expand(-1, nver, T).reshape(len(b), nver * T)
    valid = pad[:, None, :].expand(B, nver, T).reshape(B, nver * T)
    if off_v0:
        valid = valid & (s.repeat_interleave(T) != 0)[None]
    if keep is not None:                                                    # eval-only (note 211-14): a (nver,) bool of versions kept readable
        valid = valid & keep.to(pad.device).repeat_interleave(T)[None]
    if off:
        valid = torch.zeros_like(valid)
    return torch.where(valid[:, None, None, :], row[None, :, None, :], torch.full((), NEG, dtype=row.dtype, device=row.device))


class VBankN(torch.nn.Module):
    """VERSION-BANK on GPT-2 small (draft §3): prelude blocks 0-5 -> V_0; core blocks 6-9 looped (pass t's input U_t: CARRY = N(V_{t-1}), or the
    PURE-ANNEAL gather over the bank); each pass appends V_t to the bank; every core layer and the coda (blocks 10-11 + ln_f + LM head) attend over
    [bank versions 0..t-1 | the current version, causal] with the bias beta0 (V_0) / beta + gamma * age. R = LN + Linear(768 -> 20), the recode head.
    Pins (note 211-1): 1 gather = 12 heads x 64, each over its own 64-d slice of N(V); own W_q / W_k / LN copied from block 6 (c_attn q / k, ln_1);
    the content-free query wpe[j] @ W_q + b_q; gamma_q per head init 0; c = the c_anneal buffer (set by the trainer). N(V_s)[i] = gain * V_s[i] *
    rms(V_0[i]) / rms(V_s[i]). 2 ALiBi gamma, no age term on V_0. 4 the coda reads V_1..V_{t-1} + causal V_t (V_0 excluded from its bank, note 211-2)
    with its own beta / gamma rows (its beta0 rows are unused).
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
            self.gamma_q = torch.nn.Parameter(alibi_init(H, H).to(dev))         # note 211-1 pin 1 (211-2): forward-decay ALiBi on all 12 heads, -2^(-8(h+1)/12)
        self.register_buffer("c_anneal", torch.tensor(1e4, device=dev))
        self.bank_off = False                                               # True: every bank logit -inf (G1 / the calibration rollout = exact CARRY)
        self.cap = None                                                     # calibration: a dict -> {("core" | "coda", layer, t): the layer's input h}
        self.gather_log = None                                              # a list -> per pass (t, (B,) mean self-mass on (t-1, j) over heads / real slots)
        self.ver_keep = None                                                # eval-only (note 211-14): f(t, nver) -> (nver,) bool of bank versions kept (None = all)
        self.age_floor = None                                               # eval-only (note 212-3): slope floor mu on the age term of versions 1..t-2 (None = off)
        self.gather_splice = None                                           # eval-only (note 212-6): J -> gather output replaced by N(V_{t-1}) at slots j >= J
        self.gather_qmod = None                                             # eval-only (note 212-6): gather query position j -> j mod this (None = wpe_slice)
        self.gather_split_log = None                                        # eval-only (211-15 A iv): a list -> per pass (t, self, older-same-slot, cross-slot) per item
        self.slot_mask = None                                               # eval-only (212-15 MASKVOID / GAPPED): f(t, V_t) -> (B, T) bool slots REMOVED from every reader
        #   (core / coda bank entries of every version, the current version's keys, the gather) from then on (accumulated; t = 0 is called on V_0)
        self.gather_view = None                                             # eval-only (note 213-11, PA d3hist swaps): f(t, gk, NV) -> (gk', NV') for the gather at pass t
        self.keep_bank_only = False                                         # eval-only (note 213-11, PA OTHER-ABLATE): ver_keep reaches the core / coda bank, not the gather
        self.dup_foreign = None                                             # eval-only (note 213-22): [donor stored S_0..S_K] -> versions 1..t-2 := the donor's S_{t-1}
        self.dup_newest = False                                             # eval-only (note 213-21 DUP): at pass t every reader's versions 1..t-2 := the newest stored V_{t-1}
        self.ver_sub_from = None                                            # eval-only (§213 OLD-SWAP): rollout's ver_sub replaces entries for reading index >= this
        self.input_edit = None                                              # eval-only (213-1 LFK FAULT): f(t, U_t) -> U_t' (the pass input, CARRY / gather / SLVA)
        self.gather_hist_ablate = None                                      # eval-only (212-15 STAB-HIST): "self" | "other" -> the gather's keys (s, j) | (s, i != j),
        #   1 <= s <= t-2, masked for query slot j (rt_d3/rt_hist2.py); None = unchanged

    @property
    def anneals(self):
        """capability flag (§213 G-D4): the pass input is an annealed read (c_anneal drives it); the trainer keys the c schedule and its logging on this."""
        return self.mode == "pa"

    def nrm(self, V, rms0):
        return self.gain * (V * (rms0 / V.pow(2).mean(-1, keepdim=True).sqrt()))

    def gkey(self, V):
        B, T, D = V.shape; a = F.layer_norm(V, (D,), self.gln_w, self.gln_b)
        return (a @ self.gk_w + self.gk_b).view(B, T, self.H, self.hd).transpose(1, 2)

    def gather(self, t, gk, NV, pad, off_v0=False, keep=None):
        """PURE-ANNEAL (draft §3): U[j] = sum_{s < t, i} softmax(l_{j,(s,i)} - c * 1[(s, i) != (t-1, j)]) N(V_s)[i], per head on its 64-d slice; fp32 (fp64 in
        the gates), autocast off: at c = 1e4 the weights are exactly one-hot, so U == N(V_{t-1}) bitwise (gate_carry_bitwise)."""
        if self.gather_view is not None:                                    # eval-only (213-11)
            gk, NV = self.gather_view(t, gk, NV)
        if self.keep_bank_only:                                             # eval-only (213-11)
            keep = None
        B, T = pad.shape; H, hd = self.H, self.hd; dev = pad.device; dt = torch.float64 if NV[0].dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=dev.type, enabled=False):
            wq_ = DM_().wpe_slice(self.g, T) if self.gather_qmod is None else self.g.w("wpe.weight")[torch.arange(T, device=dev) % self.gather_qmod]   # 212-6 mod arm
            q = (wq_.to(dt) @ self.gq_w.to(dt) + self.gq_b.to(dt)).view(T, H, hd).transpose(0, 1)   # (H, T, hd), content-free;
            # wpe_slice = wpe[:T] unless deltamem.POS_OFFSET / POS_KEEP0 are set (eval-only position tests, note 212-x; the prelude uses the same slice)
            K_ = torch.cat(gk[:t], 2).to(dt); lg = torch.einsum("htd,bhsd->bhts", q, K_) / math.sqrt(hd)                    # (B, H, T, tT)
            s_idx = torch.arange(t, device=dev).repeat_interleave(T); i_idx = torch.arange(T, device=dev).repeat(t)
            ag_ = self.gamma_q.to(dt)[None, :, None, None] * (t - 1 - s_idx).to(dt)[None, None, None, :]
            if self.age_floor is not None:                                  # eval-only (note 212-3): the same slope floor on the gather's age term
                ap_ = ((s_idx >= 1) & (s_idx <= t - 2))[None, None, None, :]; ag_ = torch.where(ap_, torch.minimum(ag_, -self.age_floor * (t - 1 - s_idx).to(dt)[None, None, None, :]), ag_)
            lg = lg + ag_
            own = (s_idx[None, :] == t - 1) & (i_idx[None, :] == torch.arange(T, device=dev)[:, None])                   # (T, tT): the (t-1, j) entry
            lg = lg - self.c_anneal.to(dt) * (~own).to(dt)[None, None]
            valid = pad[:, None, :].expand(B, t, T).reshape(B, t * T)
            if off_v0:
                valid = valid & (s_idx != 0)[None]
            if keep is not None:
                valid = valid & keep.to(dev).repeat_interleave(T)[None]
            if self.gather_hist_ablate is None:
                w = torch.softmax(lg.masked_fill(~valid[:, None, None, :], NEG), -1)
            else:                                                           # eval-only (212-15 STAB-HIST): per query slot j, mid versions 1..t-2 at slot j (self) / i != j (other)
                mid = ((s_idx >= 1) & (s_idx <= t - 2))[None, :]; same = i_idx[None, :] == torch.arange(T, device=dev)[:, None]   # (Tq, tT)
                rm = mid & (same if self.gather_hist_ablate == "self" else ~same); assert self.gather_hist_ablate in ("self", "other")
                w = torch.softmax(lg.masked_fill(~(valid[:, None, :] & ~rm[None])[:, None], NEG), -1)

            U = (w @ torch.cat(NV[:t], 1).to(dt).view(B, t * T, H, hd).transpose(1, 2)).transpose(1, 2).reshape(B, T, H * hd)
            if self.gather_log is not None:
                sm = w[:, :, torch.arange(T, device=dev), (t - 1) * T + torch.arange(T, device=dev)].detach()                # (B, H, T)
                self.gather_log.append((t, (sm.mean(1) * pad).sum(1) / pad.sum(1)))
            if self.gather_split_log is not None:                           # eval-only (211-15 A iv): mass on (t-1, j) / on (s < t-1, j) / on (., i != j)
                same = (i_idx[None, :] == torch.arange(T, device=dev)[:, None]).to(w.dtype)                                  # (T, tT): slot j in any version
                sm_ = w[:, :, torch.arange(T, device=dev), (t - 1) * T + torch.arange(T, device=dev)]; sa_ = (w * same[None, None]).sum(-1)
                mean_ = lambda z: ((z.mean(1) * pad).sum(1) / pad.sum(1)).detach()  # noqa: E731
                self.gather_split_log.append((t, mean_(sm_), mean_(sa_ - sm_), mean_(1 - sa_)))
        if self.gather_splice is not None:                                  # eval-only (note 212-6 GATHER-SPLICE): slots j >= J take the CARRY value N(V_{t-1}[j])
            U = torch.cat([U[:, :self.gather_splice], NV[t - 1].to(U.dtype)[:, self.gather_splice:]], 1)
        return U.to(NV[0].dtype)

    def coda(self, t, Vt, kvs, pad, src=None, bp=None, km=None):
        """blocks 10-11 over [bank versions 1..t-1 (kvs {layer: [(k, v)] * t}; V_0 excluded, note 211-1 pin 4) | V_t causal] + ln_f -> (b, T, D); t <= 1: no
        bank (K' = 0 is the coda on V_0 as its causal input). src: a weight source (.w / .H; default
        the trunk) and bp: (beta, beta0, gam) rows for the 2 coda layers (default this model's) - the PROBE(t) coda copy passes its own.
        km: block_bank's kmask (eval-only, 212-15; pad then excludes the same slots from the bank)."""
        src = src or self.g; D = self.D; h = Vt; bp = bp or (self.beta[4:], self.beta0[4:], self.gam[4:])
        for r, i in enumerate(CODA):
            if t <= 1:
                h = block_bank(src, i, h, None, None, None, km)
            else:
                bk = torch.cat([k for k, _ in kvs[i][1:t]], 2); bv = torch.cat([v for _, v in kvs[i][1:t]], 2); T_ = h.shape[1]
                bb = bias_row(bp[0][r], bp[1][r], bp[2][r], t, t, pad, False, self.bank_off, self.ver_keep(t, t) if self.ver_keep else None, self.age_floor)[..., T_:].expand(-1, -1, T_, -1)
                if self.cap is not None:
                    self.cap[("coda", i, t)] = h.detach()
                h = block_bank(src, i, h, bk, bv, bb, km)
        return F.layer_norm(h, (D,), src.w("ln_f.weight"), src.w("ln_f.bias"))

    def rollout(self, x, pad, K, coda_rows=None, edit=None, v0_off_after=None, sg_input=False, prefix=None, ver_sub=None, v0=None, pause=None):
        """x (B, T) token ids [EOT, expr, ' ='] right-padded; pad (B, T) real-slot mask; K passes.
        coda_rows {t: LongTensor rows}: the coda after pass t (t = 0: on V_0) on those rows -> out["coda"][t] = (rows, h (b, T, D)).
        edit(t, V_t) -> the stored entry V_t' (bank, and the carry), or (bank_t, carry_t) under CARRY (VALUE-SWAP iii); None = V_t (interventions, MCT).
        v0_off_after: V_0's entry masked (core layers and gather) for passes > it (VALUE-SWAP iv; the coda never reads V_0, pin 4).
        sg_input: the pass input U (carry / gather) stop-gradiented - the path-specific gradflow gate (only the bank reads carry gradient across passes).
        Eval-only (note 212-3, FOREIGN): prefix = [F_1..F_o] stored versions pushed right after V_0 before pass 1 (INSERT: passes then run t = o + 1..K,
        K absolute; the carry stays V_0); ver_sub = [F_1..F_o]: entry s in 1..o is READ as F_s by every pass t > s + 1 and by the coda after pass t >= s + 1
        (REPLACE; the pass that consumes V_s still reads the real V_s). None = unchanged.
        Eval-only (212-15): v0 = a given V_0 (GAPPED's interior-masked prelude); self.slot_mask (see __init__). None = unchanged.
        §214 (D4, training + eval_pause only): pause = (B, K + 1) bool -> at pass t, U_t += pause_vec at every real slot of the items with pause[:, t]
        (after the input read and sg_input, before block 6); the computed V_t is stored and carried as at any pass. None = unchanged.
        -> {"V": computed V_0..V_K, "S": stored entries, "coda": {...}, "kv": the bank K / V}."""
        g = self.g; B, T = x.shape; V0 = DM_().gpt_lower(g, g.embed(x), CORE[0]) if v0 is None else v0; rms0 = V0.pow(2).mean(-1, keepdim=True).sqrt()
        M = None if self.slot_mask is None else self.slot_mask(0, V0).bool() & pad     # removed slots so far (212-15); None = the default path
        pd = pad if M is None else pad & ~M
        Vc, St, kv, gk, NV, out = [V0], [V0], {i: [] for i in LAYERS}, [], [], {}

        def push(Vs):
            for i in LAYERS:
                kv[i].append(bank_kv(g, i, Vs))
            NV.append(self.nrm(Vs, rms0))
            if self.mode != "carry":                                       # PA's gather keys; SLVA's per-slot version keys (boxes/slva.py)
                gk.append(self.gkey(Vs))

        def coda_at(t, Vt):
            if coda_rows and t in coda_rows:
                r_ = coda_rows[t]
                out[t] = (r_, self.coda(t, Vt[r_], {i: [(k[r_], v[r_]) for k, v in dup(view(t + 1, kv[i], fkv[i] if fkv else None), t, ("kv", i))] for i in CODA}, pd[r_],
                                        km=None if M is None else M[r_]))
        fkv = fNV = fgk = None
        if ver_sub is not None:
            fkv = {i: [bank_kv(g, i, F_) for F_ in ver_sub] for i in LAYERS}; fNV = [self.nrm(F_, rms0) for F_ in ver_sub]
            fgk = [self.gkey(F_) for F_ in ver_sub] if self.mode != "carry" else None

        fc = {}

        def dup(lst, t, kind):                                              # 213-21 / 213-22 DUP (eval-only): entries 1..t-2 read as entry t-1 (dup_newest) or as the
            if t < 3 or not (self.dup_newest or self.dup_foreign is not None):   # donor's S_{t-1} (dup_foreign; per reader kind: ("kv", layer) / "gk" / "NV")
                return lst
            if self.dup_newest:
                return [lst[0]] + [lst[t - 1]] * (t - 2) + lst[t - 1:]
            if (kind, t) not in fc:
                Vd = self.dup_foreign[t - 1]
                fc[(kind, t)] = self.nrm(Vd, rms0) if kind == "NV" else self.gkey(Vd) if kind == "gk" else bank_kv(g, kind[1], Vd)
            return [lst[0]] + [fc[(kind, t)]] * (t - 2) + lst[t - 1:]

        def view(tr, lst, flst):                                            # REPLACE: entry s (1..o) read as F_s when the reading index tr > s + 1
            if flst is None:                                                # (ver_sub_from, eval-only §213 OLD-SWAP: ... when tr >= ver_sub_from instead)
                return lst
            if self.ver_sub_from is None:
                return [flst[s_ - 1] if 1 <= s_ <= len(flst) and tr > s_ + 1 else lst[s_] for s_ in range(len(lst))]
            return [flst[s_ - 1] if 1 <= s_ <= len(flst) and tr >= self.ver_sub_from else lst[s_] for s_ in range(len(lst))]
        coda_at(0, V0); push(V0); carry = V0
        for F_ in (prefix or []):                                           # INSERT: foreign versions after V_0
            Vc.append(F_); St.append(F_); push(F_)
        for t in range(1 + len(prefix or []), K + 1):
            off_ = v0_off_after is not None and t > v0_off_after
            kp_ = self.ver_keep(t, t) if self.ver_keep else None
            h = self.nrm(carry, rms0) if self.mode == "carry" else self.gather(t, dup(view(t, gk, fgk), t, "gk"), dup(view(t, NV, fNV), t, "NV"), pd, off_, kp_)
            h = h.detach() if sg_input else h
            if pause is not None:                                           # §214 computed PAUSE pass
                h = h + ((pause[:, t][:, None] & pad).to(h.dtype)[..., None] * self.pause_vec.to(h.dtype))
            if self.input_edit is not None:                                 # eval-only (213-1 LFK FAULT): the pass input U_t -> input_edit(t, U_t)
                h = self.input_edit(t, h)
            for li, i in enumerate(CORE):
                kvi = dup(view(t, kv[i], fkv[i] if fkv else None), t, ("kv", i)); bk = torch.cat([k for k, _ in kvi], 2); bv = torch.cat([v for _, v in kvi], 2)
                bb = bias_row(self.beta[li], self.beta0[li], self.gam[li], t, t, pd, off_, self.bank_off, kp_, self.age_floor).expand(-1, -1, T, -1)
                if self.cap is not None:
                    self.cap[("core", i, t)] = h.detach()
                h = block_bank(g, i, h, bk, bv, bb, M)
            Vc.append(h)
            if M is not None:                                               # 212-15: slots removed after this pass (the coda after pass t already drops them)
                M = M | (self.slot_mask(t, h).bool() & pad); pd = pad & ~M
            coda_at(t, h)
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
    def mct(self, V, E, valid, mu=1.0, max_iter=3, aim=0.05):
        """R3 (draft §4; pin 7): the minimum-norm edit in R's row space, in LN coordinates, so that R decodes E (B, T) at every valid slot with margin
        mu nats over the runner-up: d_y = max(0, max_{c != y} l_c + mu - l_y), dz = pinv(W) d, mapped back to the residual as sd * dz / ln_w (mean
        removed); repeated (<= max_iter) while a slot misses the margin. Note 211-3: each step aims at mu + aim (1.05 nats) and W1 tests mu (1.0), so the
        share of slots needing > 1 step is informative. -> (V', {"steps": per-slot step count, "rel": [||dV|| / ||V|| per step]})."""
        W = self.rec.weight.float(); b = self.rec.bias.float(); Wp = torch.linalg.pinv(W); lw = self.rec_ln.weight.float(); lb = self.rec_ln.bias.float()
        lw_safe = torch.where(lw.abs() < 1e-3, torch.full_like(lw, 1e-3) * torch.sign(lw + 1e-12), lw)
        V2 = V.float().clone(); E_ = E.clamp_min(0)[..., None]; steps = torch.zeros(V.shape[:2], dtype=torch.long, device=V.device); rel = []
        for _ in range(max_iter):
            mu_ = V2.mean(-1, keepdim=True); sd = (V2.var(-1, unbiased=False, keepdim=True) + self.rec_ln.eps).sqrt()
            lg = ((V2 - mu_) / sd * lw + lb) @ W.T + b; ly = lg.gather(-1, E_); lo = lg.scatter(-1, E_, NEG).max(-1, keepdim=True).values
            miss = ((lo + mu - ly) > 0) & valid[..., None]                                  # W1's test: margin mu
            if not bool(miss.any()):
                break
            need = (lo + mu + aim - ly).clamp_min(0) * miss                                  # the step aims at mu + aim
            dv = (torch.zeros_like(lg).scatter(-1, E_, need) @ Wp.T) / lw_safe * sd; dv = dv - dv.mean(-1, keepdim=True)
            rel.append(float(dv.norm() / V2.norm())); V2 = V2 + dv; steps += (need[..., 0] > 0).long()
        return V2.to(V.dtype), {"steps": steps, "rel": rel}

    @torch.no_grad()
    def calibrate(self, x, pad, lens, K=KMAX, mu=0.02, q=0.99):
        """pin 3: a CARRY rollout with the bank masked (exact CARRY) to V_0..V_K, every layer's input captured; per layer (core 6-9 and coda 10-11) and
        head, the bank mass a query would give is sigmoid(beta + log A - log C) (A = the bank entries' exp-score sum, C = the rest), so beta0 is set
        first (bank = V_0; every t >= 1) and then beta (V_1..V_{t-1}, beta0 fixed; t >= 2) to logit(mu) - 0.01 - quantile_q(log A - log C) over (items,
        real query slots, t): the q-quantile mass is then just under mu (re-measured after setting). The coda layers have no V_0 entry (pin 4): beta
        only, C = the causal part alone, their beta0 rows unused. -> a report dict (the ratio test: ratio_test())."""
        g = self.g; D, H, hd = self.D, self.H, self.hd; mode0, off0 = self.mode, self.bank_off; self.bank_off = True; self.cap = {}
        try:
            r = self.rollout(x, pad, K, coda_rows={t: torch.arange(x.shape[0], device=x.device) for t in range(1, K + 1)})
            cap = self.cap
        finally:
            self.cap = None; self.bank_off = off0
        rep = {}; lmu = math.log(mu / (1 - mu)) - 0.01                                  # 0.01 nats inside the target (m99 ~ 0.0198)
        for li, i in enumerate(LAYERS):
            kind = "core" if i in CORE else "coda"; parts = {"v0": [], "rest": [], "cur": [], "t": []}
            for t in range(1 if kind == "core" else 2, K + 1):
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
            if kind == "core":
                x0 = torch.stack([(a0 - c0) for a0, c0 in zip(parts["v0"], parts["cur"])], 0)                # (K, B, H, T): log A - log C for beta0
                b0 = torch.stack([lmu - torch.quantile(x0[:, :, h_, :][:, sel[:, h_, :]].flatten(), q) for h_ in range(H)])
                self.beta0.data[li] = b0.to(self.beta0.dtype)
                m0 = torch.sigmoid(x0 + b0[None, None, :, None]); m0q = [float(torch.quantile(m0[:, :, h_, :][:, sel[:, h_, :]].flatten(), q)) for h_ in range(H)]
                x1 = torch.stack([ar - torch.logaddexp(c0, a0 + b0[None, :, None]) for ar, a0, c0, t in zip(parts["rest"], parts["v0"], parts["cur"], parts["t"]) if t >= 2], 0)
            else:                                                           # pin 4: the coda's bank is V_1..V_{t-1} (rest = its versions 1..t-1)
                b0, m0q = None, []
                x1 = torch.stack([ar - c0 for ar, c0 in zip(parts["rest"], parts["cur"])], 0)
            b1 = torch.stack([lmu - torch.quantile(x1[:, :, h_, :][:, sel[:, h_, :]].flatten(), q) for h_ in range(H)])
            self.beta.data[li] = b1.to(self.beta.dtype)
            m1 = torch.sigmoid(x1 + b1[None, None, :, None]); m1q = [float(torch.quantile(m1[:, :, h_, :][:, sel[:, h_, :]].flatten(), q)) for h_ in range(H)]
            rep[str(i)] = {"beta0": [round(float(v), 4) for v in b0] if b0 is not None else None, "beta":[round(float(v), 4) for v in b1], "m99_v0": m0q, "m99_rest": m1q}
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


def make_model(g, mode, table="train", pause="none"):
    """the loader / trainer factory (§213 G-D0): mode pa | carry -> VBankN; slva -> boxes.slva.SLVA (table train | frozen). Callers assert
    type(m).__name__ == args["arch"] (absent in pre-§213 args = "VBankN"). §214 (D4): pause != "none" adds the 768-vector pause_vec (init 0), the
    only new key; absent otherwise, so strict loads of every earlier model are unchanged."""
    if mode == "slva":
        from boxes.slva import SLVA
        m = SLVA(g, table)
    else:
        assert table == "train", "the frozen table is a §213 SLVA cell option"
        m = VBankN(g, mode)
    if pause != "none":
        m.pause_vec = torch.nn.Parameter(torch.zeros(m.D, device=g.w("wte.weight").device))
    return m


def pause_schedule(rng_p, n, p_max=8):
    """§214 PAUSE-A (D4_v1 §2.2, lastR): P ~ U{1..p_max}, D ~ U{0..P} start pauses (passes 1..D), the other S = P - D pauses uniformly over the first
    n - 1 + S mid positions (the last mid position is a reduction, so the completion pass is n + P). All draws from rng_p.
    -> a bool list over passes 1..n + P (True = pause)."""
    P = int(rng_p.integers(1, p_max + 1)); D = int(rng_p.integers(0, P + 1)); S = P - D; L1 = n - 1 + S
    mid = [False] * (L1 + 1)
    if S > 0:
        for j in rng_p.choice(L1, S, replace=False):
            mid[int(j)] = True
    return [True] * D + mid


def DM_():
    import boxes.deltamem as DM
    return DM


def batch_x(items, dev):
    """[EOT] + expr + [' ='] right-padded with EOT -> (x (B, T), pad (B, T) bool, lens (B,))."""
    seqs = [[N.EOT] + N.ids(it["syms"]) + [N.TID[" ="]] for it in items]; ids_, m_ = N.pad(seqs)
    return torch.tensor(ids_, device=dev), torch.tensor(m_, device=dev), torch.tensor([len(q_) for q_ in seqs], device=dev)



# ------------------------------------------------------------------ Part 4: READY-RECODE training (draft §4; note 211-1 engineering pins c-e)
VAL0 = CID[" 0"]                                                            # the value classes are VAL0 .. VAL0 + 10


def rec_targets(items, Kb, T, dev):
    """(B, Kb + 1, T) class ids of E_0..E_Kb (HOLD past n) on each item's slots; -100 on padding."""
    E = np.full((len(items), Kb + 1, T), -100, dtype=np.int64)
    for r, it in enumerate(items):
        if "E_self" in it:                                                  # §215: a self item's composed targets (no root, no gold), HOLD-padded to Kb
            es = it["E_self"]; es = es[:Kb + 1] + [es[-1]] * (Kb + 1 - len(es)); e = np.array(es)
        else:
            e = np.array(recode_targets(it["root"], Kb))
        E[r, :, :e.shape[1]] = e
    return torch.tensor(E, device=dev)


def vb_loss(m, items, Ks, dev, ac, rec_w=1.0, tf_p=0.0, tf_rng=None, mu=1.0, sg_input=False, keep=False, pz=None):
    """L = L_ans + rec_w * L_rec for items with per-item pass budgets Ks (K_i); pin (c).
    R1: answer CE (full vocabulary, the ' %d' target) of the coda after pass t, t in [n_i, K_i]; mean over t per item, then over items.
    R2: per (item, t), t in 1..K_i: 1/2 the weighted-mean CE of R(V_t) over changed slots (E_t != E_{t-1}) + 1/2 over unchanged real slots, value-class
    targets weighted 2; with no changed slot (HOLD) the unchanged mean alone; mean over t per item, then over items.
    R3 (tf_p > 0; pin (e)): each stored entry V_s (s >= 1) of each item is, with probability tf_p (tf_rng), replaced by sg(MCT(V_s)) in the bank and the
    carry; the losses read the computed V_s. -> (L, stats, the rollout if keep).
    §214 pz (per item: None = plain, or pause_schedule's bool list over passes 1..n + P): the rollout gets the pause mask; R2's target at pass t is
    E_{rho(t)}, rho(t) = t - #pauses in passes <= t (so a pause pass is HOLD); R1's window is [tau_i, K_i], tau_i = n_i + P_i (= n_i when plain).
    Named assertion (G-P2): no paused item has an R1 term at t < tau_i."""
    x, pad, lens = batch_x(items, dev); B, T = x.shape; Kb = int(max(Ks)); n_ = torch.tensor([it["n"] for it in items], device=dev); Kt = torch.tensor(Ks, device=dev)
    E = rec_targets(items, Kb, T, dev); pmask = None; tau = n_
    if pz is not None and any(z is not None for z in pz):                   # §214: map targets, pause mask, R1 window
        pmask = torch.zeros(B, Kb + 1, dtype=torch.bool, device=dev); rho = torch.arange(Kb + 1, device=dev)[None, :].repeat(B, 1)
        for b, z in enumerate(pz):
            if z is not None:
                assert len(z) <= Kb and sum(1 for q in z if not q) == int(n_[b]), "pause schedule != n reductions within K"
                pmask[b, 1:len(z) + 1] = torch.tensor(z, device=dev); rho[b] = torch.arange(Kb + 1, device=dev) - torch.cumsum(pmask[b].long(), 0)
        E = E.gather(1, rho.clamp(0, Kb)[:, :, None].expand(-1, -1, T))
        tau = n_ + pmask.sum(1)
    win = {t: torch.nonzero((tau <= t) & (t <= Kt)).flatten() for t in range(0, Kb + 1)}; win = {t: r_ for t, r_ in win.items() if len(r_)}
    for t, r_ in win.items():                                               # G-P2 named assertion: no R1 term before an item's completion pass
        assert bool((tau[r_] <= t).all()), "R1 term before tau"
    st = {"tf_n": 0, "tf_tot": 0, "mct_slots": 0, "mct_multi": 0, "mct_rel": []}

    def edit(t, Vt):
        if tf_p <= 0:
            return None
        live = t <= Kt; draw = torch.tensor(tf_rng.random(B) < tf_p, device=dev) & live
        st["tf_tot"] += int(live.sum()); st["tf_n"] += int(draw.sum())
        if not bool(draw.any()):
            return None
        Vm, ms = m.mct(Vt.detach(), E[:, t], pad, mu); stp = ms["steps"][draw]
        st["mct_slots"] += int((stp >= 1).sum()); st["mct_multi"] += int((stp > 1).sum()); st["mct_rel"] += ms["rel"]
        return torch.where(draw[:, None, None], Vm.to(Vt.dtype), Vt)
    with ac():
        r = m.rollout(x, pad, Kb, coda_rows=win, edit=edit, sg_input=sg_input, pause=pmask)
    y = torch.tensor(N.NUM_IDS[[it["t"] for it in items]], device=dev)
    ans_sum = torch.zeros(B, device=dev); ans_cnt = torch.zeros(B, device=dev)
    for t, (rows, h) in r["coda"].items():
        ce = F.cross_entropy(m.answer_logits(h, lens[rows]), y[rows], reduction="none").float()
        ans_sum = ans_sum.index_add(0, rows, ce); ans_cnt = ans_cnt.index_add(0, rows, torch.ones_like(ce))
    rec_sum = torch.zeros(B, device=dev); rec_cnt = torch.zeros(B, device=dev)
    for t in range(1, Kb + 1):
        Et, Ep = E[:, t], E[:, t - 1]; valid = pad & (t <= Kt)[:, None]
        if not bool(valid.any()):
            continue
        ce = F.cross_entropy(m.rec_logits(r["V"][t]).transpose(1, 2), Et.clamp_min(0), reduction="none").float()
        w = 1.0 + (Et >= VAL0).float(); ch = ((Et != Ep) & valid).float(); un = ((Et == Ep) & valid).float()
        tc = (w * ce * ch).sum(1) / (w * ch).sum(1).clamp_min(1e-9); tu = (w * ce * un).sum(1) / (w * un).sum(1).clamp_min(1e-9)
        live = (t <= Kt).float(); term = torch.where(ch.sum(1) > 0, 0.5 * tc + 0.5 * tu, tu)
        rec_sum = rec_sum + term * live; rec_cnt = rec_cnt + live
    has_a = ans_cnt > 0; L_ans = (ans_sum[has_a] / ans_cnt[has_a]).mean(); L_rec = (rec_sum / rec_cnt.clamp_min(1)).mean()
    st.update(answer_ce=float(L_ans.detach()), rec_ce=float(L_rec.detach()), n_paused=int(sum(z is not None for z in pz)) if pz else 0)
    return L_ans + rec_w * L_rec, st, (r if keep else None)


class CurGate:
    """R4 (draft §4; REDTEAM F9b; pin (d)): stages n_max = ns[0] -> ... -> ns[-1]. A stage advances when `need` consecutive completed 1k-step bins
    (counted from the stage's first step) of the per-step readiness-window answer CE (not the total loss) average < thr, or after `cap` steps in the
    stage; the final stage runs to --steps (the LR cosine is over --steps)."""
    def __init__(self, ns, thr=0.3, need=2, cap=8000, bin_=1000):
        self.ns, self.thr, self.need, self.cap, self.bin = list(ns), thr, need, cap, bin_; self.stage = 0; self.start = 1; self.cur = []; self.bins = []; self.hist = []

    def n_max(self):
        return self.ns[self.stage]

    def update(self, step, ans_ce):
        """after step `step` (ans_ce None = a non-finite step, skipped) -> (new stage, reason, last bin mean) on an advance, else None."""
        if self.stage == len(self.ns) - 1:
            return None
        if ans_ce is not None:
            self.cur.append(ans_ce)
        reason = None
        if (step - self.start + 1) % self.bin == 0:
            self.bins.append(float(np.mean(self.cur)) if self.cur else float("inf")); self.cur = []
            if len(self.bins) >= self.need and all(b_ < self.thr for b_ in self.bins[-self.need:]):
                reason = "answer_ce < %.2f for %d bins" % (self.thr, self.need)
        if reason is None and step - self.start + 1 >= self.cap:
            reason = "cap %d steps" % self.cap
        if reason is None:
            return None
        self.hist.append({"step": step, "from": self.ns[self.stage], "to": self.ns[self.stage + 1], "reason": reason, "bins": self.bins[-4:]})
        self.stage += 1; self.start = step + 1; self.cur = []; self.bins = []
        return self.stage, reason, (self.hist[-1]["bins"][-1] if self.hist[-1]["bins"] else None)

    def state(self):
        return {k: getattr(self, k) for k in ("stage", "start", "cur", "bins", "hist")}

    def load(self, d):
        for k, v in d.items():
            setattr(self, k, v)


def c_at(step, hold=1000, end=8000, c1=20.0):
    """PURE-ANNEAL c(step) (note 211-3): 1e4 to `hold` (exact CARRY, function preservation), then c1 * (1 - (step - hold) / (end - hold)), linear to 0
    at `end` (a buffer, not a parameter). The old 1e4-linear schedule was a step switch at ~8k (off-self mass 3e-60 at 7.9k, 0.997 at 8.0k)."""
    return 1e4 if step <= hold else max(0.0, c1 * (1.0 - (step - hold) / (end - hold)))


def tf_at(step, hold=4000, end=24000):
    """R3 p_TF(step): 1 to `hold`, linear to 0 at `end`."""
    return 1.0 if step <= hold else max(0.0, (end - step) / (end - hold))


def main():
    import argparse
    import json
    import os
    import time
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["pa", "carry", "slva"]); ap.add_argument("--mct", action="store_true", help="N3: R3 minimal-correction teacher forcing")
    ap.add_argument("--table", default="train", choices=["train", "frozen"], help="§213: frozen = wpe.requires_grad False before the optimizer (D3-F; slva only)")
    ap.add_argument("--args_ref", default="", help="§213 G-D7: an args.json to diff against; the diff is printed (ARGS-DIFF) and written to args_diff.json")
    ap.add_argument("--pause", default="none", choices=["none", "A"], help="§214 D4: A = PAUSE-A (computed pause passes, lastR)")
    ap.add_argument("--pause_p", type=float, default=0.5); ap.add_argument("--pause_max", type=int, default=8)
    ap.add_argument("--mem_check_steps", type=int, default=1, help="§214 G-P6: optimizer steps in --mem_check (2 allocates the AdamW moments first)")
    ap.add_argument("--self_pool", default="", help="§215 SELF-DEPTH: the round's pool jsonl (pool_meta.json beside it); empty = no self steps (FT-CTRL)")
    ap.add_argument("--p_self", type=float, default=0.0); ap.add_argument("--self_kmax", type=int, default=16); ap.add_argument("--self_kextra", type=int, default=4)
    ap.add_argument("--self_micro", type=int, default=2, help="§215: micro on self steps (auto-picked at the deepest self shape when --auto_micro)")
    ap.add_argument("--mem_check_self", action="store_true", help="§215 G-S9: --mem_check at the 64 longest n_top pool records, K self_kmax, micro self_micro")
    ap.add_argument("--draws_log_steps", type=int, default=0, help="§215 G-S3: steps 1..N logged to draws_log.json (both FT cells: 20000); 0 = off")
    ap.add_argument("--init_from", default="", help="FT (SELF-DEPTH / D4): a base model.pt; strict load (only pause_vec may be missing), calibrate() NOT called, "
                    "the base dir's calibration.json copied; a fresh optimizer")
    ap.add_argument("--steps", type=int, default=48000); ap.add_argument("--bs", type=int, default=64); ap.add_argument("--micro", type=int, default=1); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=1); ap.add_argument("--cur", default="3,5,8"); ap.add_argument("--cur_gate", default="0.3:2:8000", help="thr:need:cap (pin (d))")
    ap.add_argument("--c_sched", default="1000,8000"); ap.add_argument("--tf_sched", default="4000,24000"); ap.add_argument("--mu", type=float, default=1.0)
    ap.add_argument("--rec_w", type=float, default=1.0); ap.add_argument("--kmax", type=int, default=KMAX); ap.add_argument("--kextra", type=int, default=4)
    ap.add_argument("--ckpt_every", type=int, default=2000); ap.add_argument("--resume", default=None); ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--time_steps", type=int, default=0, help="G-TIME: train this many steps on forced final-stage batches (n <= 8, K 12), write timing.json + eval.json, exit")
    ap.add_argument("--mem_check", action="store_true", help="peak memory: one training step on the longest n <= 8 items at K 12 + an n 16 eval forward at K' 24 -> mem_check.json")
    ap.add_argument("--eval_bs", type=int, default=64, help="the host's descriptive acc_n2 eval (250 OOMed at n 16 on 24 GB, §211 10-16)"); ap.add_argument("--per_n", type=int, default=1000); ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--auto_micro", action="store_true", help="before training: the smallest micro in 1, 2, 4, 8 whose forward + backward on the longest final-stage "
                    "batch (n <= 8, K 12, fixed items) peaks <= --mem_budget GiB (no optimizer step; the run's rngs untouched) -> logged")
    ap.add_argument("--mem_budget", type=float, default=21.0)
    ap.add_argument("--stop_at", type=int, default=0, help="smokes (G1b): stop after this step, schedules unchanged (no final eval)")
    ap.add_argument("--keep_ckpt_steps", default="", help="comma steps (0 = after calibration) at which a model-only copy ckpt_model_<step>.pt is kept (G1b smoke)")
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--out", required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); dev = a.device
    from boxes import s5
    from boxes.fsq120 import lr_at, driver_version, env_info
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); rng_k = np.random.default_rng(a.seed + 21100); rng_tf = np.random.default_rng(a.seed + 21200)
    amp = None if (a.fp32 or not dev.startswith("cuda") or torch.cuda.get_device_capability(dev)[0] < 8) else torch.bfloat16
    ac = lambda: torch.autocast("cuda", dtype=amp, enabled=amp is not None)  # noqa: E731
    TE = N.test_set(a.per_n); EXCL = {tuple(it["syms"]) for it in TE}; r_sp = np.random.default_rng(17300)    # §170's exclusion: the test set + the SERIAL pool
    for _ in range(2200):
        EXCL.add(tuple(N.render(N.generate(r_sp, 8)[0])))
    ns = [int(v) for v in a.cur.split(",")]; thr, need, cap = a.cur_gate.split(":"); GATE = CurGate(ns, float(thr), int(need), int(cap))
    ch0, ch1 = (int(v) for v in a.c_sched.split(",")); tf0, tf1 = (int(v) for v in a.tf_sched.split(","))
    import boxes.deltamem as DMm
    assert DMm.POS_OFFSET == 0 and not DMm.POS_KEEP0, "training needs the plain position table (POS_OFFSET 0)"   # §213 G-D7
    m = make_model(s5.load_gpt2(dev), a.mode, a.table, a.pause).to(dev); a.arch = type(m).__name__
    rng_p = np.random.default_rng(a.seed + 21300) if a.pause != "none" else None   # §214: the coin, P, D and positions only (G-P12); rng / rng_k untouched
    pool, rng_s, selfst = None, None, {"cursor": 0, "n_self_steps": 0, "self_reuse": 0}   # not "sd" (init_from binds and dels sd)
    if a.self_pool:                                                         # §215: the pool (sha + key-set checked); rng_s = seed + 21800 (coin + self U only)
        import boxes.selfdepth as SD
        pool, pool_meta = SD.load_pool(a.self_pool, os.path.join(os.path.dirname(a.self_pool), "pool_meta.json")); rng_s = np.random.default_rng(a.seed + 21800)
        print("SELF_POOL", a.self_pool, len(pool), "records, sha", pool_meta["sha256"][:16], flush=True)
    wpe0 = None
    if a.table == "frozen":                                                 # §213 D3-F: frozen BEFORE the optimizer is built (no decoupled decay)
        from boxes.slva import wpe_sha
        m.g.p["wpe_weight"].requires_grad_(False); wpe0 = wpe_sha(m)
    opt = torch.optim.AdamW([p_ for p_ in m.parameters() if p_.requires_grad], lr=a.lr, weight_decay=0.01)   # pa / carry: every parameter, as before
    info = {"train_gpu": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu", "train_driver": driver_version(), **env_info()}
    step0, t_prev, n_nonfinite = 1, 0.0, 0
    if a.resume and os.path.exists(a.resume):
        ck = torch.load(a.resume, map_location=dev, weights_only=False); m.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        rng.bit_generator.state = ck["np_rng"]; rng_k.bit_generator.state = ck["rng_k"]; rng_tf.bit_generator.state = ck["rng_tf"]; torch.set_rng_state(ck["torch_rng"].cpu())
        if rng_p is not None:
            rng_p.bit_generator.state = ck["rng_p"]
        if rng_s is not None:                                               # §215: cursor, rng_s and self counters
            rng_s.bit_generator.state = ck["rng_s"]; selfst.update(ck["self_state"])
        GATE.load(ck["gate"]); step0 = ck["step"] + 1; t_prev = ck["elapsed"]; n_nonfinite = ck["n_nonfinite"]; cal = ck["calibration"]
        print("RESUME", a.resume, "step", ck["step"], flush=True); del ck
    else:                                                                   # pin 3: beta0 / beta calibrated at step 0 (CARRY rollout, bank masked; fp32)
        if a.init_from:                                                     # FT: the base's weights and calibration; calibrate() would overwrite beta / beta0
            sd = torch.load(a.init_from, map_location=dev, weights_only=True); mk = m.load_state_dict(sd, strict=False); del sd
            assert not mk.unexpected_keys and all(k_ == "pause_vec" for k_ in mk.missing_keys), mk
            assert a.pause != "none" or mk.missing_keys == [], ("§215: init_from must load every key", mk.missing_keys)
            cal = json.load(open(os.path.join(os.path.dirname(a.init_from), "calibration.json")))
            print("INIT_FROM %s (missing: %s); calibrate() NOT called; base calibration.json copied" % (a.init_from, mk.missing_keys), flush=True)
        else:
            r_c = np.random.default_rng(21110); cal_its = []
            while len(cal_its) < 64:
                r_, y_ = N.generate(r_c, int(r_c.integers(1, 9))); s_ = N.render(r_)
                if tuple(s_) not in EXCL:
                    cal_its.append({"syms": s_, "t": y_, "root": r_})
            xc, pc, lc = batch_x(cal_its, dev); cal = m.calibrate(xc, pc, lc); cal["ratio_test"] = m.ratio_test(xc, pc, lc)
        json.dump(cal, open(os.path.join(a.out, "calibration.json"), "w", newline="\n"), indent=1)
        json.dump({**vars(a), **info}, open(os.path.join(a.out, "args.json"), "w", newline="\n"), indent=1)
        if a.args_ref:                                                      # §213 G-D7 ARGS-DIFF (launch log)
            ref = json.load(open(a.args_ref)); cur = {**vars(a), **info}; skip = {"args_ref"}
            dif = {k: [ref.get(k), cur.get(k)] for k in sorted(set(ref) | set(cur)) if k not in skip and ref.get(k) != cur.get(k)}
            json.dump({"ref": a.args_ref, "diff": dif}, open(os.path.join(a.out, "args_diff.json"), "w", newline="\n"), indent=1); print("ARGS-DIFF", json.dumps(dif), flush=True)
        print("CALIBRATION max_m99 %.4f ratio/e5 %s" % (cal["max_m99"], cal["ratio_test"]["ratio_over_e5"]), flush=True)
    if a.auto_micro and dev.startswith("cuda"):                             # the largest training shape (final stage, K 12, the longest items)
        big = sorted(N.train_items(np.random.default_rng(0), 8 * a.bs, ns[-1], set()), key=lambda it: -len(it["syms"]))[:a.bs]; am = {}
        for mic in (1, 2, 4, 8):
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(dev); ok = True
            try:
                for k_ in range(mic):
                    sub = big[k_ * (a.bs // mic):(k_ + 1) * (a.bs // mic)]; L_, _, _ = vb_loss(m, sub, [a.kmax] * len(sub), dev, ac, a.rec_w, 0.0)
                    (L_ / mic).backward(); del L_
                    if a.mct:                                               # the MCT path's extra memory (p_TF 1) on the same chunk
                        L_, _, _ = vb_loss(m, sub, [a.kmax] * len(sub), dev, ac, a.rec_w, 1.0, np.random.default_rng(0)); (L_ / mic).backward(); del L_
            except torch.OutOfMemoryError:
                ok = False
            m.zero_grad(set_to_none=True); pk = torch.cuda.max_memory_allocated(dev) / 2 ** 30; am[str(mic)] = round(pk, 2) if ok else "OOM"
            if ok and pk <= a.mem_budget:
                a.micro = mic; break
        torch.cuda.empty_cache(); log_am = {"auto_micro": a.micro, "peak_gib_by_micro": am, "mem_budget": a.mem_budget}; print("AUTO_MICRO", log_am, flush=True)
        json.dump(log_am, open(os.path.join(a.out, "auto_micro.json"), "w", newline="\n"), indent=1)
    if pool is not None:                                                    # §215 G-S9: the deepest self shape (the 64 longest n 12 records, K self_kmax)
        n_top = max(r_["n"] for r_ in pool)                                 # once (inside the comprehension it was O(N^2): b310's mem_check hung 25 min)
        deep = sorted([r for r in pool if r["n"] == n_top], key=lambda r: -len(r["syms"]))[:a.bs]
        deep_its = [SD.self_item(r, a.self_kmax) for r in deep]
        if a.auto_micro and dev.startswith("cuda"):
            amS = {}
            for mic in (1, 2, 4, 8):                                         # §216 T1: {1, 2, 4, 8} (round 2 K 20)
                torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(dev); ok = True
                try:
                    for k_ in range(mic):
                        sub = deep_its[k_ * (a.bs // mic):(k_ + 1) * (a.bs // mic)]; L_, _, _ = vb_loss(m, sub, [a.self_kmax] * len(sub), dev, ac, a.rec_w, 0.0); (L_ / mic).backward(); del L_
                        if a.mct:
                            L_, _, _ = vb_loss(m, sub, [a.self_kmax] * len(sub), dev, ac, a.rec_w, 1.0, np.random.default_rng(0)); (L_ / mic).backward(); del L_
                except torch.OutOfMemoryError:
                    ok = False
                m.zero_grad(set_to_none=True); pk = torch.cuda.max_memory_allocated(dev) / 2 ** 30; amS[str(mic)] = round(pk, 2) if ok else "OOM"
                if ok and pk <= a.mem_budget:
                    a.self_micro = mic; break
            else:
                raise RuntimeError("G-S9: no self micro in {1, 2, 4, 8} fits the deepest self shape within mem_budget %.1f GiB: %s" % (a.mem_budget, amS))
            torch.cuda.empty_cache(); log_amS = {"self_micro": a.self_micro, "peak_gib_by_micro": amS, "mem_budget": a.mem_budget}; print("AUTO_MICRO_SELF", log_amS, flush=True)
            json.dump(log_amS, open(os.path.join(a.out, "auto_micro_self.json"), "w", newline="\n"), indent=1)
    KEEP = {int(v) for v in a.keep_ckpt_steps.split(",")} if a.keep_ckpt_steps else set()
    if 0 in KEEP and step0 == 1:
        torch.save({"model": m.state_dict(), "step": 0}, os.path.join(a.out, "ckpt_model_0.pt"))
    forced = bool(a.mem_check or a.time_steps)
    if forced:
        GATE.stage = len(ns) - 1
    log = open(os.path.join(a.out, "log.jsonl"), "a", newline="\n"); t0 = time.time() - t_prev; acc = []; gn_log = []; tt0 = time.time()

    def draw(nmax):
        if a.mem_check:                                                     # the longest items of the final stage
            its = sorted(N.train_items(np.random.default_rng(0), 8 * a.bs, nmax, set()), key=lambda it: -len(it["syms"]))[:a.bs]
        else:
            its = N.train_items(rng, a.bs, nmax, EXCL)
        Ks = [a.kmax if forced else min(it["n"] + int(rng_k.integers(0, a.kextra + 1)), a.kmax) for it in its]
        pz = None
        if rng_p is not None and (forced or rng_p.random() < a.pause_p):    # §214 PAUSE-A: one coin per step (forced shapes always augmented)
            pz = [pause_schedule(rng_p, it["n"], a.pause_max) for it in its]
            if not forced:                                                  # K = n + P + U, U the plain draw's own (n + U <= 12, so min() never bound)
                Ks = [k_ + sum(z) for k_, z in zip(Ks, pz)]
            assert all(len(z) <= k_ for z, k_ in zip(pz, Ks)), "K < n + P"
        return its, Ks, pz

    def draw_step(nmax, step):
        """§215: the plain batch is drawn every step (rng, rng_k: FT-CTRL's stream); rng_s's coin then may replace it by the next bs pool records
        (self U from rng_s). Forced shapes: mem_check_self = the deepest self shape; time_steps with a pool alternates self (odd) / plain steps.
        -> its, Ks, pz, is_self, the draws_log record (None when forced or not logged)."""
        its, Ks, pz = draw(nmax)
        if pool is not None and a.mem_check and a.mem_check_self:
            return deep_its, [a.self_kmax] * len(deep_its), None, True, None
        lg = (not forced) and step <= a.draws_log_steps
        if pool is None and not lg:
            return its, Ks, pz, False, None
        dl = {"step": step, "plain_hash": hashlib.sha1(json.dumps([[it["syms"], k_] for it, k_ in zip(its, Ks)]).encode()).hexdigest()} if lg else None
        coin = False if pool is None else ((step % 2 == 1) if a.time_steps else bool(rng_s.random() < a.p_self))
        if not coin:
            if lg:
                dl.update({"coin": False, "n": [it["n"] for it in its], "U": [k_ - it["n"] for it, k_ in zip(its, Ks)]})
            return its, Ks, pz, False, dl
        c0 = selfst["cursor"]
        if c0 + a.bs > len(pool):
            raise RuntimeError("§215: the self pool is exhausted (no reuse allowed)")
        recs = pool[c0:c0 + a.bs]; selfst["cursor"] = c0 + a.bs; selfst["n_self_steps"] += 1
        U = [int(rng_s.integers(0, a.self_kextra + 1)) for _ in recs]
        its = [SD.self_item(r, a.self_kmax) for r in recs]; Ks = [a.self_kmax if forced else min(r["n"] + u, a.self_kmax) for r, u in zip(recs, U)]
        if lg:
            dl.update({"coin": True, "self_n": [r["n"] for r in recs], "U": U})
        return its, Ks, None, True, dl
    last = step0 + a.time_steps - 1 if a.time_steps else (a.mem_check_steps if a.mem_check else (min(a.steps, a.stop_at) if a.stop_at else a.steps))
    draws_log = []; tself = [0.0, 0, 0.0, 0]
    if a.resume and os.path.exists(os.path.join(a.out, "draws_log.json")):
        draws_log = [d for d in json.load(open(os.path.join(a.out, "draws_log.json"))) if d["step"] < step0]                                 # §215: (self time, self steps, plain time, plain steps) for G-TIME
    if dev.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(dev)
    for step in range(step0, last + 1):
        lr = lr_at(a.lr, {}, step, a.steps)
        for gr in opt.param_groups:
            gr["lr"] = lr
        m.c_anneal.fill_(c_at(step, ch0, ch1) if m.anneals else 1e4); tfp = tf_at(step, tf0, tf1) if a.mct else 0.0   # §213 G-D4: the capability flag
        ts0 = time.time(); m.train(); its, Ks, pz, is_self, dl = draw_step(GATE.n_max(), step); opt.zero_grad(set_to_none=True); tot = 0.0
        mic = a.self_micro if is_self else a.micro; ch = len(its) // mic; sts = []
        if dl is not None:
            draws_log.append(dl)
        m.gather_log = [] if (a.mode == "pa" and step % a.log_every == 0) else None
        slog = a.mode == "slva" and step % a.log_every == 0; sm_ = {"chg": [0.0, 0], "copy": [0.0, 0]}
        for k_ in range(mic):
            if slog:
                m.slva_log = []
            L, st_, _ = vb_loss(m, its[k_ * ch:(k_ + 1) * ch], Ks[k_ * ch:(k_ + 1) * ch], dev, ac, a.rec_w, tfp, rng_tf, a.mu, pz=pz[k_ * ch:(k_ + 1) * ch] if pz else None)
            (L / mic).backward(); tot += float(L.item()) / mic; st_["is_self"] = is_self; sts.append(st_)
            if slog:                                                        # §213 G-D4: SLVA off-newest mass on changed / copy slots
                from boxes.slva import mass_stats
                ms_ = mass_stats(m, its[k_ * ch:(k_ + 1) * ch], Ks[k_ * ch:(k_ + 1) * ch], dev, pz=pz[k_ * ch:(k_ + 1) * ch] if pz else None)
                for kk in sm_:
                    sm_[kk][0] += ms_[kk][0]; sm_[kk][1] += ms_[kk][1]
        if slog:
            m.slva_log = None; sg_ = [p_.grad for p_ in m.slva_params() if p_.grad is not None]
            slva_gn = float(torch.stack([g_.float().pow(2).sum() for g_ in sg_]).sum().sqrt()) if sg_ else 0.0
        if a.mode == "pa" and step % a.log_every == 0:                    # note 211-3 gate: the gather's parameter gradient norm (pre-clip)
            gg_ = [p_.grad for n_, p_ in m.named_parameters() if n_ in ("gq_w", "gq_b", "gk_w", "gk_b", "gln_w", "gln_b", "gamma_q") and p_.grad is not None]
            gather_gn = float(torch.stack([g_.float().pow(2).sum() for g_ in gg_]).sum().sqrt()) if gg_ else 0.0
        gn = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); gn_log.append(float(gn))
        ans = float(np.mean([s_["answer_ce"] for s_ in sts])); rec = float(np.mean([s_["rec_ce"] for s_ in sts]))
        if not (math.isfinite(tot) and bool(torch.isfinite(gn))):
            n_nonfinite += 1; opt.zero_grad(set_to_none=True); log.write(json.dumps({"step": step, "nonfinite": 1}) + "\n"); log.flush(); ans_g = None
        else:
            opt.step(); acc.append((tot, ans, rec, sts)); ans_g = ans
        if dev.startswith("cuda"):
            torch.cuda.synchronize(dev)
        if pool is not None:
            tself[0 if is_self else 2] += time.time() - ts0; tself[1 if is_self else 3] += 1
        adv = None if (forced or is_self) else GATE.update(step, ans_g)   # §215: no self CurGate
        if adv is not None:
            log.write(json.dumps({"step": step, "cur_stage": adv[0], "n_max_train": GATE.n_max(), "reason": adv[1], "bin_mean": adv[2]}) + "\n"); log.flush()
        if step % a.log_every == 0 or step == last:
            rec_ = {"step": step, "loss": float(np.mean([v[0] for v in acc])) if acc else None, "sec": round(time.time() - t0),
                    "answer_ce": float(np.mean([v[1] for v in acc])) if acc else None, "rec_ce": float(np.mean([v[2] for v in acc])) if acc else None,
                    "n_max_train": GATE.n_max(), "cur_stage": GATE.stage, "lr": lr, "c": float(m.c_anneal), "gn_preclip": gn_log[-1],
                    "clip_frac_100": float(np.mean([v > 1.0 for v in gn_log[-100:]]))}
            if a.mct:
                ss = [s_ for v in acc for s_ in v[3]]; tot_ = sum(s_["tf_tot"] for s_ in ss); sl = sum(s_["mct_slots"] for s_ in ss); rl = [x_ for s_ in ss for x_ in s_["mct_rel"]]
                rec_.update({"tf_p": tfp, "tf_frac": (sum(s_["tf_n"] for s_ in ss) / tot_) if tot_ else None, "mct_multi_frac": (sum(s_["mct_multi"] for s_ in ss) / sl) if sl else None,
                             "mct_rel_mean": float(np.mean(rl)) if rl else None})
            if pool is not None:                                            # §215 fields
                sv = [v for v in acc if v[3] and v[3][0].get("is_self")]; pv = [v for v in acc if v[3] and not v[3][0].get("is_self")]
                rec_.update({"self_frac": len(sv) / len(acc) if acc else None, "self_answer_ce": float(np.mean([v[1] for v in sv])) if sv else None,
                             "self_rec_ce": float(np.mean([v[2] for v in sv])) if sv else None, "plain_answer_ce": float(np.mean([v[1] for v in pv])) if pv else None,
                             "pool_cursor": selfst["cursor"], "n_self_steps": selfst["n_self_steps"], "self_reuse": selfst["self_reuse"]})
            if rng_p is not None:                                           # §214: share of paused items over the log window
                rec_["pause_item_frac"] = float(np.mean([s_.get("n_paused", 0) / max(1, ch) for v in acc for s_ in v[3]])) if acc else None
            if slog:
                rec_.update({"slva_offnewest_chg": sm_["chg"][0] / sm_["chg"][1] if sm_["chg"][1] else None,
                             "slva_offnewest_copy": sm_["copy"][0] / sm_["copy"][1] if sm_["copy"][1] else None, "slva_gn": slva_gn})
            if m.gather_log:
                rec_["carry_mass"] = float(torch.cat([v_ for _, v_ in m.gather_log]).mean()); rec_["gather_offself"] = 1.0 - rec_["carry_mass"]; rec_["gather_gn"] = gather_gn
            log.write(json.dumps(rec_) + "\n"); log.flush(); print(rec_, flush=True); acc = []
        m.gather_log = None
        if step in KEEP:
            torch.save({"model": m.state_dict(), "step": step}, os.path.join(a.out, "ckpt_model_%d.pt" % step))
        if a.ckpt_every and step % a.ckpt_every == 0 and step < a.steps and not forced:
            cp = os.path.join(a.out, "ckpt_last.pt")
            if wpe0 is not None:                                            # §213 G-D3: D3-F's table bytes == pretrained at every checkpoint
                from boxes.slva import wpe_sha
                assert wpe_sha(m) == wpe0, "D3-F: the frozen table changed"; log.write(json.dumps({"step": step, "wpe_frozen_ok": True}) + "\n"); log.flush()
            if os.path.exists(cp):
                os.replace(cp, os.path.join(a.out, "ckpt_prev.pt"))
            torch.save({"model": m.state_dict(), "opt": opt.state_dict(), "np_rng": rng.bit_generator.state, "rng_k": rng_k.bit_generator.state, "rng_tf": rng_tf.bit_generator.state,
                        "torch_rng": torch.get_rng_state(), "gate": GATE.state(), "step": step, "elapsed": time.time() - t0, "n_nonfinite": n_nonfinite, "calibration": cal,
                        **({"rng_p": rng_p.bit_generator.state} if rng_p is not None else {}),
                        **({"rng_s": rng_s.bit_generator.state, "self_state": dict(selfst)} if rng_s is not None else {})}, cp)
            if draws_log:
                json.dump(draws_log, open(os.path.join(a.out, "draws_log.json"), "w", newline="\n"))
    if a.mem_check:
        pk_train = torch.cuda.max_memory_allocated(dev) / 2 ** 30; opt.zero_grad(set_to_none=True); torch.cuda.reset_peak_memory_stats(dev); m.eval()
        ev = sorted([it for it in TE if it["n"] == 16], key=lambda it: -len(it["syms"]))[:a.eval_bs]; xe, pe, le = batch_x(ev, dev)
        with torch.no_grad(), ac():
            m.rollout(xe, pe, 24, coda_rows={24: torch.arange(len(ev), device=dev)})
        pk_eval = torch.cuda.max_memory_allocated(dev) / 2 ** 30
        mj = {"peak_gib_train_step": pk_train, "peak_gib_eval_n16_k24": pk_eval, "bs": a.bs, "micro": a.micro, "eval_bs": a.eval_bs, "gpu": info["train_gpu"], "amp": str(amp)}
        for f_ in ("mem_check.json", "eval.json"):                         # eval.json: the queue / drain line-done marker (§213 queue line 1, as --time_steps)
            json.dump(mj, open(os.path.join(a.out, f_), "w", newline="\n"), indent=1)
        print("MEMCHECK", pk_train, pk_eval, flush=True); return
    if draws_log:
        json.dump(draws_log, open(os.path.join(a.out, "draws_log.json"), "w", newline="\n"))
    if a.time_steps:
        sps = (time.time() - tt0) / a.time_steps
        if pool is not None:                                                # §215 G-S10: the p_self-weighted mix of the forced self / plain step times
            sps_self = tself[0] / max(1, tself[1]); sps_plain = tself[2] / max(1, tself[3]); sps = a.p_self * sps_self + (1 - a.p_self) * sps_plain
        tj = {"mode": a.mode, "mct": a.mct, "steps_timed": a.time_steps, "sec_per_step": sps, "est_hours": sps * a.steps / 3600, "gpu": info["train_gpu"], "micro": a.micro,
              **({"sec_per_step_self": tself[0] / max(1, tself[1]), "sec_per_step_plain": tself[2] / max(1, tself[3]), "self_micro": a.self_micro, "p_self": a.p_self} if pool is not None else {}),
              "forced": "final stage n <= 8, K = 12 every item" + ("; with a pool: odd steps self (pool order, K self_kmax), even steps plain; sec_per_step = p_self-weighted mix" if pool is not None else "")}
        for f_ in ("timing.json", "eval.json"):
            json.dump(tj, open(os.path.join(a.out, f_), "w", newline="\n"), indent=1)
        print("TIMING", sps, flush=True); return
    if a.stop_at and last < a.steps:
        print("STOPPED_AT", last, flush=True); return
    train_sec = round(time.time() - t0); torch.save(m.state_dict(), os.path.join(a.out, "model.pt"))
    for ck_ in ("ckpt_last.pt", "ckpt_prev.pt"):                            # Audit 06:2x 10-16: right after model.pt, before the descriptive eval (a late eval
        if os.path.exists(os.path.join(a.out, ck_)):                        # failure left 3 GB of resume state for the drain to carry)
            os.remove(os.path.join(a.out, ck_))
    opt.zero_grad(set_to_none=True); opt.state.clear(); m.eval(); t1 = time.time()
    if dev.startswith("cuda"):
        torch.cuda.empty_cache()
    Nn = np.array([it["n"] for it in TE]); Y = np.array([it["t"] for it in TE]); A = np.zeros(len(TE), dtype=np.int64)   # host-side, descriptive: acc at K' = n + 2
    for n in range(1, 17):
        idx = np.nonzero(Nn == n)[0]
        for i in range(0, len(idx), a.eval_bs):
            ch_ = [TE[j] for j in idx[i:i + a.eval_bs]]; xe, pe, le = batch_x(ch_, dev)
            with torch.no_grad(), ac():
                r_ = m.rollout(xe, pe, n + 2, coda_rows={n + 2: torch.arange(len(ch_), device=dev)})
                A[idx[i:i + a.eval_bs]] = m.answer_logits(r_["coda"][n + 2][1], le)[:, N.NUM_IDS].argmax(-1).cpu().numpy()
    if wpe0 is not None:
        from boxes.slva import wpe_sha
        assert wpe_sha(m) == wpe0, "D3-F: the frozen table changed"
    res = {"task": "nested", "arm": "vbank", "mode": a.mode, "arch": a.arch, "table": a.table, "wpe_frozen_ok": (wpe0 is not None) or None, "mct": a.mct, "seed": a.seed, "acc_n2_host": {str(n): float((A == Y)[Nn == n].mean()) for n in range(1, 17)},
           "train_sec": train_sec, "n_nonfinite": n_nonfinite, "gate_hist": GATE.hist, "eval_dtype": str(amp), **info, "eval_sec": round(time.time() - t1),
           "note": "descriptive host eval; the registered readings come from the local fp32 eval_vb.json (note 211-1)"}
    json.dump(res, open(os.path.join(a.out, "eval.json"), "w", newline="\n"), indent=1)
    print("EVAL", json.dumps(res["acc_n2_host"]), flush=True); print("DONE", a.out, flush=True)

if __name__ == "__main__":
    import json
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "g1":
        print(json.dumps(gate_g1()))
    elif len(sys.argv) > 1 and sys.argv[1] == "rerender":
        print(json.dumps(gate_rerender()))
    else:
        main()
