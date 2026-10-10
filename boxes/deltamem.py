"""THEORY §147 DELTA-STATE (b1d7741c / note 147-1): a linear-recurrent associative memory as the ONLY cross-segment channel of GPT-2, on §140's
digit-pair multiplication harness (boxes/multpair.py --carrier deltamem).
    branch after block l_m (6 blocks done; entering block 6, the core.py inject convention): x <- x + fade * W_o [additive heads || delta heads](LN_m(x))
    6 additive heads: S <- S + phi(k) v^T (phi = elu + 1; no decay, no erase), read phi(q)^T S
    6 gated-delta heads: S <- alpha S (I - beta k k^T) + beta v k^T (k, q L2-normalised; beta = 2 sigmoid(.) in (0, 2); alpha = sigmoid(.), bias +5), read S q
    note 147-2: per-type learnable read gains g = exp(s) (log-parametrised, Audit: a relative Adam step), s in the branch's AdamW group; g_del = 1 (s 0), g_add = the median ||read_del|| / ||read_add||
    over a probe batch's hand-offs at step 0 (W_o = 0 then, so still function-preserving)
q / k / v (and LN_m) are initialised from block l_m's own ln_1 / attn.c_attn; W_o = 0 (fade-in buffer, 500 steps): the pretrained function is
unchanged at step 0 (max |delta logits| = 0). Segments [EOT, a_i, b_j, column flag] (pairs) / [EOT, " ="] (flush): each a fresh GPT-2 pass,
attention inside the segment only; the memory state (B, 12, 64, 64) is carried to the next segment and is the only cross-segment channel.
Emission: at a column's last pair, GPT-2's own LM head at the segment's last position predicts p_k (CE over the 10 digit tokens' vocabulary
rows: full-vocabulary CE, as text)."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

H_ADD, H_DEL, DH = 6, 6, 64
FAST_ADD = False                                                            # Audit 2026-10-09: R-mode additive writes as one sum over the pass (opt-in per run)


class DeltaMem(nn.Module):
    def __init__(self, g, layer=6, write=None, h_add=H_ADD, h_del=H_DEL, add_norm=None):
        """§178: the width D comes from the trunk (768 for GPT-2) and the head counts are arguments (6 + 6 by default; (h_add + h_del) x DH = D).
        §185 (2a1aac15) add_norm: None (unchanged) | "den" (additive reads / (phi(q)^T z), z = sum phi(k) carried as one extra state block) |
        "rms" (additive reads RMS-normalised per head, learned per-head gain init 1)."""
        super().__init__()
        assert add_norm in (None, "den", "rms"); self.add_norm = add_norm
        self.layer = layer; pre = "h.%d." % layer; dev = g.w("wte.weight").device; D = g.w("wte.weight").shape[1]
        assert (h_add + h_del) * DH == D, "DeltaMem: (h_add + h_del) x %d must equal the width %d" % (DH, D)
        self.D, self.h_add, self.h_del = D, h_add, h_del
        self.ln = nn.LayerNorm(D).to(dev)
        with torch.no_grad():
            self.ln.weight.copy_(g.w(pre + "ln_1.weight").float()); self.ln.bias.copy_(g.w(pre + "ln_1.bias").float())
        self.qkv = nn.Linear(D, 3 * D).to(dev)
        with torch.no_grad():                                               # core.py stores Conv1D-style (in, out) weights
            self.qkv.weight.copy_(g.w(pre + "attn.c_attn.weight").float().T); self.qkv.bias.copy_(g.w(pre + "attn.c_attn.bias").float())
        self.gate = nn.Linear(D, 2 * h_del).to(dev)                         # per delta head: alpha logit, beta logit
        with torch.no_grad():
            self.gate.weight.zero_(); self.gate.bias[:h_del].fill_(5.0); self.gate.bias[h_del:].zero_()   # alpha ~ 0.993, beta = 1
        self.wo = nn.Linear(D, D).to(dev)
        with torch.no_grad():
            self.wo.weight.zero_(); self.wo.bias.zero_()
        self.register_buffer("fade", torch.zeros((), device=dev)); self.ablate = None   # "add" | "delta": that head type's read zeroed
        self.s_add = nn.Parameter(torch.zeros((), device=dev)); self.s_del = nn.Parameter(torch.zeros((), device=dev))   # note 147-2: gains g = exp(s) (Audit: log-param.)
        if add_norm == "rms":                                               # §185 rms: TransNormer's NormAttention gain, registered only under rms
            self.add_gain = nn.Parameter(torch.ones(h_add, device=dev))
        self.alpha_rec = None                                               # note 147-3: list -> per segment, mean alpha per delta head
        self.norms = None                                                   # list -> per segment (last token): (||g_del read_del||, ||g_add read_add||) per row
        self.write_mode = write                                             # §155 DELTA-R: None (§147) | "top" | "block9"
        if write is not None:                                               # LN_w on the write source (init: block 11's ln_1); gamma mixes the write input
            self.ln_w = nn.LayerNorm(D).to(dev); lw = "h.%d.ln_1." % (g.L - 1)  # the last block's ln_1 (block 11 for GPT-2)
            with torch.no_grad():
                self.ln_w.weight.copy_(g.w(lw + "weight").float()); self.ln_w.bias.copy_(g.w(lw + "bias").float())
            self.register_buffer("gamma", torch.zeros((), device=dev)); self.wstat = None   # wstat: list -> per pass (||LN_w(h_w)||, ||LN_6(h_6)||) medians

    def zero_state(self, B, dev):
        return torch.zeros(B, self.h_add + self.h_del + (1 if getattr(self, "add_norm", None) == "den" else 0), DH, DH, device=dev)   # §185 den: + z block

    def forward(self, x, S, rec=None):
        """x (B, T, 768) residual stream after l_m blocks; S (B, 12, DH, DH) carried state. Sequential over T. Returns (x + branch, S_new).
        The branch runs in fp32 with autocast off (exact sums in the additive heads under a bf16 trunk)."""
        with torch.autocast("cuda", enabled=False):
            return self._forward(x, S, rec)

    def _forward(self, x, S, rec=None):
        H_ADD, H_DEL = self.h_add, self.h_del                               # §178: this memory's head counts (6 / 6 by default)
        B, T, D = x.shape; a = self.ln(x.float()); q, k, v = self.qkv(a).split(D, -1)
        q, k, v = (t.view(B, T, H_ADD + H_DEL, DH) for t in (q, k, v)); g_ = self.gate(a); al = torch.sigmoid(g_[..., :H_DEL]); be = 2 * torch.sigmoid(g_[..., H_DEL:])
        Sa, Sd = S[:, :H_ADD].float(), S[:, H_ADD:H_ADD + H_DEL].float(); outs = []; den = self.add_norm == "den"
        z = S[:, H_ADD + H_DEL, :H_ADD].float() if den else None           # §185 den: z = sum phi(k) (B, H_ADD, DH)
        if self.alpha_rec is not None:
            self.alpha_rec.append(al.mean(dim=(0, 1)).detach())
        for t in range(T):
            pk, pq = F.elu(k[:, t, :H_ADD]) + 1, F.elu(q[:, t, :H_ADD]) + 1     # additive: S <- S + phi(k) v^T; read phi(q)^T S
            Sa = Sa + pk[..., :, None] * v[:, t, :H_ADD][..., None, :]
            if den:
                z = z + pk
            oa = add_read(self, pq, Sa, z)
            kd = F.normalize(k[:, t, H_ADD:], dim=-1); qd = F.normalize(q[:, t, H_ADD:], dim=-1); vd = v[:, t, H_ADD:]
            b_ = be[:, t][..., None, None]; Skk = torch.einsum("bhvk,bhk->bhv", Sd, kd)   # S (v x k): S (I - b k k^T) = S - b (S k) k^T
            Sd = al[:, t][..., None, None] * (Sd - b_ * Skk[..., :, None] * kd[..., None, :]) + b_ * vd[..., :, None] * kd[..., None, :]
            od = torch.einsum("bhvk,bhk->bhv", Sd, qd)
            oa = torch.exp(self.s_add) * oa; od = torch.exp(self.s_del) * od   # note 147-2: x <- x + W_o [g_add read_add || g_del read_del], g = exp(s)
            if self.norms is not None and t == T - 1:
                self.norms.append((od.flatten(1).norm(dim=1).detach(), oa.flatten(1).norm(dim=1).detach()))
            if self.ablate == "add":
                oa = torch.zeros_like(oa)
            if self.ablate == "delta":
                od = torch.zeros_like(od)
            outs.append(torch.cat([oa, od], 1).reshape(B, D))
            if rec is not None and t == T - 1:
                rec.append(oa.reshape(B, -1).detach())
        o = torch.stack(outs, 1)
        return x + (self.fade * self.wo(o)).to(x.dtype), pack_state(self, Sa, Sd, z)


def state_z(mem, S):
    """§185 den: z (B, H_ADD, DH) from the state's extra block (rows 0..H_ADD-1 of block H_ADD + H_DEL); None otherwise."""
    if getattr(mem, "add_norm", None) != "den":
        return None
    return S[:, mem.h_add + mem.h_del, :mem.h_add].float()


def pack_state(mem, Sa, Sd, z=None):
    """(Sa, Sd) -> the carried state; §185 den appends the z block (z in rows 0..H_ADD-1, rows H_ADD..DH-1 exactly 0)."""
    if getattr(mem, "add_norm", None) != "den":
        return torch.cat([Sa, Sd], 1)
    zb = torch.cat([z, torch.zeros(z.shape[0], DH - z.shape[1], DH, device=z.device, dtype=z.dtype)], 1)[:, None]
    return torch.cat([Sa, Sd, zb], 1)


def add_read(mem, pq, Sa, z=None):
    """the additive heads' read phi(q)^T S_add (pq (B, T, H, DH) or (B, H, DH)), before the gain exp(s_add). §185: den divides by phi(q)^T z
    (+ 1e-6) per head; rms normalises each head's read to unit RMS over DH and multiplies a learned per-head gain (init 1). None: unchanged."""
    tok = pq.dim() == 4; oa = torch.einsum("bthk,bhkv->bthv" if tok else "bhk,bhkv->bhv", pq, Sa); md = getattr(mem, "add_norm", None)
    if md == "den":
        oa = oa / (torch.einsum("bthk,bhk->bth" if tok else "bhk,bhk->bh", pq, z)[..., None] + 1e-6)
    elif md == "rms":
        oa = oa * torch.rsqrt(oa.pow(2).mean(-1, keepdim=True) + 1e-6) * mem.add_gain[:, None]
    return oa


def read_fixed(mem, x, S, rec=None):
    """§155: the read on the pass-start state S at every token (no write); the same gains / ablation / fade / W_o as DeltaMem.forward.
    rec / mem.norms record the last token's reads (as forward)."""
    H_ADD, H_DEL = mem.h_add, mem.h_del                                     # §178: this memory's head counts (6 / 6 by default)
    with torch.autocast("cuda", enabled=False):
        B, T, D = x.shape; a = mem.ln(x.float()); q = mem.qkv(a)[..., :D].view(B, T, H_ADD + H_DEL, DH)
        oa = torch.exp(mem.s_add) * add_read(mem, F.elu(q[:, :, :H_ADD]) + 1, S[:, :H_ADD].float(), state_z(mem, S))
        od = torch.exp(mem.s_del) * torch.einsum("bhvk,bthk->bthv", S[:, H_ADD:H_ADD + H_DEL].float(), F.normalize(q[:, :, H_ADD:], dim=-1))
        if mem.norms is not None:
            mem.norms.append((od[:, -1].flatten(1).norm(dim=1).detach(), oa[:, -1].flatten(1).norm(dim=1).detach()))
        if mem.ablate == "add":
            oa = torch.zeros_like(oa)
        if mem.ablate == "delta":
            od = torch.zeros_like(od)
        if rec is not None:
            rec.append(oa[:, -1].reshape(B, -1).detach())
        return x + (mem.fade * mem.wo(torch.cat([oa, od], 2).reshape(B, T, D))).to(x.dtype)


def write_seq(mem, h6, hw, S, mask=None):
    """§155: S_{t+1} from the pass's writes, sequential over tokens as DeltaMem.forward, with k / v / alpha / beta from
    u = (1 - gamma) LN_6(h_6) + gamma LN_w(h_w) (one shared set of write projections: qkv's k / v rows and the gate)."""
    H_ADD, H_DEL = mem.h_add, mem.h_del                                     # §178: this memory's head counts (6 / 6 by default)
    with torch.autocast("cuda", enabled=False):
        B, T, D = h6.shape; a6 = mem.ln(h6.float()); aw = mem.ln_w(hw.float()); gm = mem.gamma
        if mem.wstat is not None:
            mem.wstat.append((float(aw.norm(dim=-1).median()), float(a6.norm(dim=-1).median())))
        u = (1 - gm) * a6 + gm * aw; _, k, v = mem.qkv(u).split(D, -1); k, v = (t_.view(B, T, H_ADD + H_DEL, DH) for t_ in (k, v))
        g_ = mem.gate(u); al = torch.sigmoid(g_[..., :H_DEL]); be = 2 * torch.sigmoid(g_[..., H_DEL:])
        if mem.alpha_rec is not None:
            mem.alpha_rec.append(al.mean(dim=(0, 1)).detach())
        Sa, Sd = S[:, :H_ADD].float(), S[:, H_ADD:H_ADD + H_DEL].float(); den = getattr(mem, "add_norm", None) == "den"; z = state_z(mem, S)
        if den:
            assert S.shape[1] == H_ADD + H_DEL + 1, "§185 den: the state carries one z block"
        if FAST_ADD:                                                        # reads see only the pass-start S, so the additive writes commute: one einsum
            pk = F.elu(k[:, :, :H_ADD]) + 1                                 # (fp32 summation order differs from the loop; gate: scratch/gate_fastadd.py)
            if mask is not None:
                pk = pk * mask[:, :, None, None].to(pk.dtype)               # padding tokens contribute nothing
            Sa_fast = Sa + torch.einsum("bthk,bthv->bhkv", pk, v[:, :, :H_ADD])
            if den:                                                         # §185: z <- z + sum_t phi(k_t) (masked), the same masked sum as S_add
                z_fast = z + pk.sum(1)
        for t in range(T):
            Sa_ = Sa if FAST_ADD else Sa + (F.elu(k[:, t, :H_ADD]) + 1)[..., :, None] * v[:, t, :H_ADD][..., None, :]
            z_ = z if (FAST_ADD or not den) else z + (F.elu(k[:, t, :H_ADD]) + 1)
            kd = F.normalize(k[:, t, H_ADD:], dim=-1); vd = v[:, t, H_ADD:]; b_ = be[:, t][..., None, None]
            Skk = torch.einsum("bhvk,bhk->bhv", Sd, kd)
            Sd_ = al[:, t][..., None, None] * (Sd - b_ * Skk[..., :, None] * kd[..., None, :]) + b_ * vd[..., :, None] * kd[..., None, :]
            if mask is None:
                Sa, Sd = Sa_, Sd_
                if den:
                    z = z_
            else:                                                           # §170: padding tokens (mask False) leave the state unchanged
                m_ = mask[:, t][:, None, None, None]; Sa = torch.where(m_, Sa_, Sa); Sd = torch.where(m_, Sd_, Sd)
                if den and not FAST_ADD:
                    z = torch.where(mask[:, t][:, None, None], z_, z)
        if FAST_ADD:
            Sa = Sa_fast
            if den:
                z = z_fast
        return pack_state(mem, Sa, Sd, z)


def _block(g, i, h, hd):
    B, T, D = h.shape; pre = f"h.{i}."
    a = F.layer_norm(h, (D,), g.w(pre + "ln_1.weight"), g.w(pre + "ln_1.bias"))
    q, k, v = (a @ g.w(pre + "attn.c_attn.weight") + g.w(pre + "attn.c_attn.bias")).split(D, -1)
    q, k, v = (t.view(B, T, g.H, hd).transpose(1, 2) for t in (q, k, v))
    o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    h = h + o.transpose(1, 2).reshape(B, T, D) @ g.w(pre + "attn.c_proj.weight") + g.w(pre + "attn.c_proj.bias")
    a = F.layer_norm(h, (D,), g.w(pre + "ln_2.weight"), g.w(pre + "ln_2.bias"))
    m = F.gelu(a @ g.w(pre + "mlp.c_fc.weight") + g.w(pre + "mlp.c_fc.bias"), approximate="tanh")
    return h + m @ g.w(pre + "mlp.c_proj.weight") + g.w(pre + "mlp.c_proj.bias")


def gpt_forward_mem_r(g, x, mem, S, rec=None, cap=None, mask=None):
    """§155 DELTA-R pass: the read (pass-start S) entering block mem.layer, the 12 blocks, then the writes from
    u = (1 - gamma) LN_6(h_6) + gamma LN_w(h_w) (h_w = the residual after block 11, pre-ln_f, or block 9's output). cap: dict for probes."""
    B, T, D = x.shape; hd = D // g.H; h = x + g.w("wpe.weight")[:T][None]; h6 = hw = None
    for i in range(g.L):
        if i == mem.layer:
            h6 = h; h = read_fixed(mem, h, S, rec)
            if cap is not None:
                cap["h6"] = h6.detach(); cap["read"] = (h - h6).detach()
        if mem.write_mode == "block9" and i == 10:
            hw = h
        h = _block(g, i, h, hd)
    hw = h if mem.write_mode == "top" else hw
    if cap is not None:
        cap["hw"] = hw.detach()
    return F.layer_norm(h, (D,), g.w("ln_f.weight"), g.w("ln_f.bias")), write_seq(mem, h6, hw, S, mask)


def gpt_forward_mem(g, x, mem, S, rec=None, mask=None):
    """core.GPT.forward (no past / inject / capture) with the memory branch entering block mem.layer. x (B, T, D) input embeddings."""
    if getattr(mem, "write_mode", None) is not None:                         # §155 DELTA-R
        return gpt_forward_mem_r(g, x, mem, S, rec, mask=mask)
    assert mask is None, "a write mask needs the R write path (§155 DELTA-R)"
    B, T, D = x.shape; hd = D // g.H; h = x + g.w("wpe.weight")[:T][None]
    for i in range(g.L):
        if i == mem.layer:
            h, S = mem(h, S, rec)
        pre = f"h.{i}."
        a = F.layer_norm(h, (D,), g.w(pre + "ln_1.weight"), g.w(pre + "ln_1.bias"))
        q, k, v = (a @ g.w(pre + "attn.c_attn.weight") + g.w(pre + "attn.c_attn.bias")).split(D, -1)
        q, k, v = (t.view(B, T, g.H, hd).transpose(1, 2) for t in (q, k, v))
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        h = h + o.transpose(1, 2).reshape(B, T, D) @ g.w(pre + "attn.c_proj.weight") + g.w(pre + "attn.c_proj.bias")
        a = F.layer_norm(h, (D,), g.w(pre + "ln_2.weight"), g.w(pre + "ln_2.bias"))
        m = F.gelu(a @ g.w(pre + "mlp.c_fc.weight") + g.w(pre + "mlp.c_fc.bias"), approximate="tanh")
        h = h + m @ g.w(pre + "mlp.c_proj.weight") + g.w(pre + "mlp.c_proj.bias")
    return F.layer_norm(h, (D,), g.w("ln_f.weight"), g.w("ln_f.bias")), S
