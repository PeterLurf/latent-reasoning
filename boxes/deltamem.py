"""THEORY §147 DELTA-STATE (b1d7741c / note 147-1): a linear-recurrent associative memory as the ONLY cross-segment channel of GPT-2, on §140's
digit-pair multiplication harness (boxes/multpair.py --carrier deltamem).
    branch after block l_m (6 blocks done; entering block 6, the core.py inject convention): x <- x + fade * W_o [additive heads || delta heads](LN_m(x))
    6 additive heads: S <- S + phi(k) v^T (phi = elu + 1; no decay, no erase), read phi(q)^T S
    6 gated-delta heads: S <- alpha S (I - beta k k^T) + beta v k^T (k, q L2-normalised; beta = 2 sigmoid(.) in (0, 2); alpha = sigmoid(.), bias +5), read S q
    note 147-2: per-type learnable read gains g_add / g_del (the branch's AdamW group); g_del = 1, g_add = the median ||read_del|| / ||read_add||
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


class DeltaMem(nn.Module):
    def __init__(self, g, layer=6):
        super().__init__()
        self.layer = layer; pre = "h.%d." % layer; dev = g.w("wte.weight").device
        self.ln = nn.LayerNorm(768).to(dev)
        with torch.no_grad():
            self.ln.weight.copy_(g.w(pre + "ln_1.weight").float()); self.ln.bias.copy_(g.w(pre + "ln_1.bias").float())
        self.qkv = nn.Linear(768, 3 * 768).to(dev)
        with torch.no_grad():                                               # core.py stores Conv1D-style (in, out) weights
            self.qkv.weight.copy_(g.w(pre + "attn.c_attn.weight").float().T); self.qkv.bias.copy_(g.w(pre + "attn.c_attn.bias").float())
        self.gate = nn.Linear(768, 2 * H_DEL).to(dev)                       # per delta head: alpha logit, beta logit
        with torch.no_grad():
            self.gate.weight.zero_(); self.gate.bias[:H_DEL].fill_(5.0); self.gate.bias[H_DEL:].zero_()   # alpha ~ 0.993, beta = 1
        self.wo = nn.Linear(768, 768).to(dev)
        with torch.no_grad():
            self.wo.weight.zero_(); self.wo.bias.zero_()
        self.register_buffer("fade", torch.zeros((), device=dev)); self.ablate = None   # "add" | "delta": that head type's read zeroed
        self.g_add = nn.Parameter(torch.ones((), device=dev)); self.g_del = nn.Parameter(torch.ones((), device=dev))   # note 147-2: per-type read gains
        self.norms = None                                                   # list -> per segment (last token): (||g_del read_del||, ||g_add read_add||) per row

    def zero_state(self, B, dev):
        return torch.zeros(B, H_ADD + H_DEL, DH, DH, device=dev)

    def forward(self, x, S, rec=None):
        """x (B, T, 768) residual stream after l_m blocks; S (B, 12, DH, DH) carried state. Sequential over T. Returns (x + branch, S_new).
        The branch runs in fp32 with autocast off (exact sums in the additive heads under a bf16 trunk)."""
        with torch.autocast("cuda", enabled=False):
            return self._forward(x, S, rec)

    def _forward(self, x, S, rec=None):
        B, T, D = x.shape; a = self.ln(x.float()); q, k, v = self.qkv(a).split(D, -1)
        q, k, v = (t.view(B, T, H_ADD + H_DEL, DH) for t in (q, k, v)); g_ = self.gate(a); al = torch.sigmoid(g_[..., :H_DEL]); be = 2 * torch.sigmoid(g_[..., H_DEL:])
        Sa, Sd = S[:, :H_ADD].float(), S[:, H_ADD:].float(); outs = []
        for t in range(T):
            pk, pq = F.elu(k[:, t, :H_ADD]) + 1, F.elu(q[:, t, :H_ADD]) + 1     # additive: S <- S + phi(k) v^T; read phi(q)^T S
            Sa = Sa + pk[..., :, None] * v[:, t, :H_ADD][..., None, :]
            oa = torch.einsum("bhk,bhkv->bhv", pq, Sa)
            kd = F.normalize(k[:, t, H_ADD:], dim=-1); qd = F.normalize(q[:, t, H_ADD:], dim=-1); vd = v[:, t, H_ADD:]
            b_ = be[:, t][..., None, None]; Skk = torch.einsum("bhvk,bhk->bhv", Sd, kd)   # S (v x k): S (I - b k k^T) = S - b (S k) k^T
            Sd = al[:, t][..., None, None] * (Sd - b_ * Skk[..., :, None] * kd[..., None, :]) + b_ * vd[..., :, None] * kd[..., None, :]
            od = torch.einsum("bhvk,bhk->bhv", Sd, qd)
            oa = self.g_add * oa; od = self.g_del * od                      # note 147-2: x <- x + W_o [g_add read_add || g_del read_del]
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
        return x + (self.fade * self.wo(o)).to(x.dtype), torch.cat([Sa, Sd], 1)


def gpt_forward_mem(g, x, mem, S, rec=None):
    """core.GPT.forward (no past / inject / capture) with the memory branch entering block mem.layer. x (B, T, D) input embeddings."""
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
