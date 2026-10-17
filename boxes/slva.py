"""§213 D3 (registered f08b14c0; docs/research_bottleneck_2026-10-16/D3_v1.md §2.1, §6 S0-4): SLOT-LOCAL VERSION ATTENTION on VERSION-BANK.
SLVA subclasses vbank.VBankN and changes only the pass input. The prelude, core, coda, bank, R, losses and curriculum are VBankN's; the pa / carry
paths are untouched. At pass t >= 1, per slot j and head h (12 heads x 64-d slices of N(V)):
    U_t[j]_h = sum_{s<t} alpha^h_{j,s} N(V_s)[j]_h,
    alpha^h_{j,s} = softmax_s( <W_q LN_q(N(V_{t-1})[j]), W_k LN_k(V_s[j])>_h / 8 + beta0_h 1[s = 0] + gamma_h (t-1-s) 1[s >= 1] - c 1[s != t-1] ).
Content queries from the slot's newest version, keys = the same slot's versions; no position query, no wpe input. The age term acts on s >= 1 only;
own-slot keys are never pad-masked (padded slots read finite garbage, masked downstream). Init: W_q / W_k / LN_q / LN_k from block 6 (c_attn q / k,
ln_1), gamma = alibi_init(H, H) (-ALiBi slope, as N2's gather), beta0 = 0. c = the c_anneal buffer (trainer: c_at(step, 1000, 8000) via the
`anneals` capability flag). Keys, logits, softmax and read run with autocast disabled (fp32; fp64 in the gates).
table: "train" (D3-T: wpe trains as in N1 / N2) | "frozen" (D3-F: the trainer sets wpe.requires_grad False before building the optimizer and
checksums its bytes at every checkpoint).
Eval hooks (G-D5): keep (ver_keep), off_v0 (v0_off_after), age_floor, gather_splice, gather_log, gather_split_log (cross-slot mass 0 by construction),
gather_hist_ablate ("self" = own-slot versions 1..t-2 masked; "other" = no-op, SLVA has no cross-slot keys) and FOREIGN's view (vbank.rollout passes
the viewed key / value lists) are implemented; gather_qmod is undefined (no position query) and asserts."""
import hashlib
import math

import torch
import torch.nn.functional as F

import boxes.vbank as VB


class SLVA(VB.VBankN):
    def __init__(self, g, table="train"):
        assert table in ("train", "frozen"), table
        super().__init__(g, "carry")                                        # the shared parts; no PA gather parameters (VBankN builds them for "pa" only)
        self.mode = "slva"; self.table = table; D, H = self.D, self.H
        W, b = g.w("h.6.attn.c_attn.weight").detach(), g.w("h.6.attn.c_attn.bias").detach()
        self.slva_wq = torch.nn.Parameter(W[:, :D].clone()); self.slva_bq = torch.nn.Parameter(b[:D].clone())
        self.slva_wk = torch.nn.Parameter(W[:, D:2 * D].clone()); self.slva_bk = torch.nn.Parameter(b[D:2 * D].clone())
        self.slva_ln_q = torch.nn.LayerNorm(D).to(W.device); self.slva_ln_k = torch.nn.LayerNorm(D).to(W.device)
        for ln_ in (self.slva_ln_q, self.slva_ln_k):
            ln_.weight.data.copy_(g.w("h.6.ln_1.weight").detach()); ln_.bias.data.copy_(g.w("h.6.ln_1.bias").detach())
        self.slva_beta0 = torch.nn.Parameter(torch.zeros(H, device=W.device))
        self.slva_gamma = torch.nn.Parameter(VB.alibi_init(H, H).to(W.device))
        self.slva_log = None                                                # trainer diagnostics: a list -> per pass (t, (B, T) newest-version mass, head mean)
        self.slva_view = None                                               # eval-only (§213 swap triple): f(t, gk, NV) -> (gk', NV') for SLVA's read at pass t only
        self.keep_bank_only = False                                         # eval-only (§213 OTHER-ABLATE): ver_keep reaches the core / coda bank but not SLVA

    @property
    def anneals(self):
        return True

    def slva_params(self):
        return [p_ for n_, p_ in self.named_parameters() if n_.startswith("slva_")]

    def gkey(self, V):
        """version s's keys at every slot: W_k LN_k(V_s[j]) -> (B, T, H, hd), fp32 (fp64 in the gates), autocast off."""
        B, T, D = V.shape; dt = torch.float64 if V.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=V.device.type, enabled=False):
            a = F.layer_norm(V.to(dt), (D,), self.slva_ln_k.weight.to(dt), self.slva_ln_k.bias.to(dt))
            return (a @ self.slva_wk.to(dt) + self.slva_bk.to(dt)).view(B, T, self.H, self.hd)

    def gather(self, t, gk, NV, pad, off_v0=False, keep=None):
        """SLVA read for pass t (vbank.rollout's gather slot): gk / NV = the (viewed) lists of version keys / N(V_s), s = 0..; pad is NOT used (own-slot
        keys are never masked; slot_mask's removals do not reach a slot-local read). -> U_t (B, T, D) in NV's dtype."""
        assert self.gather_qmod is None, "gather_qmod is undefined for SLVA (no position query)"
        if self.slva_view is not None:
            gk, NV = self.slva_view(t, gk, NV)
        if self.keep_bank_only:
            keep = None
        B, T, D = NV[0].shape; H, hd = self.H, self.hd; dev = NV[0].device; dt = torch.float64 if NV[0].dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=dev.type, enabled=False):
            qa = F.layer_norm(NV[t - 1].to(dt), (D,), self.slva_ln_q.weight.to(dt), self.slva_ln_q.bias.to(dt))
            q = (qa @ self.slva_wq.to(dt) + self.slva_bq.to(dt)).view(B, T, H, hd)
            K_ = torch.stack([k_.to(dt) for k_ in gk[:t]], 3)                                         # (B, T, H, t, hd)
            lg = torch.einsum("bjhd,bjhsd->bjhs", q, K_) / math.sqrt(hd)                              # (B, T, H, t)
            s = torch.arange(t, device=dev); age = (t - 1 - s).to(dt)
            agev = self.slva_gamma.to(dt)[:, None] * age[None, :]                                     # (H, t)
            if self.age_floor is not None:                                  # eval-only (note 212-3): slope floor on versions 1..t-2
                ap_ = (s >= 1) & (s <= t - 2); agev = torch.where(ap_[None, :], torch.minimum(agev, -self.age_floor * age[None, :]), agev)
            bias = torch.where(s[None, :] == 0, self.slva_beta0.to(dt)[:, None].expand(H, t), agev)  # beta0 on V_0 (no age term), gamma * age on s >= 1
            lg = lg + bias[None, None] - self.c_anneal.to(dt) * (s != t - 1).to(dt)[None, None, None, :]
            valid = torch.ones(t, dtype=torch.bool, device=dev)
            if off_v0:
                valid = valid & (s != 0)
            if keep is not None:
                valid = valid & keep.to(dev)[:t]
            if self.gather_hist_ablate == "self":                           # 212-15 / §213 SELF-ABLATE: own-slot versions 1..t-2
                valid = valid & ~((s >= 1) & (s <= t - 2))
            else:
                assert self.gather_hist_ablate in (None, "other"), self.gather_hist_ablate   # "other": SLVA has no cross-slot keys (no-op)
            assert bool(valid[t - 1]), "SLVA: the newest version must stay readable"
            w = torch.softmax(lg.masked_fill(~valid[None, None, None, :], VB.NEG), -1)              # (B, T, H, t)
            Vst = torch.stack([v_.to(dt).view(B, T, H, hd) for v_ in NV[:t]], 3)                      # (B, T, H, t, hd)
            U = torch.einsum("bjhs,bjhsd->bjhd", w, Vst).reshape(B, T, H * hd)
            newest = w[..., t - 1].mean(2).detach()                                                   # (B, T): newest-version mass, head mean
            if self.gather_log is not None:
                self.gather_log.append((t, (newest * pad).sum(1) / pad.sum(1)))
            if self.gather_split_log is not None:                           # (t, newest, older same-slot, cross-slot = 0)
                mean_ = lambda z: (z * pad).sum(1) / pad.sum(1)  # noqa: E731
                self.gather_split_log.append((t, mean_(newest), mean_(1 - newest), torch.zeros(B, dtype=newest.dtype, device=dev)))
            if self.slva_log is not None:
                self.slva_log.append((t, newest))
        if self.gather_splice is not None:                                  # eval-only (note 212-6): slots j >= J take N(V_{t-1}[j])
            U = torch.cat([U[:, :self.gather_splice], NV[t - 1].to(U.dtype)[:, self.gather_splice:]], 1)
        return U.to(NV[0].dtype)


def wpe_sha(m):
    """sha256 of the position table's bytes (D3-F's frozen-table checksum, G-D3)."""
    w = m.g.p["wpe_weight"].detach().float().cpu().contiguous()
    return hashlib.sha256(w.numpy().tobytes()).hexdigest()


@torch.no_grad()
def mass_stats(m, items, Ks, dev, pz=None):
    """trainer diagnostics (G-D4) from m.slva_log after one micro-batch's vb_loss: summed off-newest SLVA mass and slot counts on changed
    (E_t != E_{t-1}) and copy (unchanged, non-VOID) real slots, t <= K_i. -> {"chg": [sum, n], "copy": [sum, n]}; clears m.slva_log."""
    x, pad, lens = VB.batch_x(items, dev); B, T = x.shape; Kb = int(max(Ks)); E = VB.rec_targets(items, Kb, T, dev); Kt = torch.tensor(Ks, device=dev); out = {"chg": [0.0, 0], "copy": [0.0, 0]}
    if pz is not None and any(z is not None for z in pz):                   # §214 G-P14: the mapped targets E_{rho(t)} on paused items (diagnostic)
        rho = torch.arange(Kb + 1, device=dev)[None, :].repeat(B, 1)
        for b, z in enumerate(pz):
            if z is not None:
                pm = torch.zeros(Kb + 1, dtype=torch.long, device=dev); pm[1:len(z) + 1] = torch.tensor(z, device=dev).long(); rho[b] = rho[b] - torch.cumsum(pm, 0)
        E = E.gather(1, rho.clamp(0, Kb)[:, :, None].expand(-1, -1, T))
    for t, nw in (m.slva_log or []):
        live = pad & (t <= Kt)[:, None]; Et, Ep = E[:, t], E[:, t - 1]; off = 1.0 - nw.float()
        for k_, sel in (("chg", (Et != Ep) & live), ("copy", (Et == Ep) & (Et != VB.VOID) & live)):
            out[k_][0] += float(off[sel].sum()); out[k_][1] += int(sel.sum())
    m.slva_log = []
    return out
