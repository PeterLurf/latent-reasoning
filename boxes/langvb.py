"""§218 LANG-VB (registered f10546be; LANG_v1.md at 83f2dea4 §2.1): LANG-SLVA = D3-T's recipe (boxes/slva.SLVA) over a SLOT SUBSET of natural-language
text. New module; vbank / slva / deltamem are imported read-only (bank_kv, block_bank, _block, gpt_lower, SLVA.gather / gkey / nrm).
  Prelude: x = [EOT] + " " + text, blocks 0-5 over all T tokens -> H_6; V_0 = H_6 at the slot positions; the bank's version-0 entries are ALL T rows of
    H_6 (beta0 per head, no age term; token padding masked).
  Core pass t = 1..K (blocks 6-9): queries = the current version's |S| slot rows; keys / values = [version 0: all T rows] + [versions 1..t-1: slot rows]
    + [current slots, causal in position order (slots are sorted by position; D-4)], from each block's own ln_1 + c_attn (vbank.bank_kv); bias
    beta + gamma (t - 1 - s) on versions >= 1 (padded slots masked); calibrated per head at step 0 (99th-percentile bank mass <= 0.02, vbank's rule).
  Pass input: SLVA's per-slot version attention (slva.SLVA.gather, unchanged, on (B, S, D) slot tensors), c = c_anneal (trainer: vbank.c_at).
  Write: V_t = the core output at the slots, appended (never overwritten); no pass token; slot position ids fixed (implicit, as vbank).
  Coda after pass t (question-free): blocks 10-11 over [V_t slots | decoder tokens] with bank versions 1..t-1 (V_0 excluded); the decoder string is run
    through blocks 0-9 ALONE (no question keys, no bank) at positions T_i + 1 .. (assert T_i + L_dec <= 1,024); padded slots are removed keys for every row.
    The logits predicting decoder token k come from the row before it (the last real slot row for k = 1).
  Recode head R_P: LN + Linear(768 -> {OUT 0, IN 1}) per concept slot.
  Losses (READY-RECODE): L = R1 + rec_w R2. R1 = the gold decoder string's token CE over the readiness window t in [tau_i, K_i] (tau_i = D_i; self items:
    the stop s). R2 per (item, t), t in 1..K_i: 1/2 weighted-mean CE over changed slots (E_t != E_{t-1}) + 1/2 over unchanged real slots, IN weighted 2;
    HOLD after tau (E_t = E_tau); no changed slot -> the unchanged mean alone (vbank.vb_loss's rule). Self items (SOUND-ONLY R2): only marked slots carry
    an R2 term (IN from their first mark pass), OUT is never a target.
    The per-(item, t) combination is vbank.vb_loss's over the VALID slots (Audit G-L13 (e)): a pass whose valid slots are all changed (or all
    unchanged) takes that half's mean alone (not 1/2 of it); the item's R2 is the mean over its passes t <= K_i with >= 1 valid slot (self items:
    passes with >= 1 marked slot; plain items: every pass, all real slots valid). Note 218-2 (a0501784): intended - vb_loss generalised to
    sparse targets, equal to vb_loss on dense gold REACH.
  cand_ce(item, t) = -log softmax([sum log p("### o_R."), sum log p("### o_P.")])[o_R] from the coda after pass t (chance ln 2).
Items (boxes/prosqa_sym): ids ([EOT] + " " + text), slot_idx (sorted positions of the concepts' slot tokens) with the concept order `concepts`,
  dist_R {concept: d} (gold REACH; absent = unreachable), D, o_R / o_P (names).
"""
import math

import numpy as np
import tiktoken
import torch
import torch.nn.functional as F

import boxes.slva as SL
import boxes.vbank as VB

ENC = tiktoken.get_encoding("gpt2"); EOT = 50256
CORE, CODA, LAYERS = VB.CORE, VB.CODA, VB.LAYERS; NEG = VB.NEG; OUT, IN = 0, 1


def DM():
    import boxes.deltamem as DM_
    return DM_


def dec_ids(name):
    """the decoder string "### <name>." as GPT-2 ids (name in its leading-space form inside the string)."""
    return ENC.encode("### " + name.strip() + ".")


class LangSLVA(SL.SLVA):
    def __init__(self, g, table="train"):
        super().__init__(g, table)
        self.mode = "lslva"; D = self.D; dev = g.w("wte.weight").device
        self.rec = torch.nn.Linear(D, 2).to(dev)                            # R_P: {OUT, IN}
        self.v0_switch = None                                               # eval-only (CF): (t_from, V0full', V0slots') -> from pass t_from + 1 every version-0 read is x''s

    # ------------------------------------------------------------------ the rollout
    def prelude(self, x):
        g = self.g; return DM().gpt_lower(g, g.embed(x), CORE[0])          # (B, T, D) = H_6

    def dec_hidden(self, dec, lens):
        """decoder ids (b, L) -> the residual entering block 10: blocks 0-9 over the decoder alone at positions lens_i + k (k = 0..L-1)."""
        g = self.g; b, L = dec.shape; pos = lens[:, None] + torch.arange(L, device=dec.device)[None]
        assert int(pos.max()) < 1024, "T + L_dec > 1024"
        h = g.embed(dec) + g.w("wpe.weight")[pos]; hd = self.D // g.H
        for i in range(CODA[0]):
            h = DM()._block(g, i, h, hd)
        return h

    def coda_ls(self, t, Vt, kvs, spad, dec, lens, src=None, bp=None):
        """blocks 10-11 over [Vt (b, S, D) | decoder rows] with bank versions 1..t-1 (kvs {layer: [(k, v)] per version, entry 0 = V_0 unused}) -> ln_f
        hidden (b, S + L, D). Padded slots are removed keys for every row (block_bank kmask); decoder rows attend to every real slot."""
        src = src or self.g; D = self.D; S = Vt.shape[1]; Hd = self.dec_hidden(dec, lens); h = torch.cat([Vt, Hd.to(Vt.dtype)], 1); L = Hd.shape[1]
        km = torch.cat([~spad, torch.zeros(spad.shape[0], L, dtype=torch.bool, device=spad.device)], 1)
        bp = bp or (self.beta[4:], self.beta0[4:], self.gam[4:])
        for r, i in enumerate(CODA):
            if t <= 1:
                h = VB.block_bank(src, i, h, None, None, None, km)
            else:
                ents = self._dup(kvs[i], t)[1:t]; kp = self.ver_keep(t, t) if self.ver_keep is not None else None
                bk = torch.cat([k for k, _ in ents], 2); bv = torch.cat([v for _, v in ents], 2)
                bb = self.bias_ls(bp[0][r], bp[1][r], bp[2][r], t, None, spad, t - 1, v0=False, keep=kp).expand(-1, -1, S + L, -1)
                if self.cap is not None:
                    self.cap[("coda", i, t)] = (h.detach(), km)
                h = VB.block_bank(src, i, h, bk, bv, bb, km)
        return F.layer_norm(h, (D,), src.w("ln_f.weight"), src.w("ln_f.bias"))

    def bias_ls(self, b, b0, gm, t, tpad, spad, nslotver, v0=True, keep=None):
        """the bank bias row read at pass (or coda) index t: [version 0: beta0 on the T token rows (tpad), when v0] + [versions 1..nslotver: beta + gamma
        (t - 1 - s) on the S slot rows (spad)]; -inf where masked; bank_off -> all -inf; keep (eval-only ver_keep(t, t), bool over versions 0..t-1)
        masks the versions it drops. -> (B, H, 1, (T if v0) + nslotver * S)."""
        B, S = spad.shape; H = len(b); parts, valid = [], []
        if v0:
            T = tpad.shape[1]; parts.append(b0[:, None].expand(H, T)); valid.append(tpad if keep is None or bool(keep[0]) else torch.zeros_like(tpad))
        if nslotver:
            s = torch.arange(1, nslotver + 1, device=spad.device); age = (t - 1 - s).to(b.dtype)
            row = (b[:, None] + gm[:, None] * age[None, :])[:, :, None].expand(H, nslotver, S).reshape(H, nslotver * S); parts.append(row)
            vs = spad[:, None, :].expand(B, nslotver, S)
            if keep is not None:
                vs = vs & keep.to(spad.device)[1:nslotver + 1][None, :, None]
            valid.append(vs.reshape(B, nslotver * S))
        row = torch.cat(parts, 1); vl = torch.cat(valid, 1)
        if self.bank_off:
            vl = torch.zeros_like(vl)
        return torch.where(vl[:, None, None, :], row[None, :, None, :], torch.full((), NEG, dtype=row.dtype, device=row.device))

    def _dup(self, lst, t):
        """eval-only (vbank's 213-21 DUP): with dup_newest at a reader of pass / coda t >= 3, entries 1..t-2 read as the newest stored entry t-1."""
        if not self.dup_newest or t < 3:
            return lst
        return [lst[0]] + [lst[t - 1]] * (t - 2) + list(lst[t - 1:])

    def rollout_ls(self, x, pad, slot_idx, spad, K, coda_rows=None, dec=None, lens=None, edit=None, sg_input=False):
        """x (B, T) ids right-padded, pad (B, T) bool; slot_idx (B, S) long (padded rows point at 0), spad (B, S) bool; K passes.
        coda_rows {t: rows} with dec {t: (b, L) decoder ids for those rows} (or one (B, L) tensor for every t): the coda after pass t -> out["coda"][t] =
        (rows, h (b, S + L, D)). edit(t, V_t) -> the stored entry (None = V_t). -> {"V": V_0..V_K (slot rows), "S": stored, "coda": {...}, "kv": bank, "H6": H_6}."""
        assert self.slot_mask is None and self.dup_foreign is None and self.ver_sub_from is None and self.age_floor is None, \
            "rollout_ls does not implement slot_mask / dup_foreign / ver_sub_from / age_floor (Audit G-L13 (c))"
        g = self.g; B, T = x.shape; S = slot_idx.shape[1]; H6 = self.prelude(x)
        if self.v0_switch is not None:                                      # CF: version-0 reads after t_from come from x''s prelude
            t_from, H6sw, _ = self.v0_switch
        V0 = H6.gather(1, slot_idx[..., None].expand(-1, -1, self.D)); rms0 = V0.pow(2).mean(-1, keepdim=True).sqrt()
        kv = {i: [VB.bank_kv(g, i, H6)] for i in LAYERS}; gk = [self.gkey(V0)]; NV = [self.nrm(V0, rms0)]; Vc, St, out = [V0], [V0], {}
        kv_sw = gk_sw = NV_sw = None
        if self.v0_switch is not None:
            V0s = H6sw.gather(1, slot_idx[..., None].expand(-1, -1, self.D)); rms0_s = V0s.pow(2).mean(-1, keepdim=True).sqrt()
            kv_sw = {i: VB.bank_kv(g, i, H6sw) for i in LAYERS}; gk_sw = self.gkey(V0s); NV_sw = self.nrm(V0s, rms0_s)

        def coda_at(t, Vt):
            if coda_rows and t in coda_rows:
                r_ = coda_rows[t]; d_ = dec[t] if isinstance(dec, dict) else dec[r_]
                out[t] = (r_, self.coda_ls(t, Vt[r_], {i: [(k[r_], v[r_]) for k, v in kv[i]] for i in CODA}, spad[r_], d_, lens[r_]))
        coda_at(0, V0)
        for t in range(1, K + 1):
            sw = self.v0_switch is not None and t > t_from
            gk_t = ([gk_sw] + gk[1:]) if sw else gk; NV_t = ([NV_sw] + NV[1:]) if sw else NV
            gk_t, NV_t = self._dup(gk_t, t), self._dup(NV_t, t); kp = self.ver_keep(t, t) if self.ver_keep is not None else None
            h = self.gather(t, gk_t, NV_t, spad, keep=kp)
            h = h.detach() if sg_input else h
            if self.input_edit is not None:
                h = self.input_edit(t, h)
            for li, i in enumerate(CORE):
                kv0 = kv_sw[i] if sw else kv[i][0]; ents = self._dup(kv[i], t)
                bk = torch.cat([kv0[0]] + [k for k, _ in ents[1:t]], 2); bv = torch.cat([kv0[1]] + [v for _, v in ents[1:t]], 2)
                bb = self.bias_ls(self.beta[li], self.beta0[li], self.gam[li], t, pad, spad, t - 1, keep=kp).expand(-1, -1, S, -1)
                if self.cap is not None:
                    self.cap[("core", i, t)] = (h.detach(), None)
                h = VB.block_bank(g, i, h, bk, bv, bb)
            Vc.append(h); coda_at(t, h)
            st = edit(t, h) if edit is not None else None; st = h if st is None else st
            St.append(st)
            for i in LAYERS:
                kv[i].append(VB.bank_kv(g, i, st))
            gk.append(self.gkey(st)); NV.append(self.nrm(st, rms0_s if sw else rms0))
        return {"V": Vc, "S": St, "coda": out, "kv": kv, "H6": H6}

    # ------------------------------------------------------------------ heads and readings
    def rec_logits(self, V):
        lg = self.rec(self.rec_ln(V)); return lg if lg.dtype == torch.float64 else lg.float()

    def dec_logprobs(self, h, spad, dec):
        """coda hidden (b, S + L, D), spad (b, S), decoder ids (b, L) -> per-token log p of the decoder string (b, L): token k predicted from the row before it
        (the last real slot row for k = 0)."""
        b, SL_, _ = h.shape; L = dec.shape[1]; S = SL_ - L; last = spad.sum(1) - 1
        rows = torch.cat([h[torch.arange(b, device=h.device), last][:, None], h[:, S:S + L - 1]], 1)        # (b, L, D)
        lg = self.g.logits(rows); lp = torch.log_softmax(lg if lg.dtype == torch.float64 else lg.float(), -1)   # fp64 stays fp64 (exactness gates)
        return lp.gather(-1, dec[..., None]).squeeze(-1)

    @torch.no_grad()
    def calibrate(self, x, pad, slot_idx, spad, decR, lens, K=VB.KMAX, mu=0.02, q=0.99):
        """vbank.VBankN.calibrate's rule on LANG-SLVA's shapes: a bank-masked rollout to V_0..V_K with every layer input captured; per layer and head
        beta0 (core layers: version 0 = the T token rows) is set first, then beta (versions 1..t-1, slot rows), each to logit(mu) - 0.01 - the q-quantile
        of log A - log C over (items, real query rows, t). The coda has no version-0 entry. -> report."""
        g = self.g; D, H, hd = self.D, self.H, self.hd; off0 = self.bank_off; self.bank_off = True; self.cap = {}
        try:
            r = self.rollout_ls(x, pad, slot_idx, spad, K, coda_rows={t: torch.arange(x.shape[0], device=x.device) for t in range(1, K + 1)}, dec=decR, lens=lens)
            cap = self.cap
        finally:
            self.cap = None; self.bank_off = off0
        rep = {}; lmu = math.log(mu / (1 - mu)) - 0.01; S = slot_idx.shape[1]; T = x.shape[1]
        for li, i in enumerate(LAYERS):
            kind = "core" if i in CORE else "coda"; A0, R1_, C = [], [], []; selq = []
            for t in range(1 if kind == "core" else 2, K + 1):
                h, km = cap[(kind, i, t)]; B, Q, _ = h.shape; pre = f"h.{i}."
                a = F.layer_norm(h, (D,), g.w(pre + "ln_1.weight"), g.w(pre + "ln_1.bias"))
                qh, kh, _ = (a @ g.w(pre + "attn.c_attn.weight") + g.w(pre + "attn.c_attn.bias")).split(D, -1)
                qh, kh = (u.view(B, Q, H, hd).transpose(1, 2).float() for u in (qh, kh))
                cm = torch.ones(Q, Q, dtype=torch.bool, device=x.device).triu(1)[None, None]
                if km is not None:
                    cm = cm | km[:, None, None, :]
                sc_cur = (qh @ kh.transpose(-1, -2) / math.sqrt(hd)).masked_fill(cm, NEG); C.append(torch.logsumexp(sc_cur, -1))
                qsel = spad if kind == "core" else torch.cat([spad, torch.ones(B, Q - S, dtype=torch.bool, device=x.device)], 1); selq.append(qsel)
                if kind == "core":
                    k0 = r["kv"][i][0][0].float(); s0 = (qh @ k0.transpose(-1, -2) / math.sqrt(hd)).masked_fill(~pad[:, None, None, :], NEG); A0.append(torch.logsumexp(s0, -1))
                if t >= 2:
                    kb = torch.cat([k for k, _ in r["kv"][i][1:t]], 2).float(); sb = qh @ kb.transpose(-1, -2) / math.sqrt(hd)
                    age = (t - 1 - torch.arange(1, t, device=x.device).repeat_interleave(S)).float()
                    sb = (sb + self.gam[li].float()[None, :, None, None] * age[None, None, None, :]).masked_fill(~spad.repeat(1, t - 1)[:, None, None, :], NEG)
                    R1_.append(torch.logsumexp(sb, -1))
                else:
                    R1_.append(None)

            def qq(xs, sel):
                v = torch.cat([z.permute(1, 0, 2)[:, sl_] for z, sl_ in zip(xs, sel)], 1)   # (H, n)
                return v
            if kind == "core":
                x0 = qq([a0 - c0 for a0, c0 in zip(A0, C)], selq); b0 = torch.stack([lmu - torch.quantile(x0[h_], q) for h_ in range(H)])
                self.beta0.data[li] = b0.to(self.beta0.dtype)
                x1 = qq([ar - torch.logaddexp(c0, a0 + b0[None, :, None]) for ar, a0, c0 in zip(R1_, A0, C) if ar is not None], [s_ for s_, ar in zip(selq, R1_) if ar is not None])
            else:
                b0 = None; x1 = qq([ar - c0 for ar, c0 in zip(R1_, C) if ar is not None], [s_ for s_, ar in zip(selq, R1_) if ar is not None])
            b1 = torch.stack([lmu - torch.quantile(x1[h_], q) for h_ in range(H)]); self.beta.data[li] = b1.to(self.beta.dtype)
            m1 = torch.sigmoid(x1 + b1[:, None]); rep[str(i)] = {"beta0": None if b0 is None else [round(float(v), 4) for v in b0], "beta": [round(float(v), 4) for v in b1],
                                                                 "m99_rest": [float(torch.quantile(m1[h_], q)) for h_ in range(H)]}
            if b0 is not None:
                m0 = torch.sigmoid(x0 + b0[:, None]); rep[str(i)]["m99_v0"] = [float(torch.quantile(m0[h_], q)) for h_ in range(H)]
        rep["max_m99"] = max(max(v.get("m99_v0", []) + v["m99_rest"]) for k, v in rep.items() if k != "max_m99"); rep["mu"] = mu; rep["q"] = q
        return rep


# ---------------------------------------------------------------------- batches, targets, loss
def lvb_batch(items, dev):
    """items -> x (B, T), pad, lens (B,) = T_i, slot_idx (B, S), spad, decR / decP (B, L) (the gold and the other option's decoder strings)."""
    T = max(len(it["ids"]) for it in items); S = max(len(it["slot_idx"]) for it in items); B = len(items)
    x = torch.full((B, T), EOT, dtype=torch.long); pad = torch.zeros(B, T, dtype=torch.bool); si = torch.zeros(B, S, dtype=torch.long); sp = torch.zeros(B, S, dtype=torch.bool)
    dR = [dec_ids(it["o_R"]) for it in items]; dP = [dec_ids(it["o_P"]) for it in items]; L = len(dR[0])
    assert all(len(d) == L for d in dR + dP), "decoder strings of unequal length (NONCE names are 2 tokens)"
    for b, it in enumerate(items):
        n = len(it["ids"]); x[b, :n] = torch.tensor(it["ids"]); pad[b, :n] = True; k = len(it["slot_idx"])
        si[b, :k] = torch.tensor(it["slot_idx"]); sp[b, :k] = True
        assert list(it["slot_idx"]) == sorted(it["slot_idx"]), "slots must be sorted by position (causal order)"
    lens = pad.sum(1)
    return x.to(dev), pad.to(dev), lens.to(dev), si.to(dev), sp.to(dev), torch.tensor(dR, device=dev), torch.tensor(dP, device=dev)


def reach_targets(items, Kb, S, dev):
    """(B, Kb + 1, S) class ids E_t(c) = IN iff dist(R, c) <= t (HOLD after D by REACH's saturation), -100 on padded slots; self items ("marks": per slot the
    first pass it is soundly marked, or None) -> IN from that pass, -100 elsewhere (SOUND-ONLY: OUT is never a target)."""
    E = np.full((len(items), Kb + 1, S), -100, dtype=np.int64)
    for b, it in enumerate(items):
        for j, c in enumerate(it["concepts"]):
            if "marks" in it:
                f = it["marks"][j]
                if f is not None:
                    E[b, f:, j] = IN
            else:
                d = it["dist_R"].get(c); E[b, :, j] = OUT
                if d is not None and d <= Kb:
                    E[b, d:, j] = IN
    return torch.tensor(E, device=dev)


def lvb_loss(m, items, Ks, dev, ac, rec_w=1.0, sg_input=False, keep=False):
    """L = R1 + rec_w R2 (module docstring) for items with pass budgets Ks; tau_i = D_i (self items: their stop "stop_s"). -> (L, stats incl. plain_cand_ce
    (the mean over plain items of the per-item mean cand_ce over the readiness window), the rollout if keep)."""
    x, pad, lens, si, sp, dR, dP = lvb_batch(items, dev); B = x.shape[0]; S = si.shape[1]; Kb = int(max(Ks)); Kt = torch.tensor(Ks, device=dev)
    tau = torch.tensor([it.get("stop_s", it["D"]) for it in items], device=dev); E = reach_targets(items, Kb, S, dev)
    win = {t: torch.nonzero((tau <= t) & (t <= Kt)).flatten() for t in range(0, Kb + 1)}; win = {t: r_ for t, r_ in win.items() if len(r_)}
    with ac():
        r = m.rollout_ls(x, pad, si, sp, Kb, coda_rows=win, dec=dR, lens=lens, sg_input=sg_input)
    dt = torch.float64 if m.g.w("wte.weight").dtype == torch.float64 else torch.float32     # fp64 models (exactness gates) accumulate in fp64
    ans_sum = torch.zeros(B, device=dev, dtype=dt); ans_cnt = torch.zeros(B, device=dev, dtype=dt); cc_sum = torch.zeros(B, device=dev, dtype=dt)
    for t, (rows, h) in r["coda"].items():
        lpR = m.dec_logprobs(h, sp[rows], dR[rows]); ce = -lpR.mean(1)
        ans_sum = ans_sum.index_add(0, rows, ce); ans_cnt = ans_cnt.index_add(0, rows, torch.ones_like(ce))
        with torch.no_grad():                                               # cand_ce from the same coda: the other option's string
            with ac():
                hP = m.coda_ls(t, r["V"][t][rows], {i: [(k[rows], v[rows]) for k, v in r["kv"][i]] for i in CODA}, sp[rows], dP[rows], lens[rows])
            lpP = m.dec_logprobs(hP, sp[rows], dP[rows]); sR, sP = lpR.detach().sum(1), lpP.sum(1)
            cc_sum = cc_sum.index_add(0, rows, -torch.log_softmax(torch.stack([sR, sP], 1), 1)[:, 0])
    rec_sum = torch.zeros(B, device=dev); rec_cnt = torch.zeros(B, device=dev)
    for t in range(1, Kb + 1):
        Et, Ep = E[:, t], E[:, t - 1]; valid = sp & (t <= Kt)[:, None] & (Et >= 0)
        if not bool(valid.any()):
            continue
        ce = F.cross_entropy(m.rec_logits(r["V"][t]).transpose(1, 2), Et.clamp_min(0), reduction="none").float()
        w = 1.0 + (Et == IN).float(); ch = ((Et != Ep) & valid).float(); un = ((Et == Ep) & valid).float()
        tc = (w * ce * ch).sum(1) / (w * ch).sum(1).clamp_min(1e-9); tu = (w * ce * un).sum(1) / (w * un).sum(1).clamp_min(1e-9)
        live = ((t <= Kt) & (valid.sum(1) > 0)).float(); term = torch.where(ch.sum(1) > 0, torch.where(un.sum(1) > 0, 0.5 * tc + 0.5 * tu, tc), tu)
        rec_sum = rec_sum + term * live; rec_cnt = rec_cnt + live
    has_a = ans_cnt > 0; L_ans = (ans_sum[has_a] / ans_cnt[has_a]).mean(); L_rec = (rec_sum / rec_cnt.clamp_min(1)).mean()
    plain = torch.tensor([("marks" not in it) for it in items], device=dev) & has_a
    pcc = float((cc_sum[plain] / ans_cnt[plain]).mean()) if bool(plain.any()) else None
    st = {"answer_ce": float(L_ans.detach()), "rec_ce": float(L_rec.detach()), "plain_cand_ce": pcc, "n_plain": int(plain.sum())}
    if keep:                                                                # G-L7(b'): per-item window-mean cand_ce (None: no window row)
        st["cand_ce_items"] = [float(c_ / n_) if n_ > 0 else None for c_, n_ in zip(cc_sum.tolist(), ans_cnt.tolist())]
    return L_ans + rec_w * L_rec, st, (r if keep else None)


def make_model(g, table="train"):
    return LangSLVA(g, table)


# ---------------------------------------------------------------------- DIRECT-SYM (§2.6): plain GPT-2 on [ids | decoder string]
def direct_batch(items, key, dev):
    """x (B, T_max + L) = [ids | "### <it[key]>."] right-padded (causal, so the padding is never read), rows (B, L) = the position predicting each
    decoder token, d (B, L) the decoder ids."""
    L = len(dec_ids(items[0][key])); W = max(len(it["ids"]) for it in items) + L; B = len(items)
    x = torch.full((B, W), EOT, dtype=torch.long); rows = torch.zeros(B, L, dtype=torch.long); d = torch.zeros(B, L, dtype=torch.long)
    for b, it in enumerate(items):
        n = len(it["ids"]); dd = dec_ids(it[key]); assert len(dd) == L and n + L <= 1024, (len(dd), L, n)
        x[b, :n] = torch.tensor(it["ids"]); x[b, n:n + L] = torch.tensor(dd); rows[b] = torch.arange(n - 1, n - 1 + L); d[b] = torch.tensor(dd)
    return x.to(dev), rows.to(dev), d.to(dev)


def direct_logprobs(g, items, key, dev, ac):
    x, rows, d = direct_batch(items, key, dev)
    with ac():
        h, _ = g(g.embed(x))
    hr = h.gather(1, rows[..., None].expand(-1, -1, h.shape[-1])); lp = torch.log_softmax(g.logits(hr).float(), -1)
    return lp.gather(-1, d[..., None]).squeeze(-1)


def direct_loss(g, items, dev, ac):
    """DIRECT-SYM: the gold decoder string's token CE; cand_ce from a no-grad forward on the other option's string (chance ln 2)."""
    lpR = direct_logprobs(g, items, "o_R", dev, ac); L = -lpR.mean(1).mean()
    with torch.no_grad():
        lpP = direct_logprobs(g, items, "o_P", dev, ac); cc = -torch.log_softmax(torch.stack([lpR.detach().sum(1), lpP.sum(1)], 1), 1)[:, 0]
    return L, {"answer_ce": float(L.detach()), "rec_ce": 0.0, "plain_cand_ce": float(cc.mean()), "n_plain": len(items)}


@torch.no_grad()
def cand_scores(m, items, K, dev, ac):
    """(sum log p(gold string), sum log p(other string)) per item: LangSLVA's coda after pass K, or DIRECT's plain forward (K unused)."""
    if not isinstance(m, LangSLVA):
        return direct_logprobs(m, items, "o_R", dev, ac).sum(1), direct_logprobs(m, items, "o_P", dev, ac).sum(1)
    x, pad, lens, si, sp, dR, dP = lvb_batch(items, dev); rows = torch.arange(len(items), device=dev)
    with ac():
        r = m.rollout_ls(x, pad, si, sp, K, coda_rows={K: rows}, dec=dR, lens=lens)
        hP = m.coda_ls(K, r["V"][K], {i: r["kv"][i] for i in CODA}, sp, dP, lens)
    return m.dec_logprobs(r["coda"][K][1], sp, dR).sum(1), m.dec_logprobs(hP, sp, dP).sum(1)


# ---------------------------------------------------------------------- the trainer (§2.3 A1, §2.4 A2 / A2-CTRL, §2.6 DIRECT-SYM)
D_MIN = 3


def PS():
    import boxes.prosqa_sym as PS_
    return PS_


def lvb_item(it):
    """a boxes.prosqa_sym item -> lvb_loss's item: concepts in slot-position order (slot_idx sorted, D-4), o_R / o_P as names."""
    cs = sorted(it["concepts"], key=lambda c: it["slot_idx"][c])
    return {"ids": it["ids"], "slot_idx": [it["slot_idx"][c] for c in cs], "concepts": cs, "dist_R": it["dist_R"], "D": it["D"],
            "o_R": it["names"][it["o_R"]], "o_P": it["names"][it["o_P"]]}


def stream_state(s):
    return {"rngs": {D: r.getstate() for D, r in s.rngs.items()}, "n": dict(s.n), "n_collisions": len(s.collisions)}


def stream_load(s, st):
    import random
    for D, g_ in st["rngs"].items():
        s.rngs[D] = random.Random(); s.rngs[D].setstate(g_)
    s.n.update(st["n"])


def main():
    """vbank.main's loop on SYM v1 (§2.3 / §2.4 / §2.6). Draw law per plain step: D_i ~ U{3..D_max} (rng = seed; one integers(size=bs) call), the item = the
    next accepted draw of band D_i of prosqa_sym.Stream(--stream_seed) with the test / GATE signatures excluded, K_i = min(D_i + U{0..kextra}, kmax)
    (rng_k = seed + 21100). CurGate D_max 4 -> 6 -> 8 reads plain_cand_ce only (micro-chunks weighted by n_plain); self steps never call it (G-L7).
    §215's self order kept exactly: the plain batch is drawn every step, then rng_s = seed + 21800 draws one coin (p_self) and, on a self step, one U per
    record (K = min(stop_s + U, self_kmax)). Calibration (lslva, not init_from): 64 items, D ~ U{3..8} from rng 21110, Stream(21110) (unregistered
    seeds, no reading uses them). --time_steps is LAW-MATCHED (G-L12): the final stage, the registered draw / coin law, --time_warmup untimed steps first."""
    import argparse
    import hashlib
    import json
    import os
    import time
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="lslva", choices=["lslva", "direct"], help="lslva = A1 / A2 / A2-CTRL; direct = DIRECT-SYM (plain GPT-2 small)")
    ap.add_argument("--table", default="train", choices=["train", "frozen"]); ap.add_argument("--args_ref", default="")
    ap.add_argument("--steps", type=int, default=24000); ap.add_argument("--bs", type=int, default=64); ap.add_argument("--micro", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--seed", type=int, default=1); ap.add_argument("--stream_seed", type=int, default=22100)
    ap.add_argument("--cur", default="4,6,8"); ap.add_argument("--cur_gate", default="0.35:2:6000", help="thr:need:cap on plain_cand_ce")
    ap.add_argument("--c_sched", default="1000,8000", help="vbank.c_at hold,end (A2: 0,1 = c 0 from step 1)"); ap.add_argument("--rec_w", type=float, default=1.0)
    ap.add_argument("--kmax", type=int, default=12); ap.add_argument("--kextra", type=int, default=4)
    ap.add_argument("--self_pool", default="", help="§2.4: the pool jsonl (pool_meta.json beside it); empty = no self steps (A1 / A2-CTRL)")
    ap.add_argument("--p_self", type=float, default=0.0); ap.add_argument("--self_kmax", type=int, default=16); ap.add_argument("--self_kextra", type=int, default=4)
    ap.add_argument("--self_micro", type=int, default=1); ap.add_argument("--mem_check_self", action="store_true")
    ap.add_argument("--draws_log_steps", type=int, default=0, help="G-L19 / G-L21: steps 1..N -> draws_log.json")
    ap.add_argument("--init_from", default="", help="A2 / A2-CTRL: A1's model.pt (strict); calibrate() not called, its calibration.json copied; fresh optimizer")
    ap.add_argument("--ckpt_every", type=int, default=2000); ap.add_argument("--resume", default=None); ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--time_steps", type=int, default=0, help="G-L12: timed steps (law-matched, final stage) -> timing.json + eval.json, exit")
    ap.add_argument("--time_warmup", type=int, default=20)
    ap.add_argument("--mem_check", action="store_true", help="G-L8: the forced worst shape (64 longest final-stage items, K kmax) -> mem_check.json + eval.json")
    ap.add_argument("--mem_check_steps", type=int, default=1); ap.add_argument("--auto_micro", action="store_true"); ap.add_argument("--mem_budget", type=float, default=21.0)
    ap.add_argument("--stop_at", type=int, default=0); ap.add_argument("--keep_ckpt_steps", default=""); ap.add_argument("--cal_n", type=int, default=64)
    ap.add_argument("--eval_bs", type=int, default=32); ap.add_argument("--per_n", type=int, default=200, help="host eval: the first per_n test items per D 3..12")
    ap.add_argument("--fp32", action="store_true"); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--out", required=True)
    ap.add_argument("--args_only", action="store_true", help="write the args.json this command would write (args_only recorded False) and exit (Theory freezes it)")
    ap.add_argument("--host_eval_at", default="8000,16000", help="Note 218-8 (descriptive): candidate accuracy at K' = D + 2 on a fixed non-test set "
                    "(Stream 21170 / rng 21171, 100 per D 3..8) logged at these steps")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); dev = a.device; direct = a.arch == "direct"
    from boxes import s5
    from boxes.fsq120 import lr_at, driver_version, env_info
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); rng_k = np.random.default_rng(a.seed + 21100)
    amp = None if (a.fp32 or not dev.startswith("cuda") or torch.cuda.get_device_capability(dev)[0] < 8) else torch.bfloat16
    ac = lambda: torch.autocast("cuda", dtype=amp, enabled=amp is not None)  # noqa: E731
    t_ex = time.time(); EXCL = PS().exclusion_sigs(); stream = PS().Stream(a.stream_seed, EXCL)
    print("EXCLUSION %d signatures (%.0f s)" % (len(EXCL), time.time() - t_ex), flush=True)
    ns = [int(v) for v in a.cur.split(",")]; thr, need, cap = a.cur_gate.split(":"); GATE = VB.CurGate(ns, float(thr), int(need), int(cap))
    ch0, ch1 = (int(v) for v in a.c_sched.split(","))
    assert DM().POS_OFFSET == 0 and not DM().POS_KEEP0, "training needs the plain position table"
    if a.args_only:                                                         # before any model / GPU allocation: the args.json the launch would write
        a.arch_class = "GPT2KV" if direct else "LangSLVA"; a.args_only = False
        info = {"train_gpu": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu", "train_driver": driver_version(), **env_info()}
        json.dump({**vars(a), **info}, open(os.path.join(a.out, "args.json"), "w", newline="\n"), indent=1); print("ARGS_ONLY", a.out, flush=True); return
    m = s5.load_gpt2(dev) if direct else make_model(s5.load_gpt2(dev), a.table).to(dev); a.arch_class = type(m).__name__
    pool, rng_s, selfst = None, None, {"cursor": 0, "n_self_steps": 0, "self_reuse": 0}
    if a.self_pool:                                                         # §2.4: the pool (key set + sha checked by the loader); rng_s = seed + 21800
        assert not direct
        import boxes.langvb_label as LL
        pool, pool_meta = LL.load_pool(a.self_pool, os.path.join(os.path.dirname(a.self_pool), "pool_meta.json")); rng_s = np.random.default_rng(a.seed + 21800)
        print("SELF_POOL", a.self_pool, len(pool), "records, sha", pool_meta["sha256"][:16], flush=True)
    if a.table == "frozen":
        assert not direct; m.g.p["wpe_weight"].requires_grad_(False)
    opt = torch.optim.AdamW([p_ for p_ in m.parameters() if p_.requires_grad], lr=a.lr, weight_decay=0.01)
    info = {"train_gpu": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu", "train_driver": driver_version(), **env_info()}
    step0, t_prev, n_nonfinite, cal, gate_init = 1, 0.0, 0, None, None
    if a.resume and os.path.exists(a.resume):
        ck = torch.load(a.resume, map_location=dev, weights_only=False); m.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        rng.bit_generator.state = ck["np_rng"]; rng_k.bit_generator.state = ck["rng_k"]; torch.set_rng_state(ck["torch_rng"].cpu()); stream_load(stream, ck["stream"])
        if rng_s is not None:
            rng_s.bit_generator.state = ck["rng_s"]; selfst.update(ck["self_state"])
        GATE.load(ck["gate"]); step0 = ck["step"] + 1; t_prev = ck["elapsed"]; n_nonfinite = ck["n_nonfinite"]; cal = ck["calibration"]
        print("RESUME", a.resume, "step", ck["step"], flush=True); del ck
    else:
        if a.init_from:
            sd = torch.load(a.init_from, map_location=dev, weights_only=True); m.load_state_dict(sd, strict=True); del sd
            if not direct:
                cal = json.load(open(os.path.join(os.path.dirname(a.init_from), "calibration.json")))
            print("INIT_FROM %s (strict); calibrate() NOT called" % a.init_from, flush=True)
            GATE.stage = len(ns) - 1; gate_init = "final"                   # §2.4: A2 / A2-CTRL plain batches from A1's FINAL-stage law (Audit G-L13 D1);
            print("GATE_INIT final: D_max %d from step 1 (CurGate never advances)" % GATE.n_max(), flush=True)   # resume restores it from the ckpt
        elif not direct:
            r_c = np.random.default_rng(21110); s_c = PS().Stream(21110, EXCL)
            cal_its = [lvb_item(s_c.draw(int(r_c.integers(D_MIN, 9)))) for _ in range(a.cal_n)]
            x_, p_, l_, si_, sp_, dR_, _ = lvb_batch(cal_its, dev); m.c_anneal.fill_(1e4)
            with torch.no_grad():
                cal = m.calibrate(x_, p_, si_, sp_, dR_, l_)
            del x_, p_, l_, si_, sp_, dR_
        if cal is not None:
            json.dump(cal, open(os.path.join(a.out, "calibration.json"), "w", newline="\n"), indent=1); print("CALIBRATION max_m99 %.4f" % cal["max_m99"], flush=True)
        json.dump({**vars(a), **info}, open(os.path.join(a.out, "args.json"), "w", newline="\n"), indent=1)
        if a.args_ref:                                                      # G-L11 ARGS-DIFF (launch log)
            ref = json.load(open(a.args_ref)); cur = {**vars(a), **info}
            dif = {k: [ref.get(k), cur.get(k)] for k in sorted(set(ref) | set(cur)) if k != "args_ref" and ref.get(k) != cur.get(k)}
            json.dump({"ref": a.args_ref, "diff": dif}, open(os.path.join(a.out, "args_diff.json"), "w", newline="\n"), indent=1); print("ARGS-DIFF", json.dumps(dif), flush=True)

    def loss_of(its, Ks):
        if direct:
            L_, st_ = direct_loss(m, its, dev, ac); return L_, st_
        L_, st_, _ = lvb_loss(m, its, Ks, dev, ac, a.rec_w); return L_, st_

    def pick_micro(its, K, tag):
        """G-L8: the smallest micro in {1, 2, 4, 8} whose forward + backward on `its` at K peaks <= mem_budget (no optimizer step; no rng consumed)."""
        pk_ = {}
        for mic in (1, 2, 4, 8):
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(dev); ok = True; ch_ = len(its) // mic
            try:
                for k_ in range(mic):
                    L_, _ = loss_of(its[k_ * ch_:(k_ + 1) * ch_], [K] * ch_); (L_ / mic).backward(); del L_
            except torch.OutOfMemoryError:
                ok = False
            m.zero_grad(set_to_none=True); pk = torch.cuda.max_memory_allocated(dev) / 2 ** 30; pk_[str(mic)] = round(pk, 2) if ok else "OOM"
            if ok and pk <= a.mem_budget:
                torch.cuda.empty_cache(); lg_ = {tag: mic, "peak_gib_by_micro": pk_, "mem_budget": a.mem_budget, "K": K}; print("AUTO_MICRO", lg_, flush=True)
                json.dump(lg_, open(os.path.join(a.out, "auto_%s.json" % tag), "w", newline="\n"), indent=1); return mic
        raise RuntimeError("G-L8: no %s in {1, 2, 4, 8} fits within %.1f GiB: %s" % (tag, a.mem_budget, pk_))
    s_big = PS().Stream(21120, EXCL)                                        # the forced worst plain shape: the 64 longest of 8 bs final-stage draws
    big = sorted([lvb_item(s_big.draw(ns[-1])) for _ in range(8 * a.bs)], key=lambda it: (-len(it["ids"]), -len(it["slot_idx"])))[:a.bs] if (a.mem_check or a.auto_micro) else None
    if a.auto_micro and dev.startswith("cuda"):
        a.micro = pick_micro(big, a.kmax, "micro")
    deep_its = None
    if pool is not None:                                                    # the deepest self shape: the 64 longest records (D is not stored), K self_kmax
        deep = sorted(range(len(pool)), key=lambda j: -len(pool[j]["text_ids"]))[:a.bs]; deep_its = [LL.self_item(pool[j], a.self_kmax) for j in deep]
        if a.auto_micro and dev.startswith("cuda"):
            a.self_micro = pick_micro(deep_its, a.self_kmax, "self_micro")
    KEEP = {int(v) for v in a.keep_ckpt_steps.split(",")} if a.keep_ckpt_steps else set()
    HEV = {int(v) for v in a.host_eval_at.split(",")} if a.host_eval_at else set(); HSET = {}
    if 0 in KEEP and step0 == 1:
        torch.save({"model": m.state_dict(), "step": 0}, os.path.join(a.out, "ckpt_model_0.pt"))
    if a.mem_check or a.time_steps:
        GATE.stage = len(ns) - 1
    log = open(os.path.join(a.out, "log.jsonl"), "a", newline="\n"); t0 = time.time() - t_prev; acc = []; gn_log = []; tmax = [0]; hits = [0]   # tmax: the max plain T drawn (T_band); hits: test / GATE sigs drawn
    if gate_init:
        log.write(json.dumps({"step": 0, "gate_init": gate_init, "cur_stage": GATE.stage, "D_max_train": GATE.n_max()}) + "\n"); log.flush()

    def draw(Dmax):
        if a.mem_check:
            return big, [a.kmax] * len(big)
        raws = [stream.draw(int(D)) for D in rng.integers(D_MIN, Dmax + 1, size=a.bs)]; hits[0] += sum(r_["sig"] in EXCL for r_ in raws)   # CONTAMINATED check
        its = [lvb_item(r_) for r_ in raws]; tmax[0] = max(tmax[0], max(len(it["ids"]) for it in its))
        return its, [min(it["D"] + int(rng_k.integers(0, a.kextra + 1)), a.kmax) for it in its]

    def draw_step(Dmax, step):
        """vbank.draw_step's order: the plain batch every step (rng, rng_k), then rng_s's coin; a self step takes the next bs pool records (one U each)."""
        its, Ks = draw(Dmax)
        if pool is not None and a.mem_check and a.mem_check_self:
            return deep_its, [a.self_kmax] * len(deep_its), True, None
        lg = (not a.mem_check) and step <= a.draws_log_steps
        if pool is None and not lg:
            return its, Ks, False, None
        dl = {"step": step, "plain_hash": hashlib.sha1(json.dumps([[it["ids"], k_] for it, k_ in zip(its, Ks)]).encode()).hexdigest()} if lg else None
        coin = False if pool is None else bool(rng_s.random() < a.p_self)
        if not coin:
            if lg:
                dl.update({"coin": False, "D": [it["D"] for it in its], "U": [k_ - it["D"] for it, k_ in zip(its, Ks)]})
            return its, Ks, False, dl
        c0 = selfst["cursor"]
        if c0 + a.bs > len(pool):
            raise RuntimeError("§2.4: the self pool is exhausted (no reuse allowed)")
        recs = pool[c0:c0 + a.bs]; selfst["cursor"] = c0 + a.bs; selfst["n_self_steps"] += 1
        U = [int(rng_s.integers(0, a.self_kextra + 1)) for _ in recs]
        its = [LL.self_item(r_, a.self_kmax) for r_ in recs]; Ks = [min(r_["stop_s"] + u, a.self_kmax) for r_, u in zip(recs, U)]
        if lg:
            dl.update({"coin": True, "stop_s": [r_["stop_s"] for r_ in recs], "U": U})
        return its, Ks, True, dl

    def gate_update(step, v, is_self):
        assert not is_self, "G-L7: a self batch reached GATE.update"
        return GATE.update(step, v)
    n_time = a.time_warmup + a.time_steps if a.time_steps else 0
    last = step0 + n_time - 1 if a.time_steps else (a.mem_check_steps if a.mem_check else (min(a.steps, a.stop_at) if a.stop_at else a.steps))
    draws_log = []; tself = [0.0, 0, 0.0, 0]; tt0 = None
    if a.resume and os.path.exists(os.path.join(a.out, "draws_log.json")):
        draws_log = [d for d in json.load(open(os.path.join(a.out, "draws_log.json"))) if d["step"] < step0]
    if dev.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(dev)
    for step in range(step0, last + 1):
        if a.time_steps and step == step0 + a.time_warmup:
            if dev.startswith("cuda"):
                torch.cuda.synchronize(dev)
            tt0 = time.time(); tself = [0.0, 0, 0.0, 0]
        lr = lr_at(a.lr, {}, step, a.steps)
        for gr in opt.param_groups:
            gr["lr"] = lr
        if not direct:
            m.c_anneal.fill_(VB.c_at(step, ch0, ch1) if m.anneals else 1e4)
        ts0 = time.time(); m.train(); its, Ks, is_self, dl = draw_step(GATE.n_max(), step); opt.zero_grad(set_to_none=True); tot = 0.0
        mic = a.self_micro if is_self else a.micro; ch = len(its) // mic; sts = []
        if dl is not None:
            draws_log.append(dl)
        for k_ in range(mic):
            L, st_ = loss_of(its[k_ * ch:(k_ + 1) * ch], Ks[k_ * ch:(k_ + 1) * ch]); (L / mic).backward(); tot += float(L.item()) / mic; st_["is_self"] = is_self; sts.append(st_)
        slva_gn = None
        if not direct and step % a.log_every == 0:
            sg_ = [p_.grad for p_ in m.slva_params() if p_.grad is not None]; slva_gn = float(torch.stack([g_.float().pow(2).sum() for g_ in sg_]).sum().sqrt()) if sg_ else 0.0
        gn = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); gn_log.append(float(gn))
        ans = float(np.mean([s_["answer_ce"] for s_ in sts])); rec = float(np.mean([s_["rec_ce"] for s_ in sts]))
        pc_ = [(s_["plain_cand_ce"], s_["n_plain"]) for s_ in sts if s_["plain_cand_ce"] is not None]
        pcc = sum(v_ * n_ for v_, n_ in pc_) / sum(n_ for _, n_ in pc_) if pc_ else None
        if not (math.isfinite(tot) and bool(torch.isfinite(gn))):
            n_nonfinite += 1; opt.zero_grad(set_to_none=True); log.write(json.dumps({"step": step, "nonfinite": 1}) + "\n"); log.flush(); pcc_g = None
        else:
            opt.step(); acc.append((tot, ans, rec, sts, pcc)); pcc_g = pcc
        if dev.startswith("cuda"):
            torch.cuda.synchronize(dev)
        tself[0 if is_self else 2] += time.time() - ts0; tself[1 if is_self else 3] += 1
        adv = None if (a.mem_check or a.time_steps or is_self) else gate_update(step, pcc_g, is_self)
        if adv is not None:
            log.write(json.dumps({"step": step, "cur_stage": adv[0], "D_max_train": GATE.n_max(), "reason": adv[1], "bin_mean": adv[2]}) + "\n"); log.flush()
        if step % a.log_every == 0 or step == last:
            pv = [v for v in acc if not v[3][0]["is_self"]]; sv = [v for v in acc if v[3][0]["is_self"]]
            rec_ = {"step": step, "loss": float(np.mean([v[0] for v in acc])) if acc else None, "sec": round(time.time() - t0),
                    "answer_ce": float(np.mean([v[1] for v in acc])) if acc else None, "rec_ce": float(np.mean([v[2] for v in acc])) if acc else None,
                    "plain_cand_ce": float(np.mean([v[4] for v in pv if v[4] is not None])) if any(v[4] is not None for v in pv) else None,
                    "D_max_train": GATE.n_max(), "cur_stage": GATE.stage, "lr": lr, "c": None if direct else float(m.c_anneal), "gn_preclip": gn_log[-1],
                    "clip_frac_100": float(np.mean([v > 1.0 for v in gn_log[-100:]])), "slva_gn": slva_gn, "stream_collisions": len(stream.collisions), "t_train_max": tmax[0], "test_gate_sig_hits": hits[0]}
            if pool is not None:
                rec_.update({"self_frac": len(sv) / len(acc) if acc else None, "self_answer_ce": float(np.mean([v[1] for v in sv])) if sv else None,
                             "self_rec_ce": float(np.mean([v[2] for v in sv])) if sv else None, "plain_answer_ce": float(np.mean([v[1] for v in pv])) if pv else None,
                             "pool_cursor": selfst["cursor"], "n_self_steps": selfst["n_self_steps"], "self_reuse": selfst["self_reuse"]})
            log.write(json.dumps(rec_) + "\n"); log.flush(); print(rec_, flush=True); acc = []
        if step in KEEP:
            torch.save({"model": m.state_dict(), "step": step}, os.path.join(a.out, "ckpt_model_%d.pt" % step))
        if step in HEV and not (a.mem_check or a.time_steps):             # Note 218-8: descriptive host candidate accuracy (read by no rule)
            if not HSET:
                s_h = PS().Stream(21170, EXCL); HSET.update({D: [lvb_item(s_h.draw(D)) for _ in range(100)] for D in range(3, 9)})
            m.eval(); ha = {}
            for D, its_ in HSET.items():
                ok_ = []
                for i_ in range(0, len(its_), a.eval_bs):
                    sR, sP = cand_scores(m, its_[i_:i_ + a.eval_bs], D + 2, dev, ac); ok_ += (sR > sP).tolist()
                ha[str(D)] = float(np.mean(ok_))
            log.write(json.dumps({"step": step, "host_cand_acc": ha, "host_set": "Stream 21170, 100 per D 3..8, K' = D + 2"}) + "\n"); log.flush(); print("HOST_ACC", step, ha, flush=True)
        if a.ckpt_every and step % a.ckpt_every == 0 and step < a.steps and not (a.mem_check or a.time_steps):
            cp = os.path.join(a.out, "ckpt_last.pt")
            if os.path.exists(cp):
                os.replace(cp, os.path.join(a.out, "ckpt_prev.pt"))
            torch.save({"model": m.state_dict(), "opt": opt.state_dict(), "np_rng": rng.bit_generator.state, "rng_k": rng_k.bit_generator.state,
                        "torch_rng": torch.get_rng_state(), "stream": stream_state(stream), "gate": GATE.state(), "step": step, "elapsed": time.time() - t0,
                        "n_nonfinite": n_nonfinite, "calibration": cal,
                        **({"rng_s": rng_s.bit_generator.state, "self_state": dict(selfst)} if rng_s is not None else {})}, cp)
            if draws_log:
                json.dump(draws_log, open(os.path.join(a.out, "draws_log.json"), "w", newline="\n"))
    assert selfst["self_reuse"] == 0
    if draws_log:
        json.dump(draws_log, open(os.path.join(a.out, "draws_log.json"), "w", newline="\n"))
    if a.mem_check:
        pk = torch.cuda.max_memory_allocated(dev) / 2 ** 30
        mj = {"peak_gib_train_step": pk, "shape": "self (64 longest records, K %d)" % a.self_kmax if (pool is not None and a.mem_check_self) else
              "plain (64 longest of %d D %d draws, K %d)" % (8 * a.bs, ns[-1], a.kmax), "bs": a.bs, "micro": a.self_micro if a.mem_check_self else a.micro,
              "arch": a.arch, "gpu": info["train_gpu"], "amp": str(amp)}
        for f_ in ("mem_check.json", "eval.json"):
            json.dump(mj, open(os.path.join(a.out, f_), "w", newline="\n"), indent=1)
        print("MEMCHECK", pk, flush=True); return
    if a.time_steps:
        el = time.time() - tt0; sps_self = tself[0] / max(1, tself[1]); sps_plain = tself[2] / max(1, tself[3])
        sps = a.p_self * sps_self + (1 - a.p_self) * sps_plain if pool is not None else el / a.time_steps
        tj = {"arch": a.arch, "steps_timed": a.time_steps, "warmup": a.time_warmup, "sec_per_step": sps, "sec_per_step_wall": el / a.time_steps,
              "est_hours": sps * a.steps / 3600, "gpu": info["train_gpu"], "micro": a.micro, "self_micro": a.self_micro, "p_self": a.p_self,
              "sec_per_step_self": sps_self if tself[1] else None, "sec_per_step_plain": sps_plain if tself[3] else None, "n_self_timed": tself[1],
              "law": "final stage D U{%d..%d}, K min(D + U{0..%d}, %d); self / plain by the registered rng_s coin" % (D_MIN, ns[-1], a.kextra, a.kmax)}
        for f_ in ("timing.json", "eval.json"):
            json.dump(tj, open(os.path.join(a.out, f_), "w", newline="\n"), indent=1)
        print("TIMING", sps, flush=True); return
    if a.stop_at and last < a.steps:
        print("STOPPED_AT", last, flush=True); return
    train_sec = round(time.time() - t0); torch.save(m.state_dict(), os.path.join(a.out, "model.pt"))
    for ck_ in ("ckpt_last.pt", "ckpt_prev.pt"):                            # right after model.pt, before the descriptive eval
        if os.path.exists(os.path.join(a.out, ck_)):
            os.remove(os.path.join(a.out, ck_))
    opt.zero_grad(set_to_none=True); opt.state.clear(); m.eval(); t1 = time.time()
    if dev.startswith("cuda"):
        torch.cuda.empty_cache()
    TE = PS().test_set(a.per_n, range(3, 13)); accD = {}                   # host-side, descriptive: cand accuracy at K' = D + 2 (D 13-16 untouched)
    for D, its_ in TE.items():
        its_ = [lvb_item(it) for it in its_]; ok = []
        for i in range(0, len(its_), a.eval_bs):
            sR, sP = cand_scores(m, its_[i:i + a.eval_bs], D + 2, dev, ac); ok += (sR > sP).tolist()
        accD[str(D)] = float(np.mean(ok))
    res = {"task": "prosqa_sym_v1", "arm": "langvb", "arch": a.arch, "seed": a.seed, "acc_cand_host": accD, "per_n": a.per_n, "train_sec": train_sec,
           "n_nonfinite": n_nonfinite, "gate_hist": GATE.hist, "stream_collisions": len(stream.collisions), "t_train_max_since_resume": tmax[0], "test_gate_sig_hits_since_resume": hits[0], "self_state": selfst if pool is not None else None,
           "eval_dtype": str(amp), **info, "eval_sec": round(time.time() - t1),
           "note": "descriptive host eval (cand accuracy at K' = D + 2, D 3..12, first per_n test items); the registered readings come from langvb_eval (fp32)"}
    json.dump(res, open(os.path.join(a.out, "eval.json"), "w", newline="\n"), indent=1)
    print("EVAL", json.dumps(accD), flush=True); print("DONE", a.out, flush=True)


if __name__ == "__main__":
    main()
