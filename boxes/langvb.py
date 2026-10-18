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
        self.slot_keep = None                                               # eval-only: f(t, s) -> (B, S) bool slots of version s taken from a donor (slot-local splice), set by evals

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
                bk = torch.cat([k for k, _ in kvs[i][1:t]], 2); bv = torch.cat([v for _, v in kvs[i][1:t]], 2)
                bb = self.bias_ls(bp[0][r], bp[1][r], bp[2][r], t, None, spad, t - 1, v0=False).expand(-1, -1, S + L, -1)
                if self.cap is not None:
                    self.cap[("coda", i, t)] = (h.detach(), km)
                h = VB.block_bank(src, i, h, bk, bv, bb, km)
        return F.layer_norm(h, (D,), src.w("ln_f.weight"), src.w("ln_f.bias"))

    def bias_ls(self, b, b0, gm, t, tpad, spad, nslotver, v0=True):
        """the bank bias row read at pass (or coda) index t: [version 0: beta0 on the T token rows (tpad), when v0] + [versions 1..nslotver: beta + gamma
        (t - 1 - s) on the S slot rows (spad)]; -inf where masked; bank_off -> all -inf. -> (B, H, 1, (T if v0) + nslotver * S)."""
        B, S = spad.shape; H = len(b); parts, valid = [], []
        if v0:
            T = tpad.shape[1]; parts.append(b0[:, None].expand(H, T)); valid.append(tpad)
        if nslotver:
            s = torch.arange(1, nslotver + 1, device=spad.device); age = (t - 1 - s).to(b.dtype)
            row = (b[:, None] + gm[:, None] * age[None, :])[:, :, None].expand(H, nslotver, S).reshape(H, nslotver * S); parts.append(row)
            valid.append(spad[:, None, :].expand(B, nslotver, S).reshape(B, nslotver * S))
        row = torch.cat(parts, 1); vl = torch.cat(valid, 1)
        if self.bank_off:
            vl = torch.zeros_like(vl)
        return torch.where(vl[:, None, None, :], row[None, :, None, :], torch.full((), NEG, dtype=row.dtype, device=row.device))

    def rollout_ls(self, x, pad, slot_idx, spad, K, coda_rows=None, dec=None, lens=None, edit=None, sg_input=False):
        """x (B, T) ids right-padded, pad (B, T) bool; slot_idx (B, S) long (padded rows point at 0), spad (B, S) bool; K passes.
        coda_rows {t: rows} with dec {t: (b, L) decoder ids for those rows} (or one (B, L) tensor for every t): the coda after pass t -> out["coda"][t] =
        (rows, h (b, S + L, D)). edit(t, V_t) -> the stored entry (None = V_t). -> {"V": V_0..V_K (slot rows), "S": stored, "coda": {...}, "kv": bank, "H6": H_6}."""
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
            h = self.gather(t, gk_t, NV_t, spad)
            h = h.detach() if sg_input else h
            if self.input_edit is not None:
                h = self.input_edit(t, h)
            for li, i in enumerate(CORE):
                kv0 = kv_sw[i] if sw else kv[i][0]
                bk = torch.cat([kv0[0]] + [k for k, _ in kv[i][1:t]], 2); bv = torch.cat([kv0[1]] + [v for _, v in kv[i][1:t]], 2)
                bb = self.bias_ls(self.beta[li], self.beta0[li], self.gam[li], t, pad, spad, t - 1).expand(-1, -1, S, -1)
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
        lp = torch.log_softmax(self.g.logits(rows).float(), -1)
        return lp.gather(-1, dec[..., None]).squeeze(-1)

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
    ans_sum = torch.zeros(B, device=dev); ans_cnt = torch.zeros(B, device=dev); cc_sum = torch.zeros(B, device=dev)
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
    return L_ans + rec_w * L_rec, st, (r if keep else None)


def make_model(g, table="train"):
    return LangSLVA(g, table)
