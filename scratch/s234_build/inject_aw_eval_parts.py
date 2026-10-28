SWAP_CAP = 1000; SWAP_TRIES = 3000; SEED_SWAP_REC, SEED_SWAP_DON = 23244, 23245   # aw_checks PAIRS (1,000 kept of 1,613; 0 overlap train / TEST2)
SEED_PROBE = 23471; PROBE_N = 3000; P = 11                                  # aw_gate's G-AW0 item stream (d 2, EXCL), 60 / 20 / 20 by sig (rng 0)
SEED_SWAP_REC_REG, SEED_SWAP_DON_REG, SEED_PROBE_REG = 23810, 23811, 23812  # §236 REG (mul_s_p 0) eval-own streams: new seeds (unused, grep 10-27)
SEED_SWAP_REC_DF, SEED_SWAP_DON_DF, SEED_PROBE_DF = 24020, 24021, 24022     # §238 DF (mul_s_p 2) eval-own streams: new seeds (unused, grep 10-27)
SEED_PROD_REC_DF, SEED_PROD_DON_DF = 24050, 24051                          # §238 PROD-SWAP (DF d 3, step 3 = mul_s): recipients / donors' redraw rng


def law(a):
    """§236 / §238: the run's OW law (args mul_s_p; absent = 1). RAISES on anything but 0 / 1 / 2 (DF_v1 §1: no silent fallback)."""
    v = a.get("mul_s_p", 1)
    if v not in (0, 1, 2):
        raise ValueError("unknown mul_s_p law %r (0 REG / 1 default / 2 DF)" % (v,))
    return int(v)


def reg_kw(a):
    """the OW law kwargs of the run: {} for the default law, else {"mul_s_p": law} (§236 REG 0, §238 DF 2)."""
    v = law(a); return {} if v == 1 else {"mul_s_p": v}


def swap_pairs(a):
    """aw_checks.chk_pairs's law: recipients d 2 from Stream(23244), donors OW.redraw(rng 23245) (same skeleton: names / unit / ops); kept iff
    |{own, donor final, cf_a(1)}| = 3, cf_b(1) not in {own, donor final}, equal text token length; the first 1,000. Cues equal (asserted)."""
    sr, sd = {1: (SEED_SWAP_REC, SEED_SWAP_DON), 0: (SEED_SWAP_REC_REG, SEED_SWAP_DON_REG), 2: (SEED_SWAP_REC_DF, SEED_SWAP_DON_DF)}[law(a)]   # per-law seeds
    r2 = np.random.default_rng(sd); s2 = OW.Stream(sr, frames=OW.FRAME_TRAIN, **IO.OWKW, **reg_kw(a)); kept = []; tried = 0
    while len(kept) < SWAP_CAP and tried < SWAP_TRIES:
        rec = s2.draw(2); don = OW.redraw(r2, rec)
        if don is None or don["sig"] == rec["sig"]:
            continue
        tried += 1; own, dfin, ca, cb = rec["answer"], don["answer"], IO.cf(rec, don, 1, "a"), IO.cf(rec, don, 1, "b")
        if len({own, dfin, ca}) == 3 and cb not in (own, dfin) and len(IO.ENC.encode(rec["text"])) == len(IO.ENC.encode(don["text"])):
            kept.append((rec, don, ca, cb))
    for rec, don, _, _ in kept:
        assert all(ST.cue_str(rec, c) == ST.cue_str(don, c) for c in (1, 2)), "SWAP: the cues must be equal (same skeleton)"
    tk = lambda it: AW.tok_aw(it, a.get("aux_target", "value"), a.get("cue", "none"), a.get("write_src", "all"))  # noqa: E731
    return [tk(p_[0]) for p_ in kept], [tk(p_[1]) for p_ in kept], kept, tried


def grab_states(m, items, dev, bs, ts):
    """S_t (fp32, CPU) for t in ts, per item, from the run's own passes."""
    St = {t: [] for t in ts}

    def grab(t, S, sl):
        if t in St:
            assert sl.start == sum(len(x_) for x_ in St[t]), "grab_states: batches in order"; St[t].append(S.float().cpu())
    passes(m, items, dev, bs, {max(ts)}, on_state=grab); return {t: torch.cat(v) for t, v in St.items()}


def dec_with(m, items, Sx, dev, bs, shared=False):
    """the decoder on items' own dec tokens reading per-item states Sx (N, H, DH, DH) or (shared) one state (H, DH, DH) for every item."""
    p_ = []; assert Sx.dim() == (3 if shared else 4) and (shared or len(Sx) == len(items))
    for i in range(0, len(items), bs):
        ch = items[i:i + bs]; S_ = Sx[None].expand(len(ch), *Sx.shape) if shared else Sx[i:i + len(ch)]
        p_.append(pred_of(m.dec_logits(S_.to(dev).contiguous(), ch, dev)))
    return np.concatenate(p_)


@torch.no_grad()
def part_swap_k(m, g, sets, a, dev, bs):
    K = a["think_k"]; R, Dn, kept, tried = swap_pairs(a); own = np.array([p_[0]["answer"] for p_ in kept]); dfin = np.array([p_[1]["answer"] for p_ in kept])
    ca = np.array([p_[2] for p_ in kept]); SD = grab_states(m, Dn, dev, bs, (K,))[K]; I, _ = passes(m, R, dev, bs, {K})
    sw = dec_with(m, R, SD, dev, bs); mn = dec_with(m, R, SD.mean(0), dev, bs, shared=True); inU = lambda p: (p == dfin) | (p == ca)  # noqa: E731
    out = {"n_pairs": len(kept), "tried": tried, "follow_U": float(inU(sw).mean()), "floor": float(inU(mn).mean()), "follow_dfin": float((sw == dfin).mean()),
           "follow_ca": float((sw == ca).mean()), "stay_own": float((sw == own).mean()), "intact_acc": float((I[K] == own).mean()), "mean_acc": float((mn == own).mean()),
           "k": K, "cfa_eq_cfb": int(sum(int(p_[2] == p_[3]) for p_ in kept)),
           "items": {"rec_sig": [p_[0]["sig"] for p_ in kept], "don_sig": [p_[1]["sig"] for p_ in kept], "own": own.tolist(), "dfin": dfin.tolist(), "ca": ca.tolist(),
                     "cb": [p_[3] for p_ in kept], "intact": I[K].tolist(), "swap": sw.tolist(), "mean": mn.tolist()},
           "def": ("SWAP-K: recipient's S_K := donor's S_K (K = think_k), the decoder on the recipient's own dec tokens; U = {donor final, cf_a(t = 1)}; follow_U = share "
                   "of swap answers in U; floor = share in U of the MEAN-K answers (S_K := the mean of the 1,000 donors' S_K) on the same recipients; binding "
                   "follow_U >= 0.50 and >= floor + 0.30 (the scorer's)")}
    return out


@torch.no_grad()
def part_swap_1(m, g, sets, a, dev, bs):
    K = a["think_k"]; R, Dn, kept, tried = swap_pairs(a); own = np.array([p_[0]["answer"] for p_ in kept]); dfin = np.array([p_[1]["answer"] for p_ in kept])
    ca = np.array([p_[2] for p_ in kept]); S1 = grab_states(m, Dn, dev, bs, (1,))[1]; ks = sorted({2, K})
    A, _ = passes(m, R, dev, bs, set(ks), edit=lambda t, S, sl: S1[sl].to(dev, S.dtype) if t == 1 else None)
    out = {"n_pairs": len(kept), "by_k": {str(k): {"follow_ca": float((A[k] == ca).mean()), "follow_dfin": float((A[k] == dfin).mean()), "stay_own": float((A[k] == own).mean())}
                                          for k in ks},
           "items": {"rec_sig": [p_[0]["sig"] for p_ in kept], "own": own.tolist(), "dfin": dfin.tolist(), "ca": ca.tolist(), "swap1": {str(k): A[k].tolist() for k in ks}},
           "def": "SWAP-1 (descriptive): recipient's S_1 := donor's S_1, passes 2..K' on the recipient's own inputs; follow cf_a(1) at K' = 2 and K' = think_k"}
    return out


def fit_probe(X, y, tr, va, te):
    """aw_gate.fit_probe (Audit's 231-1 protocol), CPU."""
    import torch.nn.functional as Fn
    mu = X[tr].mean(0, keepdim=True); sd = X[tr].std(0, keepdim=True) + 1e-6; Z = (X - mu) / sd; Y = torch.tensor(y); best = None
    for wd in (1e-4, 1e-3, 1e-2):
        W = torch.zeros(Z.shape[1], P, requires_grad=True); b = torch.zeros(P, requires_grad=True)
        opt = torch.optim.LBFGS([W, b], lr=1, max_iter=200, history_size=20, line_search_fn="strong_wolfe")

        def clo():
            opt.zero_grad(); l_ = Fn.cross_entropy(Z[tr] @ W + b, Y[tr]) + 0.5 * wd * (W ** 2).sum(); l_.backward(); return l_
        with torch.enable_grad():
            opt.step(clo)
        with torch.no_grad():
            pr = (Z @ W + b).argmax(-1); va_acc = float((pr[va] == Y[va]).float().mean())
            if best is None or va_acc > best[0]:
                best = (va_acc, wd, pr[te].numpy(), float((pr[tr] == Y[tr]).float().mean()))
    yt = y[te]
    return {"test_acc": float((best[2] == yt).mean()), "val_acc": best[0], "train_acc": best[3], "wd": best[1], "n_test": int(len(te)),
            "chance_majority": float(np.bincount(yt, minlength=P).max() / len(yt)), "chance_uniform": 1 / P, "test_pred": best[2].tolist(), "feature_dim": int(X.shape[1])}


def read_vec(mem, h, S):
    """deltamem.read_fixed's pre-W_o read [oa || od] for rows h (B, D) on state S (no fade / W_o)."""
    H_ADD, H_DEL = mem.h_add, mem.h_del; B, D = h.shape
    a_ = mem.ln(h.float())[:, None]; q = mem.qkv(a_)[..., :D].view(B, 1, H_ADD + H_DEL, DM.DH)
    oa = torch.exp(mem.s_add) * DM.add_read(mem, torch.nn.functional.elu(q[:, :, :H_ADD]) + 1, S[:, :H_ADD].float(), DM.state_z(mem, S))
    od = torch.exp(mem.s_del) * torch.einsum("bhvk,bthk->bthv", S[:, H_ADD:H_ADD + H_DEL].float(), torch.nn.functional.normalize(q[:, :, H_ADD:], dim=-1))
    return torch.cat([oa, od], 2).reshape(B, D)


@torch.no_grad()
def part_probe(m, g, sets, a, dev, bs):
    """the end-of-run probes on the TRAINED model (d 2, aw_gate's item stream): u1|v1 = mem.ln_w(h_w) at ':' of pass 1 (S_0 = 0); read1|v1 = the
    pre-W_o read [oa || od] at the decoder ':' from S_1; l0|v1 control (wte + wpe at ':'); the trunk's own argmax at pass-1 ':' (= WS pass 1); the
    pass-1 / pass-2 write-key cosine at ':' (add heads: phi(k); delta heads: normalised k)."""
    assert float(m.mem.gamma) == 1.0, "§234 probes: gamma = 1 (the trained AW model)"
    s = OW.Stream({1: SEED_PROBE, 0: SEED_PROBE_REG, 2: SEED_PROBE_DF}[law(a)], sets["excl"], frames=OW.FRAME_TRAIN, **IO.OWKW, **reg_kw(a)); raw = [s.draw(2) for _ in range(PROBE_N)]
    T = [AW.tok_aw(it, a.get("aux_target", "value"), a.get("cue", "none"), a.get("write_src", "all")) for it in raw]
    mem = m.mem; D = g.w("wte.weight").shape[1]; H_ADD = mem.h_add; F = {"u1": [], "read1": [], "l0": []}; beh = []; cos_a = []; cos_d = []; chk = 0.0
    num = torch.tensor(NUM_IDS, device=dev)
    for i in range(0, len(T), bs):
        ch = T[i:i + bs]; B = len(ch); ar = torch.arange(B, device=dev); xs, msk = AW.cue_inputs(m, ch, dev); S0 = mem.zero_state(B, dev)
        S1, h1, p1 = m.pass_aw(S0, xs[0], msk); _, h2, p2 = m.pass_aw(S1, xs[1], msk)
        u1 = mem.ln_w(h1[ar, p1].float()); u2 = mem.ln_w(h2[ar, p2].float()); F["u1"].append(u1.cpu())
        beh.append(m.colon_logits(h1, p1)[:, num].argmax(-1).cpu())
        k1 = mem.qkv(u1)[:, D:2 * D].view(B, -1, DM.DH); k2 = mem.qkv(u2)[:, D:2 * D].view(B, -1, DM.DH)
        ph = lambda k: torch.nn.functional.elu(k) + 1  # noqa: E731
        cos_a.append(torch.nn.functional.cosine_similarity(ph(k1[:, :H_ADD]), ph(k2[:, :H_ADD]), dim=-1).mean(1).cpu())
        cos_d.append(torch.nn.functional.cosine_similarity(k1[:, H_ADD:], k2[:, H_ADD:], dim=-1).mean(1).cpu())
        ids_, _ = pad([it["dec"] for it in ch]); xd = torch.tensor(ids_, device=dev); dp = torch.tensor([len(it["dec"]) - 1 for it in ch], device=dev)
        hl = DM.gpt_lower(g, g.embed(xd), mem.layer)[ar, dp]; rv = read_vec(mem, hl, S1); F["read1"].append(rv.cpu())
        rc_ = []; ref = DM.read_fixed(mem, hl[:, None], S1, rc_)[:, 0]               # read_fixed's own capture of oa (last token) + its full output
        chk = max(chk, float((rc_[0] - rv[:, :H_ADD * DM.DH]).abs().max() / rc_[0].abs().max().clamp_min(1e-30)),
                  float((ref - (hl + mem.fade * mem.wo(rv)).to(hl.dtype)).abs().max() / ref.abs().max().clamp_min(1e-30)))
        F["l0"].append((g.embed(xs[0]) + DM.wpe_slice(g, xs[0].shape[1])[None])[ar, p1].float().cpu())
    X = {k: torch.cat(v) for k, v in F.items()}; y1 = np.array([int(it["values"][1]) for it in raw]); bh = torch.cat(beh).numpy()
    sigs = [it["sig"] for it in raw]; us = sorted(set(sigs)); perm = np.random.default_rng(0).permutation(len(us)); rank = {us[j]: k for k, j in enumerate(perm)}
    rr = np.array([rank[s_] for s_ in sigs]) / len(us); tr, va, te = [np.where(c)[0] for c in (rr < 0.6, (rr >= 0.6) & (rr < 0.8), rr >= 0.8)]
    pr = {nm + "|v1": fit_probe(X[nm], y1, tr, va, te) for nm in ("u1", "read1", "l0")}
    out = {"probes": pr, "behaviour_pass1_argmax_acc_v1": float((bh == y1).mean()), "behaviour_pass1_test_acc_v1": float((bh[te] == y1[te]).mean()),
           "key_cos_pass1_pass2": {"add_phi_mean": float(torch.cat(cos_a).mean()), "delta_mean": float(torch.cat(cos_d).mean())}, "read_vec_check_rel": chk,
           "n": len(raw), "split": {"train": int(len(tr)), "val": int(len(va)), "test": int(len(te))},
           "items": {"sig": sigs, "v1": y1.tolist(), "test_idx": te.tolist(), "beh_pass1": bh.tolist(), "key_cos_add": torch.cat(cos_a).tolist(), "key_cos_delta": torch.cat(cos_d).tolist()},
           "def": ("end-of-run probes (AW_v1 §3, fix 5): u1|v1 separates AW-SOURCE-DRIFT (the ':' code eroded; G-AW0 init 1.000) from AW-NOT-STORED; read1|v1 = the "
                   "pre-W_o read at the decoder ':' from S_1 (replaces s1|v1); l0|v1 control; items: OW.Stream(23471, EXCL) d 2 x 3,000 (aw_gate's stream, "
                   "no skip filter); probe = Audit's 231-1 protocol")}
    assert chk <= 1e-5, ("read_vec != read_fixed's read", chk); return out


@torch.no_grad()
def part_ws(m, g, sets, a, dev, bs):
    K = a["think_k"]; T = tokd(sets, a, range(1, a["d_max"] + 1)); r_, pv = AW.ws_readout(m, T, dev, K, bs, per_item=True)   # Note 234-5 (c): d 1..d_max
    return {"by_d": r_, "items": {str(d): items_rec(T[d], {"pred_by_pass": pv[str(d)], "v": [[int(it["raw"]["values"][min(t, d)]) for t in range(1, K + 1)] for it in T[d]]})
                                  for d in T},
            "def": "WS-READOUT (descriptive): the trunk's argmax over the 11 number tokens at ':' of pass t vs v_min(t, d), TEST2 d 1..d_max (all 1,000 per d), fp32"}


SEED_SWAPD_REC, SEED_SWAPD_DON = 23260, 23270                              # §234 P-4 (Note 234-5 b): recipients Stream(23260 + d), donors redraw rng (23270 + d), d 3 / 4
SEED_SWAPD_REC_REG, SEED_SWAPD_DON_REG = 23820, 23830                      # §236 REG: (+ d), new seeds
SEED_SWAPD2_REC_REG, SEED_SWAPD2_DON_REG = 23840, 23850                    # §236 REG d 3 at t 2 (by_d.3.swap_mid_t2, descriptive): (+ d), new seeds
SEED_SWAPD_REC_DF, SEED_SWAPD_DON_DF = 24030, 24040                        # §238 DF: (+ d), new seeds


def swap_t(d):
    """Note 234-5 (b) + amendment: the swap step t = ceil(d / 2), except d 3 -> t 1 (at t 2 step 3 is a state reference, so cf_a(2) = donor final: 0 kept)."""
    return 1 if d == 3 else math.ceil(d / 2)


def swap_pairs_d(a, d, t=None, seeds=None):
    """Note 234-5 (b) + amendment: the PAIRS law at depth d with t = swap_t(d) (ceil(d / 2); d 3 -> 1): recipients d from Stream(23260 + d), donors OW.redraw(rng 23270 + d); kept iff
    |{own, donor final, cf_a(t)}| = 3, cf_b(t) not in {own, donor final}, equal text token length; the first 1,000 (3,000 tries). Cues equal (asserted)."""
    t = swap_t(d) if t is None else t                                      # §236: t / seeds overridable (REG d 3 at t 2); default = P-4's
    sr, sd = seeds if seeds is not None else {1: (SEED_SWAPD_REC + d, SEED_SWAPD_DON + d), 0: (SEED_SWAPD_REC_REG + d, SEED_SWAPD_DON_REG + d),
                                              2: (SEED_SWAPD_REC_DF + d, SEED_SWAPD_DON_DF + d)}[law(a)]
    r2 = np.random.default_rng(sd); s2 = OW.Stream(sr, frames=OW.FRAME_TRAIN, **IO.OWKW, **reg_kw(a)); kept = []; tried = 0
    while len(kept) < SWAP_CAP and tried < SWAP_TRIES:
        rec = s2.draw(d); don = OW.redraw(r2, rec)
        if don is None or don["sig"] == rec["sig"]:
            continue
        tried += 1; own, dfin, ca, cb = rec["answer"], don["answer"], IO.cf(rec, don, t, "a"), IO.cf(rec, don, t, "b")
        if len({own, dfin, ca}) == 3 and cb not in (own, dfin) and len(IO.ENC.encode(rec["text"])) == len(IO.ENC.encode(don["text"])):
            kept.append((rec, don, ca, cb))
    for rec, don, _, _ in kept:
        assert all(ST.cue_str(rec, c) == ST.cue_str(don, c) for c in range(1, d + 1)), "SWAP: the cues must be equal (same skeleton)"
    tk = lambda it: AW.tok_aw(it, a.get("aux_target", "value"), a.get("cue", "none"), a.get("write_src", "all"))  # noqa: E731
    return [tk(p_[0]) for p_ in kept], [tk(p_[1]) for p_ in kept], kept, tried, t


@torch.no_grad()
def part_swap_kd(m, g, sets, a, dev, bs):
    """Note 234-5 (b) + amendment, descriptive, d 3..d_max, t = swap_t(d): SWAP-K (S_K := donor's S_K; U = {donor final, cf_a(t)}; floor = MEAN-K, the donors' mean S_K) and
    SWAP-MID (S after pass t := donor's, the recipient's own later passes; U = {cf_a(t), cf_b(t)}; floor = MEAN at t, the donors' mean S_t), K' = d and K."""
    K = a["think_k"]; out = {"by_d": {}, "items": {}}
    for d in [d_ for d_ in range(3, a["d_max"] + 1)]:
        R, Dn, kept, tried, t = swap_pairs_d(a, d)
        if not kept:                                                        # guard: no kept pair (d 3 at t 2 was structurally empty; swap_t uses t 1 there)
            out["by_d"][str(d)] = {"n_pairs": 0, "tried": tried, "t_mid": t, "note": "no kept pair under the PAIRS law"}; continue
        own = np.array([p_[0]["answer"] for p_ in kept]); dfin = np.array([p_[1]["answer"] for p_ in kept])
        ca = np.array([p_[2] for p_ in kept]); cb = np.array([p_[3] for p_ in kept]); ks = sorted({d, K})
        SDs = grab_states(m, Dn, dev, bs, (t, K)); SD, ST_ = SDs[K], SDs[t]; I, _ = passes(m, R, dev, bs, set(ks))
        sw = dec_with(m, R, SD, dev, bs); mn = dec_with(m, R, SD.mean(0), dev, bs, shared=True); mt = ST_.mean(0)
        M_, _ = passes(m, R, dev, bs, set(ks), edit=lambda t_, S, sl: ST_[sl].to(dev, S.dtype) if t_ == t else None)
        F_, _ = passes(m, R, dev, bs, set(ks), edit=lambda t_, S, sl: mt.to(dev, S.dtype)[None].expand(S.shape[0], *mt.shape).contiguous() if t_ == t else None)
        inK = lambda p: (p == dfin) | (p == ca); inM = lambda p: (p == ca) | (p == cb)  # noqa: E731
        out["by_d"][str(d)] = {"n_pairs": len(kept), "tried": tried, "t_mid": t, "intact_acc": {str(k): float((I[k] == own).mean()) for k in ks},
                               "swap_k": {"follow_U": float(inK(sw).mean()), "floor": float(inK(mn).mean()), "follow_dfin": float((sw == dfin).mean()),
                                          "follow_ca": float((sw == ca).mean()), "stay_own": float((sw == own).mean())},
                               "swap_mid": {str(k): {"follow_U": float(inM(M_[k]).mean()), "floor": float(inM(F_[k]).mean()), "follow_ca": float((M_[k] == ca).mean()),
                                                     "follow_cb": float((M_[k] == cb).mean()), "stay_own": float((M_[k] == own).mean())} for k in ks}}
        out["items"][str(d)] = {"rec_sig": [p_[0]["sig"] for p_ in kept], "don_sig": [p_[1]["sig"] for p_ in kept], "own": own.tolist(), "dfin": dfin.tolist(), "ca": ca.tolist(),
                                "cb": cb.tolist(), "intact": {str(k): I[k].tolist() for k in ks}, "swap_k": sw.tolist(), "mean_k": mn.tolist(),
                                "swap_mid": {str(k): M_[k].tolist() for k in ks}, "mean_mid": {str(k): F_[k].tolist() for k in ks}}
        if d == 3 and law(a) == 0:                                          # §236 (RC_v1 §1): REG d 3 also at t 2 (SWAP-MID only, descriptive; REG only)
            R2, Dn2, kp2, tr2, t2 = swap_pairs_d(a, 3, 2, (SEED_SWAPD2_REC_REG + 3, SEED_SWAPD2_DON_REG + 3))
            out["by_d"]["3"].update({"n_pairs_t2": len(kp2), "tried_t2": tr2})
            if kp2:
                own2 = np.array([p_[0]["answer"] for p_ in kp2]); ca2 = np.array([p_[2] for p_ in kp2]); cb2 = np.array([p_[3] for p_ in kp2])
                S2_ = grab_states(m, Dn2, dev, bs, (t2,))[t2]; mt2 = S2_.mean(0)
                M2, _ = passes(m, R2, dev, bs, set(ks), edit=lambda t_, S, sl: S2_[sl].to(dev, S.dtype) if t_ == t2 else None)
                F2, _ = passes(m, R2, dev, bs, set(ks), edit=lambda t_, S, sl: mt2.to(dev, S.dtype)[None].expand(S.shape[0], *mt2.shape).contiguous() if t_ == t2 else None)
                inM2 = lambda p: (p == ca2) | (p == cb2)  # noqa: E731
                out["by_d"]["3"]["swap_mid_t2"] = {str(k): {"follow_U": float(inM2(M2[k]).mean()), "floor": float(inM2(F2[k]).mean()), "follow_ca": float((M2[k] == ca2).mean()),
                                                            "follow_cb": float((M2[k] == cb2).mean()), "stay_own": float((M2[k] == own2).mean())} for k in ks}
                out["items"]["3"]["t2"] = {"rec_sig": [p_[0]["sig"] for p_ in kp2], "don_sig": [p_[1]["sig"] for p_ in kp2], "own": own2.tolist(),
                                           "dfin": [p_[1]["answer"] for p_ in kp2], "ca": ca2.tolist(), "cb": cb2.tolist(),
                                           "swap_mid": {str(k): M2[k].tolist() for k in ks}, "mean_mid": {str(k): F2[k].tolist() for k in ks}}
    out["def"] = ("Note 234-5 (b) + amendment, descriptive, d >= 3: pairs = the PAIRS law at d with t = ceil(d/2), d 3 -> t 1 (recipients Stream(23260 + d), donors redraw rng 23270 + d, cap 1,000); "
                  "SWAP-K: S_K := donor's S_K, U = {donor final, cf_a(t)}, floor = MEAN-K (the donors' mean S_K); SWAP-MID: S after pass t := donor's S_t, the recipient's own "
                  "later passes, U = {cf_a(t), cf_b(t)}, floor = MEAN at t (the donors' mean S_t); answers at K' = d and think_k. §236 REG (args mul_s_p 0): every pair stream "
                  "under the REG law from new seeds (recipients 23820 + d, donors 23830 + d); d 3 also at t 2 (by_d.3.swap_mid_t2 / n_pairs_t2, items.3.t2; seeds 23843 / 23853)")
    return out


def law_eval_sigs(a, excl):
    """§236 / §238: {set name: sigs (recipients + donors / items)} of the eval module's OWN streams under the run's non-default law (args a: d_max, mul_s_p) -
    swap_k d 2, swap_kd d 3..d_max (+ REG: d 3 at t 2), probe (drawn excluding excl = build_sets(law)["excl"]); DF also OPCLASS-DF and the PROD-SWAP pairs.
    The training EXCL and the disjointness gates use it."""
    assert law(a) != 1, "law_eval_sigs: non-default laws only"
    out = {}; _, _, kp, _ = swap_pairs(a); out["swap_k_d2"] = {x["sig"] for p_ in kp for x in p_[:2]}
    for d in range(3, int(a["d_max"]) + 1):
        _, _, kp, _, _ = swap_pairs_d(a, d); out["swap_kd_d%d" % d] = {x["sig"] for p_ in kp for x in p_[:2]}
    if int(a["d_max"]) >= 3 and law(a) == 0:
        _, _, kp, _, _ = swap_pairs_d(a, 3, 2, (SEED_SWAPD2_REC_REG + 3, SEED_SWAPD2_DON_REG + 3)); out["swap_kd_d3_t2"] = {x["sig"] for p_ in kp for x in p_[:2]}
    s = OW.Stream({0: SEED_PROBE_REG, 2: SEED_PROBE_DF}[law(a)], excl, frames=OW.FRAME_TRAIN, **IO.OWKW, **reg_kw(a)); out["probe"] = {s.draw(2)["sig"] for _ in range(PROBE_N)}
    if law(a) == 2:                                                         # §238: the DF-only eval sets
        out["opclass"] = {it["sig"] for it in IO.opclass_set()}
        _, _, kp, _ = prod_pairs(a); out["swap_prod"] = {x["sig"] for p_ in kp for x in p_[:2]}
    return out


reg_eval_sigs = law_eval_sigs                                               # §236 name (inject_aw's reg_excl calls it)


def prod_pairs(a):
    """§238 PROD-SWAP pairs (DF_v1 §2): d 3 recipients from OW.Stream(24050, frames 0-5, topo, dag, mul_s_p 2) conditioned on step 3 = mul_s (rejection; its partner is
    j = 1), donors OW.redraw(rng 24051) (same skeleton, so step 3 = mul_s on j = 1 too); kept iff own != donor final and cb not in {own, donor final}, cb = cf_b(2) =
    donor S_2 x recipient S_1; the first 1,000 (3,000 tries); cues equal (asserted). -> tokenised recipients, donors, kept [(rec, don, cb)], tried."""
    assert law(a) == 2, "PROD-SWAP: DF runs only"
    r2 = np.random.default_rng(SEED_PROD_DON_DF); s2 = OW.Stream(SEED_PROD_REC_DF, frames=OW.FRAME_TRAIN, **IO.OWKW, mul_s_p=2); kept = []; tried = 0
    while len(kept) < SWAP_CAP and tried < SWAP_TRIES:
        rec = s2.draw(3)
        if rec["ops"][2] != "mul_s":
            continue
        don = OW.redraw(r2, rec)
        if don is None or don["sig"] == rec["sig"]:
            continue
        tried += 1; own, dfin, cb = rec["answer"], don["answer"], IO.cf(rec, don, 2, "b")
        assert don["ops"][2] == "mul_s" and rec["src_j"][2] == don["src_j"][2] == 1 and cb == OW.apply("mul_s", don["values"][2], rec["values"][1]), "PROD-SWAP law"
        if own != dfin and cb not in (own, dfin):
            kept.append((rec, don, cb))
    for rec, don, _ in kept:
        assert all(ST.cue_str(rec, c) == ST.cue_str(don, c) for c in (1, 2, 3)), "PROD-SWAP: the cues must be equal (same skeleton)"
    tk = lambda it: AW.tok_aw(it, a.get("aux_target", "value"), a.get("cue", "none"), a.get("write_src", "all"))  # noqa: E731
    return [tk(p_[0]) for p_ in kept], [tk(p_[1]) for p_ in kept], kept, tried


@torch.no_grad()
def part_swap_prod(m, g, sets, a, dev, bs):
    """§238 PROD-SWAP (DF_v1 §2): the recipient's memory after pass 2 := the donor's S_2, then passes 3..K' on the recipient's own inputs (pass 3 = its own cue);
    answers at K' = 3 (= d; the flat fields) and think_k (by_k). follow_dfin = both factors from memory; follow_cb = S_2 from memory, the partner S_1 recomputed from
    the text; follow_U = follow_dfin + follow_cb (disjoint by the kept rule); floor = the share in {dfin, cb} with S_2 := the donors' mean S_2 (MEAN at pass 2)."""
    K = a["think_k"]; R, Dn, kept, tried = prod_pairs(a); own = np.array([p_[0]["answer"] for p_ in kept]); dfin = np.array([p_[1]["answer"] for p_ in kept])
    cb = np.array([p_[2] for p_ in kept]); ks = sorted({3, K}); S2 = grab_states(m, Dn, dev, bs, (2,))[2]; m2 = S2.mean(0)
    I, _ = passes(m, R, dev, bs, set(ks))
    M, _ = passes(m, R, dev, bs, set(ks), edit=lambda t_, S, sl: S2[sl].to(dev, S.dtype) if t_ == 2 else None)
    F, _ = passes(m, R, dev, bs, set(ks), edit=lambda t_, S, sl: m2.to(dev, S.dtype)[None].expand(S.shape[0], *m2.shape).contiguous() if t_ == 2 else None)
    inU = lambda p: (p == dfin) | (p == cb)  # noqa: E731
    by_k = {str(k): {"follow_dfin": float((M[k] == dfin).mean()), "follow_cb": float((M[k] == cb).mean()), "follow_U": float(inU(M[k]).mean()),
                     "floor": float(inU(F[k]).mean()), "stay_own": float((M[k] == own).mean()), "intact_acc": float((I[k] == own).mean())} for k in ks}
    return {"n_pairs": len(kept), "tried": tried, **by_k["3"], "k_flat": 3, "by_k": by_k,
            "items": {"rec_sig": [p_[0]["sig"] for p_ in kept], "don_sig": [p_[1]["sig"] for p_ in kept], "own": own.tolist(), "dfin": dfin.tolist(), "cb": cb.tolist(),
                      "intact": {str(k): I[k].tolist() for k in ks}, "swap": {str(k): M[k].tolist() for k in ks}, "mean": {str(k): F[k].tolist() for k in ks}},
            "def": ("PROD-SWAP (§238 DF_v1 §2): DF d 3 pairs with step 3 = mul_s (S_3 = S_2 x S_1), recipients Stream(24050) / donors redraw rng 24051, kept iff own != dfin and "
                    "cb not in {own, dfin} (cb = donor S_2 x recipient S_1), cap 1,000; S after pass 2 := donor's; flat fields at K' = 3, by_k also K' = think_k; "
                    "floor: S after pass 2 := the donors' mean S_2")}


@torch.no_grad()
def part_opclass(m, g, sets, a, dev, bs):
    """§238 OPCLASS-DF (DF_v1 §2): the WS-READOUT (trunk argmax at ':' of pass t vs v_min(t, 4)) on the 3,000 d 4 items of inject_ow.opclass_set(), in the ws item
    format: items {sig, ans, pred_by_pass {"1".."8"}, v}; by_pass acc. For the pass-4 per-class readings (the scorer splits mul_s by j)."""
    K = a["think_k"]; its = [AW.tok_aw(it, a.get("aux_target", "value"), a.get("cue", "none"), a.get("write_src", "all")) for it in IO.opclass_set()]
    r_, pv = AW.ws_readout(m, {4: its}, dev, K, bs, per_item=True)
    return {"by_pass": r_["4"], "items": items_rec(its, {"pred_by_pass": pv["4"], "v": [[int(it["raw"]["values"][min(t, 4)]) for t in range(1, K + 1)] for it in its]}),
            "n": len(its), "def": "OPCLASS-DF (§238): Stream(24015, frames 0-5, topo, dag, mul_s_p 2) x 3,000 at d 4 (no EXCL; inject_ow.opclass_set), WS-READOUT per pass, fp32"}


@torch.no_grad()
def part_lp5h(m, g, sets, a, dev, bs):
    """Note 234-5 (a), LP5-4: MEAN-AT-ceil(d/2) on TEST2 d 3..d_max: S after pass ceil(d/2) := the TEST2-d mean of S_ceil(d/2), the other passes on their own cues;
    answers at K' = d (binding, pooled over d 3-4 by the scorer) and K' = think_k, + the intact answers from the same code path; per item."""
    K = a["think_k"]; ds = [d_ for d_ in range(3, a["d_max"] + 1)]; T = tokd(sets, a, ds); out = {"by_d": {}, "items": {}}
    for d in ds:
        t = math.ceil(d / 2); acc_ = {"sum": None}

        def grab(t_, S, sl):
            if t_ == t:
                s_ = S.float().sum(0); acc_["sum"] = s_ if acc_["sum"] is None else acc_["sum"] + s_
        passes(m, T[d], dev, bs, {t}, on_state=grab); mean_t = acc_["sum"] / len(T[d])
        ks = sorted({d, K}); y = np.array([it["ans"] for it in T[d]])
        I, _ = passes(m, T[d], dev, bs, set(ks))
        M, _ = passes(m, T[d], dev, bs, set(ks), edit=lambda t_, S, sl: mean_t.to(S.dtype)[None].expand(S.shape[0], *S.shape[1:]).contiguous() if t_ == t else None)
        out["by_d"][str(d)] = {"t_mean": t, "intact": {str(k): float((I[k] == y).mean()) for k in ks}, "mean_at_t": {str(k): float((M[k] == y).mean()) for k in ks},
                               "k_binding": d, "mean_state_norm": float(mean_t.norm())}
        out["items"][str(d)] = items_rec(T[d], {"intact": {str(k): I[k].tolist() for k in ks}, "mean_at_t": {str(k): M[k].tolist() for k in ks}})
    out["def"] = "LP5-4 MEAN-AT-ceil(d/2) (Note 234-5 a): S after pass ceil(d/2) := the TEST2-d mean of S_ceil(d/2); binding at K' = d, pooled over d 3-4 (drop >= 0.30)"
    return out




@torch.no_grad()
def part_long(m, g, sets, a, dev, bs, kplist, ds):
    """§239 LONG-RUN (LR_v1 §2): on TEST2 of the run's law, d in ds, ONE pass sequence per batch up to max(kplist) (every K' from the same call, Note 239-1): answers
    at each K'; s_norm[K'] = the mean ||S_K'|| (Frobenius over the whole state); s_cos[K'] = the mean cos(S_K', S_8) (flattened, per item); nonfinite = items whose
    S_K' has a non-finite entry. kplist must hold 8 and nothing below it (the cos reference)."""
    assert 8 in kplist and min(kplist) >= 8, ("kprime_list needs 8 and K' >= 8", kplist)
    T = tokd(sets, a, ds); out = {"by_d": {}, "items": {}}
    for d in ds:
        nrm = {k: [] for k in kplist}; cs = {k: [] for k in kplist}; nf = {k: 0 for k in kplist}; ref = {}

        def grab(t, S, sl):
            if t == 8:
                ref["S"] = S.float().flatten(1).clone()
            if t in nrm:
                f = S.float().flatten(1); nrm[t].append(f.norm(dim=1).cpu()); cs[t].append(torch.nn.functional.cosine_similarity(f, ref["S"], dim=1).cpu())
                nf[t] += int((~torch.isfinite(f)).any(dim=1).sum())
        A, _ = passes(m, T[d], dev, bs, set(kplist), on_state=grab); y = np.array([it["ans"] for it in T[d]])
        out["by_d"][str(d)] = {"acc": {str(k): float((A[k] == y).mean()) for k in kplist}, "s_norm": {str(k): float(torch.cat(nrm[k]).mean()) for k in kplist},
                               "s_cos": {str(k): float(torch.cat(cs[k]).mean()) for k in kplist}, "nonfinite": {str(k): nf[k] for k in kplist}}
        out["items"][str(d)] = items_rec(T[d], {"pred": {str(k): A[k].tolist() for k in kplist}})
    out["def"] = ("LONG-RUN (§239 LR_v1 §2): TEST2 of the run's law, d " + ",".join(str(d) for d in ds) + "; one pass sequence to max K' per batch; acc / s_norm (mean ||S_K'||) / "
                  "s_cos (mean cos(S_K', S_8)) per K'; nonfinite = items with a non-finite S_K' entry (COLLAPSE by LR_v1 §5)")
    return out


def run_long(out, dev, bs, kplist, ds, name="eval_long.json"):
    """§239: load <out>/model.pt as load() does, build the run law's sets (set shas asserted vs args.json), part_long, write ONLY <out>/<name> (default
    eval_long.json; §240: eval_long_k128.json)."""
    t1 = time.time(); a, m, g = load(out, dev); assert m is not None, "LONG-RUN: inject runs only"; sets = IO.build_sets(mul_s_p=law(a))
    for k_, v_ in sets["shas"].items():
        assert a.get(k_) in (None, v_), ("§239: set sha mismatch vs args.json", k_)
    if str(dev).startswith("cuda"):
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(dev)
    r_ = part_long(m, g, sets, a, dev, bs, kplist, ds)
    r_.update({"eval_dtype": "fp32", "eval_gpu": torch.cuda.get_device_name(dev) if str(dev).startswith("cuda") else "cpu", "eval_bs": bs, "kprime_list": kplist, "ds": ds,
               "law": law(a), "set_law": sets["shas"].get("set_law", "default"), "think_k": a["think_k"], "module_sha256": IO.sha_file(os.path.abspath(__file__)),
               "peak_gib": torch.cuda.max_memory_allocated(dev) / 2 ** 30 if str(dev).startswith("cuda") else None, "sec": round(time.time() - t1)})
    json.dump(r_, open(os.path.join(out, name), "w", newline="\n")); print("LONG done", out, name, r_["sec"], "s", flush=True)


SEED_LS_A, SEED_LS_C, SEED_LS_B = 24100, 24101, 24102                      # §241 LEN-STEP (LS_v1 §2 + Note 241-1): A / C streams, B's re-render rng (unused, grep 10-28)
LS_N, LS_DPOOL, LS_DMIN, LS_EMAX = 1000, 5000, 300, 20                     # items per cell; D's candidate programs (C's stream); D's floor; B's extra-distractor search range


def rerender_dis(rng, it, n_extra=0, drop=False):
    """§241 (Note 241-1 (1)): the SAME program re-rendered with only the distractor sentences changed. drop: every distractor removed (no rng); else n_extra new
    distractors (names from the pool unused by the item, values U{1..P-1}), each inserted at a uniform slot of the kept statement order (distractors have no
    deps, so any slot keeps the order a linear extension). Asserts identical ops / operands / src_j / values / answer / mul_s_t / frames / names (bar distractors)."""
    sp, gn, dn, dv, order = it["names"]["spine"], it["names"]["givens"], list(it["names"]["distractors"]), list(it["distractor_values"]), list(it["order"])
    if drop:
        dn, dv, order = [], [], [x for x in order if not x.startswith("D")]
    elif n_extra:
        used = set(sp) | set(gn) | set(dn); pool = [x for x in OW.NAMES if x not in used]
        for i in rng.choice(len(pool), size=n_extra, replace=False):
            dn.append(pool[int(i)]); dv.append(int(rng.integers(1, OW.P))); order.insert(int(rng.integers(0, len(order) + 1)), "D%d" % (len(dn) - 1))
    new = OW._decorate(OW.NW._finish(it["d"], it["ops"], it["operands"], it["src_j"], it["values"], it["mul_s_t"], sp, gn, dn, it["unit"], it["frames"], dv, order, len(dn)),
                       it["order_mode"], None, it.get("dag", True))
    for k in ("wrap_final", "wrap_any"):
        if k in it:
            new[k] = it[k]
    for k in ("d", "ops", "operands", "src_j", "values", "answer", "mul_s_t", "frames", "unit", "givens"):
        assert new[k] == it[k], ("§241 re-render changed the program", k)
    assert new["names"]["spine"] == sp and new["names"]["givens"] == gn and set(new) == set(it), "§241 re-render keys / names"
    assert OW.solve(new["text"]) == it["answer"], "§241 re-render: the independent solver disagrees"
    return new


def ls_len(a, it):
    """token length of the pass-1 input (part_acc's L)."""
    return len(seq_std(AW.tok_aw(it, a.get("aux_target", "value"), a.get("cue", "none"), a.get("write_src", "all")), 1))


def lenstep_cells(a):
    """§241 cells (DF law = TEST2-DF's Stream law and frames, new seeds): A d 4 (LS_N), C d 6 (LS_N), B = A's programs + e extra distractors (e: the smallest
    |mean_B - mean_C| among e in 0..LS_EMAX passing the length gate, |mean| <= 5 and |p90| <= 10; rng SEED_LS_B fresh per e), D = the first LS_DPOOL programs of
    C's stream (C = the first LS_N) with 0 distractors, kept if length <= A's p90 (first LS_N kept; omitted when < LS_DMIN). -> cells {X: (items, prog_sigs)}, info."""
    assert law(a) == 2, "§241: DF runs only"
    kw = dict(frames=OW.FRAME_TRAIN, mul_s_p=2, **IO.OWKW)
    sA = OW.Stream(SEED_LS_A, **kw); A = [sA.draw(4) for _ in range(LS_N)]
    sC = OW.Stream(SEED_LS_C, **kw); Cpool = [sC.draw(6) for _ in range(LS_DPOOL)]; C = Cpool[:LS_N]
    LA = np.array([ls_len(a, it) for it in A]); LC = np.array([ls_len(a, it) for it in C]); mC, pC = float(LC.mean()), float(np.percentile(LC, 90))
    search = []
    for e in range(LS_EMAX + 1):
        rng = np.random.default_rng(SEED_LS_B); LB = np.array([ls_len(a, rerender_dis(rng, it, e)) for it in A])
        search.append({"e": e, "tok_mean": float(LB.mean()), "tok_p90": float(np.percentile(LB, 90)), "pass": bool(abs(LB.mean() - mC) <= 5 and abs(np.percentile(LB, 90) - pC) <= 10)})
    ok_e = [s_ for s_ in search if s_["pass"]]; assert ok_e, ("§241: no e passes the B / C length gate", search)
    e = min(ok_e, key=lambda s_: abs(s_["tok_mean"] - mC))["e"]; rng = np.random.default_rng(SEED_LS_B); B = [rerender_dis(rng, it, e) for it in A]
    pA = float(np.percentile(LA, 90)); Dsrc, D = [], []
    for it in Cpool:
        r_ = rerender_dis(None, it, drop=True)
        if ls_len(a, r_) <= pA:
            Dsrc.append(it["sig"]); D.append(r_)
            if len(D) == LS_N:
                break
    cells = {"A": (A, [it["sig"] for it in A]), "B": (B, [it["sig"] for it in A]), "C": (C, [it["sig"] for it in C])}
    if len(D) >= LS_DMIN:
        cells["D"] = (D, Dsrc)
    info = {"e": e, "search": search, "A_p90": pA, "C_mean": mC, "C_p90": pC, "D_qualified": len(D), "D_pool_scanned": len(Cpool) if len(D) < LS_N else None,
            "D_in_C": sum(s_ in set(cells["C"][1]) for s_ in Dsrc), "D_omitted": len(D) < LS_DMIN}
    return cells, info


@torch.no_grad()
def part_lenstep(m, g, sets, a, dev, bs):
    """§241 LEN-STEP (LS_v1 §2 + Note 241-1): per cell acc at K' 8 (= think_k), WS per pass 1..8 (AW.ws_readout), tok_mean / tok_p90 (pass-1 input length),
    items {prog_sig (the source A / C item's sig, shared across renders), ok (0/1 at K' 8), sig, L, pred, ws_pred}."""
    K = a["think_k"]; assert K == 8, ("§241: K' 8 = think_k", K)
    cells, info = lenstep_cells(a); out = {"cells": {}, "build": info}
    for c_, (its, psig) in cells.items():
        d = its[0]["d"]; T = [AW.tok_aw(it, a.get("aux_target", "value"), a.get("cue", "none"), a.get("write_src", "all")) for it in its]
        P, _ = passes(m, T, dev, bs, {K}); y = np.array([it["ans"] for it in T]); ok = (P[K] == y).astype(int); L = np.array([len(seq_std(it, 1)) for it in T])
        ws, pv = AW.ws_readout(m, {d: T}, dev, K, bs, per_item=True)
        out["cells"][c_] = {"d": d, "n": len(T), "tok_mean": float(L.mean()), "tok_p90": float(np.percentile(L, 90)), "acc": float(ok.mean()), "ws": ws[str(d)],
                            "n_distractors_mean": float(np.mean([it["n_distractors"] for it in its])),
                            "items": {"prog_sig": psig, "ok": ok.tolist(), "sig": [it["sig"] for it in its], "L": L.tolist(), "pred": P[K].tolist(), "ws_pred": pv[str(d)]}}
    out["def"] = ("LEN-STEP (§241 LS_v1 + Note 241-1): DF law, new seeds A 24100 (d 4) / C 24101 (d 6) / B re-render 24102; B = A's programs + e distractors (length-matched "
                  "to C); D = C-stream programs with 0 distractors, length <= A's p90 (omitted if < 300); acc at K' 8, WS per pass, fp32, one card, one eval_bs")
    return out


NC_ARMS = (                                                                 # §242 NORM-COUNTER (NC_v1 §2 + Note 242-1): (arm, d, edit) ; edit = (pass k, ref, heads)
    ("intact_d4", 4, None), ("intact_d6", 6, None),
    ("identity_d6", 6, (4, "one", "all")),                                  # 242-1 (1): scale exactly 1.0 through the same path (must equal intact_d6 bitwise)
    ("down_d6", 6, (4, 2, "all")), ("down_add_d6", 6, (4, 2, "add")),       # S_4 -> ||S_2|| (all heads) ; 242-1 (2): the additive heads only, to their own ||S_2||
    ("up_d4", 4, (2, 4, "all")), ("harm_d4", 4, (3, 1, "all")))             # S_2 -> ||S_4|| ; S_3 -> ||S_1||


def nc_norms(S, mem):
    """per item ||S|| over all heads, the additive heads [0, h_add) and the delta heads [h_add, h_add + h_del), fp32."""
    f = S.float(); return f.flatten(1).norm(dim=1), f[:, :mem.h_add].flatten(1).norm(dim=1), f[:, mem.h_add:mem.h_add + mem.h_del].flatten(1).norm(dim=1)


@torch.no_grad()
def nc_run(m, T, dev, bs, K, edit=None, ref=None):
    """passes 1..K as passes() / ws_readout (seq_std inputs, one padding, one lower cache per distinct input, S from zero, pass_aw); per pass t: the WS argmax at
    ':' and the intact norms (all / add / del); edit (k, r, heads): after pass k, S[heads] <- S[heads] * s with s = ref[heads][r] / ||S_k[heads]|| per item
    (r "one": s = 1.0 exactly). -> {ans (N,), ws {t: (N,)}, norms {all/add/del: {t: (N,)}}, scale (N,), post_relerr (max | ||S'_k[heads]|| / target - 1 |)}."""
    mem = m.mem; num = torch.tensor(NUM_IDS, device=dev); N = len(T); ans = np.full(N, -1); wsp = {t: np.full(N, -1) for t in range(1, K + 1)}
    nrm = {h: {t: np.zeros(N, dtype=np.float32) for t in range(1, K + 1)} for h in ("all", "add", "del")}; sc = np.full(N, np.nan, dtype=np.float32); rel = 0.0
    Td = n_distinct(T)
    for i in range(0, N, bs):
        ch = T[i:i + bs]; sl = slice(i, i + len(ch)); xs = []; m0 = None
        for t in range(1, Td + 1):
            ids_, mk = pad([seq_std(it, t) for it in ch]); assert m0 is None or (mk == m0).all(), "one padding across passes"; m0 = mk
            xs.append(torch.tensor(ids_, device=dev))
        msk = torch.tensor(m0, device=dev); hls = [m.lower(x) for x in xs]; S = mem.zero_state(len(ch), dev)
        assert S.shape[1] == mem.h_add + mem.h_del, ("§242: no den block expected", S.shape)
        for t in range(1, K + 1):
            j = min(t, Td) - 1; S, h, pos = m.pass_aw(S, xs[j], msk, hls[j])
            wsp[t][sl] = m.colon_logits(h, pos)[:, num].argmax(-1).cpu().numpy()
            na, nd_add, nd_del = nc_norms(S, mem)
            for k_, v_ in (("all", na), ("add", nd_add), ("del", nd_del)):
                nrm[k_][t][sl] = v_.cpu().numpy()
            if edit is not None and t == edit[0]:
                k, r, heads = edit; cur = {"all": na, "add": nd_add}[heads]
                s_ = torch.ones_like(cur) if r == "one" else torch.tensor(ref[heads][r][sl], device=dev) / cur
                sc[sl] = s_.cpu().numpy(); hs = slice(0, S.shape[1]) if heads == "all" else slice(0, mem.h_add)
                S = S.clone(); S[:, hs] = S[:, hs] * s_.to(S.dtype).view(-1, 1, 1, 1)
                if r != "one":
                    post = S[:, hs].float().flatten(1).norm(dim=1); rel = max(rel, float((post / torch.tensor(ref[heads][r][sl], device=dev) - 1).abs().max()))
        ans[sl] = pred_of(m.dec_logits(S, ch, dev))
    return {"ans": ans, "ws": wsp, "norms": nrm, "scale": sc, "post_relerr": rel}


@torch.no_grad()
def part_normctr(m, g, sets, a, dev, bs):
    """§242 NORM-COUNTER (NC_v1 + Note 242-1): TEST2 of the run's law, d 4 / d 6 (all 1,000), K' 8 (= think_k); arms NC_ARMS; refs = the same items' intact norms
    (same call, same batches). -> {arm: {d, n, acc, ws {t}, mean_scale, post_relerr, items {sig, ans, pred, ws_pred {t}, scale}}, head_norms {d: {t: {add, del, all}}},
    identity_equal_items}."""
    K = a["think_k"]; assert K == 8, ("§242: K' 8 = think_k", K)
    T = tokd(sets, a, [4, 6]); intact = {}; out = {"head_norms": {}}
    for arm, d, ed in NC_ARMS:
        r_ = nc_run(m, T[d], dev, bs, K, ed, intact[d]["norms"] if ed is not None else None)
        if ed is None:
            intact[d] = r_; out["head_norms"][str(d)] = {str(t): {h: float(r_["norms"][h][t].mean()) for h in ("add", "del", "all")} for t in range(1, K + 1)}
        y = np.array([it["ans"] for it in T[d]]); v = {t: np.array([int(it["raw"]["values"][min(t, d)]) for it in T[d]]) for t in range(1, K + 1)}
        out[arm] = {"d": d, "n": len(T[d]), "acc": float((r_["ans"] == y).mean()), "ws": {str(t): float((r_["ws"][t] == v[t]).mean()) for t in range(1, K + 1)},
                    "mean_scale": float(np.nanmean(r_["scale"])) if ed is not None else None, "edit": list(ed) if ed is not None else None,
                    "post_relerr": r_["post_relerr"] if ed is not None else None,
                    "items": {"sig": [it["sig"] for it in T[d]], "ans": y.tolist(), "pred": r_["ans"].tolist(), "ws_pred": {str(t): r_["ws"][t].tolist() for t in range(1, K + 1)},
                              "scale": [None if np.isnan(x) else float(x) for x in r_["scale"]]}}
    out["identity_equal_items"] = bool(out["identity_d6"]["items"]["pred"] == out["intact_d6"]["items"]["pred"]
                                       and out["identity_d6"]["items"]["ws_pred"] == out["intact_d6"]["items"]["ws_pred"])
    out["def"] = ("NORM-COUNTER (§242 NC_v1 + Note 242-1): TEST2 of the run's law d 4 / d 6, K' 8, fp32; rescale S <- S * n_ref / ||S|| (all heads; down_add_d6: the 6 additive "
                  "heads to their own ||S_2||) after pass k with n_ref the same item's intact norm; identity_d6 = scale 1.0 on the same path; ws = WS-READOUT per pass")
    return out
