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


def run_long(out, dev, bs, kplist, ds):
    """§239: load <out>/model.pt as load() does, build the run law's sets (set shas asserted vs args.json), part_long, write ONLY <out>/eval_long.json."""
    t1 = time.time(); a, m, g = load(out, dev); assert m is not None, "LONG-RUN: inject runs only"; sets = IO.build_sets(mul_s_p=law(a))
    for k_, v_ in sets["shas"].items():
        assert a.get(k_) in (None, v_), ("§239: set sha mismatch vs args.json", k_)
    if str(dev).startswith("cuda"):
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(dev)
    r_ = part_long(m, g, sets, a, dev, bs, kplist, ds)
    r_.update({"eval_dtype": "fp32", "eval_gpu": torch.cuda.get_device_name(dev) if str(dev).startswith("cuda") else "cpu", "eval_bs": bs, "kprime_list": kplist, "ds": ds,
               "law": law(a), "set_law": sets["shas"].get("set_law", "default"), "think_k": a["think_k"], "module_sha256": IO.sha_file(os.path.abspath(__file__)),
               "peak_gib": torch.cuda.max_memory_allocated(dev) / 2 ** 30 if str(dev).startswith("cuda") else None, "sec": round(time.time() - t1)})
    json.dump(r_, open(os.path.join(out, "eval_long.json"), "w", newline="\n")); print("LONG done", out, r_["sec"], "s", flush=True)
