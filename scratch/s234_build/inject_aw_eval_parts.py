SWAP_CAP = 1000; SWAP_TRIES = 3000; SEED_SWAP_REC, SEED_SWAP_DON = 23244, 23245   # aw_checks PAIRS (1,000 kept of 1,613; 0 overlap train / TEST2)
SEED_PROBE = 23471; PROBE_N = 3000; P = 11                                  # aw_gate's G-AW0 item stream (d 2, EXCL), 60 / 20 / 20 by sig (rng 0)


def swap_pairs(a):
    """aw_checks.chk_pairs's law: recipients d 2 from Stream(23244), donors OW.redraw(rng 23245) (same skeleton: names / unit / ops); kept iff
    |{own, donor final, cf_a(1)}| = 3, cf_b(1) not in {own, donor final}, equal text token length; the first 1,000. Cues equal (asserted)."""
    r2 = np.random.default_rng(SEED_SWAP_DON); s2 = OW.Stream(SEED_SWAP_REC, frames=OW.FRAME_TRAIN, **IO.OWKW); kept = []; tried = 0
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
    s = OW.Stream(SEED_PROBE, sets["excl"], frames=OW.FRAME_TRAIN, **IO.OWKW); raw = [s.draw(2) for _ in range(PROBE_N)]
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


def swap_t(d):
    """Note 234-5 (b) + amendment: the swap step t = ceil(d / 2), except d 3 -> t 1 (at t 2 step 3 is a state reference, so cf_a(2) = donor final: 0 kept)."""
    return 1 if d == 3 else math.ceil(d / 2)


def swap_pairs_d(a, d):
    """Note 234-5 (b) + amendment: the PAIRS law at depth d with t = swap_t(d) (ceil(d / 2); d 3 -> 1): recipients d from Stream(23260 + d), donors OW.redraw(rng 23270 + d); kept iff
    |{own, donor final, cf_a(t)}| = 3, cf_b(t) not in {own, donor final}, equal text token length; the first 1,000 (3,000 tries). Cues equal (asserted)."""
    t = swap_t(d); r2 = np.random.default_rng(SEED_SWAPD_DON + d); s2 = OW.Stream(SEED_SWAPD_REC + d, frames=OW.FRAME_TRAIN, **IO.OWKW); kept = []; tried = 0
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
    out["def"] = ("Note 234-5 (b) + amendment, descriptive, d >= 3: pairs = the PAIRS law at d with t = ceil(d/2), d 3 -> t 1 (recipients Stream(23260 + d), donors redraw rng 23270 + d, cap 1,000); "
                  "SWAP-K: S_K := donor's S_K, U = {donor final, cf_a(t)}, floor = MEAN-K (the donors' mean S_K); SWAP-MID: S after pass t := donor's S_t, the recipient's own "
                  "later passes, U = {cf_a(t), cf_b(t)}, floor = MEAN at t (the donors' mean S_t); answers at K' = d and think_k")
    return out


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


