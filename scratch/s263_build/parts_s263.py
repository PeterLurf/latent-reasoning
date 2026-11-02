

SEED_HI_FIT = 25300                                                         # Note 259-5 C5-BUN: the halt-inc fit stream / d draws (unused elsewhere, grep 11-02)
HI_N, HI_KMAX, HI_TOL = 4000, 14, 0.02                                     # fit items; passes 1..14; acc_halt >= acc_k8 - 0.02 at every fit d 1-4
HI_GRID = np.concatenate([[0.0], np.logspace(-6, 0.5, 400)])                # candidate taus (0 = never halt)


@torch.no_grad()
def hh_dump_inc(m, T, dev, bs, K=HI_KMAX):
    """hh_dump to K' K (seq_std inputs, S from zero, pass_aw writes) + rel_inc[:, t-1] = ||S_t - S_{t-1}||_F / ||S_t||_F per item (the full state), t = 1..K ->
    (H (N, K+1, D), P (N, K+1), R (N, K))."""
    N = len(T); H = np.zeros((N, K + 1, m.g.D), dtype=np.float32); P = np.zeros((N, K + 1), dtype=np.int64); R = np.zeros((N, K), dtype=np.float64)
    for i in range(0, N, bs):
        ch = T[i:i + bs]; sl = slice(i, i + len(ch)); S = m.mem.zero_state(len(ch), dev)
        hd, pr = hh_hidden(m, S, ch, dev); H[sl, 0] = hd.cpu().numpy(); P[sl, 0] = pr
        for t in range(1, K + 1):
            ids_, mk = pad([seq_std(it, t) for it in ch]); x = torch.tensor(ids_, device=dev); msk = torch.tensor(mk, device=dev)
            S0 = S; S = m.pass_aw(S, x, msk, m.lower(x))[0]; hd, pr = hh_hidden(m, S, ch, dev); H[sl, t] = hd.cpu().numpy(); P[sl, t] = pr
            num = (S - S0).double().flatten(1).norm(dim=1); den = S.double().flatten(1).norm(dim=1).clamp_min(1e-30); R[sl, t - 1] = (num / den).cpu().numpy()
    return H, P, R


def hi_apply(R, P, tau, K=HI_KMAX):
    """halt at the first t in 1..K with rel_inc_t < tau: answer = P[:, t-1] (the decoder at K' = t-1), khalt = t-1, passes_run = t; never -> khalt K (answer P[:, K]),
    passes_run K. -> (answers, khalt, passes_run, never)."""
    hit = R < tau; first = np.where(hit.any(1), hit.argmax(1) + 1, 0); never = ~hit.any(1)
    kh = np.where(never, K, first - 1); pr_run = np.where(never, K, first); return P[np.arange(len(P)), kh], kh, pr_run, never


@torch.no_grad()
def part_halt_inc(m, g, sets, a, dev, bs):
    """Note 259-5 C5-BUN (score_c5b): the increment-norm halt. Fit: HI_N fresh REG-stream items (Stream(SEED_HI_FIT + 1, excl = the eval sets incl. TEST2), the run's law,
    d ~ U{1..4} (rng SEED_HI_FIT)); tau = the LARGEST grid value with acc_halt >= acc_k8 - 0.02 at every fit d 1-4 (scanned, no monotonicity assumed; 0 = never halt).
    Applied to TEST2 d 1..7. Per d {n, acc_k8, acc_halt, mean_khalt (t - 1), mean_passes_run (t), never_halt_n (khalt 14), khalt_hist} + items; "tau"; "fit"."""
    rng = np.random.default_rng(SEED_HI_FIT); kw = dict(frames=OW.FRAME_TRAIN, mul_s_p=law(a), **IO.OWKW)
    s_ = OW.Stream(SEED_HI_FIT + 1, excl=sets["excl"], **kw); ds = rng.integers(1, 5, size=HI_N)
    F = [AW.tok_aw(s_.draw(int(d)), a.get("aux_target", "value"), a.get("cue", "none"), a.get("write_src", "all")) for d in ds]
    t2sig = {it["sig"] for d in sets["test2"] for it in sets["test2"][d]}; assert not any(it["raw"]["sig"] in t2sig for it in F), "C5-BUN: a fit item is in TEST2"
    Pf = np.zeros((HI_N, HI_KMAX + 1), dtype=np.int64); Rf = np.zeros((HI_N, HI_KMAX)); yf = np.array([it["ans"] for it in F])
    for d in range(1, 5):                                                   # one d per batch (cue inputs share one padding)
        idx = np.where(ds == d)[0]; _, Pd, Rd = hh_dump_inc(m, [F[i] for i in idx], dev, bs); Pf[idx] = Pd; Rf[idx] = Rd
    ok_t = []
    for tau in HI_GRID:
        ah, _, _, _ = hi_apply(Rf, Pf, tau); ok_t.append(all(float((ah[ds == d] == yf[ds == d]).mean()) >= float((Pf[ds == d, 8] == yf[ds == d]).mean()) - HI_TOL for d in range(1, 5)))
    good = [float(t) for t, o in zip(HI_GRID, ok_t) if o]; tau = max(good) if good else 0.0
    ah, kh, prr, nv = hi_apply(Rf, Pf, tau)
    out = {"tau": tau, "fit": {"n": HI_N, "d_counts": {str(d): int((ds == d).sum()) for d in range(1, 5)}, "grid_n": len(HI_GRID), "n_feasible": len(good),
                              "fit_law": {0: "reg", 1: "default", 2: "df"}[law(a)],
                              "by_d": {str(d): {"acc_k8": float((Pf[ds == d, 8] == yf[ds == d]).mean()), "acc_halt": float((ah[ds == d] == yf[ds == d]).mean()),
                                                "mean_khalt": float(kh[ds == d].mean())} for d in range(1, 5)}}}
    T = tokd(sets, a, range(1, 8))
    for d in range(1, 8):
        y = np.array([it["ans"] for it in T[d]]); _, Pd, Rd = hh_dump_inc(m, T[d], dev, bs); ah, kh, prr, nv = hi_apply(Rd, Pd, tau)
        out[str(d)] = {"n": len(y), "acc_k8": float((Pd[:, 8] == y).mean()), "acc_halt": float((ah == y).mean()), "mean_khalt": float(kh.mean()),
                       "mean_passes_run": float(prr.mean()), "never_halt_n": int(nv.sum()), "khalt_hist": {str(k): int((kh == k).sum()) for k in range(HI_KMAX + 1)},
                       "items": {"sig": [it["sig"] for it in T[d]], "rel_inc": Rd.round(8).tolist(), "pred": Pd.tolist(), "khalt": kh.tolist(), "ans": y.tolist()}}
    out["def"] = ("Note 259-5 C5-BUN halt-inc: rel_inc_t = ||S_t - S_{t-1}||_F / ||S_t||_F (full state) after pass t = 1..14; halt at the first t with rel_inc_t < tau, "
                  "answer = the decoder at K' = t-1 (mean_khalt = t-1, mean_passes_run = t); never -> K' 14; tau = the largest of 401 grid values (0 + logspace(-6, 0.5, 400)) "
                  "with acc_halt >= acc_k8 - 0.02 at every d 1-4 on 4,000 fresh REG fit items (excl TEST2); TEST2 d 1..7; fp32")
    return out


@torch.no_grad()
def part_halthead2(m, g, sets, a, dev, bs):
    """Note 259-5 C5-BUN: the §248 HALT-HEAD procedure (part_halthead: SEED_HH_FIT fit stream, 6,000 items d U{1..4}, 80 / 20 item split, hh_fit / hh_prob, tau 0.90)
    with K' 0..14 and labels y = [pred(K') == pred(K' 14)]; halt at the first K' with p >= 0.90 else 14; TEST2 d 1..7. Layout as eval_halthead.json."""
    K2 = HI_KMAX; rng = np.random.default_rng(SEED_HH_FIT); kw = dict(frames=OW.FRAME_TRAIN, mul_s_p=law(a), **IO.OWKW)
    s_ = OW.Stream(SEED_HH_FIT + 1, excl=sets["excl"], **kw); ds = rng.integers(1, 5, size=HH_N)
    F = [AW.tok_aw(s_.draw(int(d)), a.get("aux_target", "value"), a.get("cue", "none"), a.get("write_src", "all")) for d in ds]
    t2sig = {it["sig"] for d in sets["test2"] for it in sets["test2"][d]}; assert not any(it["raw"]["sig"] in t2sig for it in F), "C5-BUN: a fit item is in TEST2"
    Hf = np.zeros((HH_N, K2 + 1, m.g.D), dtype=np.float32); Pf = np.zeros((HH_N, K2 + 1), dtype=np.int64)
    for d in range(1, 5):
        idx = np.where(ds == d)[0]; Hd, Pd, _ = hh_dump_inc(m, [F[i] for i in idx], dev, bs, K2); Hf[idx] = Hd; Pf[idx] = Pd
    perm = rng.permutation(HH_N); nv = int(round(HH_VAL * HH_N)); va, tr = perm[:nv], perm[nv:]
    lab = (Pf == Pf[:, [K2]]).astype(np.float64)
    Xtr, ytr = Hf[tr].reshape(-1, m.g.D), lab[tr].reshape(-1); Xva, yva = Hf[va].reshape(-1, m.g.D), lab[va].reshape(-1)
    mu, sd, w, b = hh_fit(Xtr, ytr); ptr, pva = hh_prob(Xtr, mu, sd, w, b), hh_prob(Xva, mu, sd, w, b)
    out = {"head": {"train_acc": float(((ptr >= 0.5) == (ytr == 1)).mean()), "val_acc": float(((pva >= 0.5) == (yva == 1)).mean()), "val_auc": hh_auc(pva, yva),
                    "train_rows": int(len(ytr)), "val_rows": int(len(yva)), "train_items": int(len(tr)), "val_items": int(len(va)), "y_rate_train": float(ytr.mean()),
                    "fit_d_counts": {str(d): int((ds == d).sum()) for d in range(1, 5)}, "fit_law": {0: "reg", 1: "default", 2: "df"}[law(a)], "kmax": K2,
                    "w_norm": float(np.linalg.norm(w)), "b": b}}
    T = tokd(sets, a, range(1, 8))
    for d in range(1, 8):
        y = np.array([it["ans"] for it in T[d]]); Hd, Pd, _ = hh_dump_inc(m, T[d], dev, bs, K2); p = hh_prob(Hd.reshape(-1, m.g.D), mu, sd, w, b).reshape(len(y), K2 + 1)
        hit = p >= HH_TAU; kh = np.where(hit.any(1), hit.argmax(1), K2); ah = Pd[np.arange(len(y)), kh]
        out[str(d)] = {"n": len(y), "acc_k8": float((Pd[:, 8] == y).mean()), "acc_k14": float((Pd[:, K2] == y).mean()), "acc_halt": float((ah == y).mean()),
                       "mean_khalt": float(kh.mean()), "khalt_hist": {str(k): int((kh == k).sum()) for k in range(K2 + 1)}, "never_halt_n": int((~hit.any(1)).sum()),
                       "items": {"sig": [it["sig"] for it in T[d]], "p": p.round(6).tolist(), "pred": Pd.tolist(), "khalt": kh.tolist(), "ans": y.tolist()}}
    out["def"] = ("Note 259-5 C5-BUN halthead2: §248's HALT-HEAD with K' 0..14, y = [pred(K') == pred(K' 14)], halt at the first K' with p >= 0.90 else 14; fit 6,000 fresh "
                  "REG items (seed 24750, d U{1..4}, excl the eval sets), 80 / 20 item split; TEST2 d 1..7; fp32")
    return out
