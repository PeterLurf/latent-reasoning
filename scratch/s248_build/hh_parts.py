

SEED_HH_FIT = 24750                                                         # §248 HALT-HEAD (HH_v1 §2): fit stream / d draws / split / LBFGS (unused, grep 10-29)
HH_N, HH_VAL, HH_L2, HH_TAU, HH_KMAX = 6000, 0.2, 1e-3, 0.90, 8


def hh_hidden(m, S, ch, dev):
    """the decoder pass as dec_logits (fwd_read: the read on S, final hidden after ln_f) -> (the answer-position hidden (B, 768) fp32, pred from g.logits(hidden) exactly as dec_logits)."""
    from boxes.deltas5 import fwd_read
    seqs = [it["dec"] for it in ch]; ids_, _ = pad(seqs); x = torch.tensor(ids_, device=dev)
    h = fwd_read(m.g, m.g.embed(x), m.mem, S); pos = torch.tensor([len(q) - 1 for q in seqs], device=dev); hid = h[torch.arange(len(seqs), device=dev), pos]
    return hid.float(), pred_of(m.g.logits(hid).float())


@torch.no_grad()
def hh_dump(m, T, dev, bs, K=HH_KMAX):
    """passes 1..K (seq_std inputs, one padding per pass, S from zero, pass_aw writes); the decoder's answer-position hidden + pred at every K' 0..K -> (H (N, K+1, 768), P (N, K+1))."""
    N = len(T); H = np.zeros((N, K + 1, 768), dtype=np.float32); P = np.zeros((N, K + 1), dtype=np.int64)
    for i in range(0, N, bs):
        ch = T[i:i + bs]; sl = slice(i, i + len(ch)); S = m.mem.zero_state(len(ch), dev)
        hd, pr = hh_hidden(m, S, ch, dev); H[sl, 0] = hd.cpu().numpy(); P[sl, 0] = pr
        for t in range(1, K + 1):
            ids_, mk = pad([seq_std(it, t) for it in ch]); x = torch.tensor(ids_, device=dev); msk = torch.tensor(mk, device=dev)
            S = m.pass_aw(S, x, msk, m.lower(x))[0]; hd, pr = hh_hidden(m, S, ch, dev); H[sl, t] = hd.cpu().numpy(); P[sl, t] = pr
    return H, P


def hh_auc(p, y):
    """ROC AUC by ranks (ties averaged)."""
    o = np.argsort(p, kind="mergesort"); r = np.empty(len(p)); ps = p[o]; i = 0
    while i < len(ps):
        j = i
        while j + 1 < len(ps) and ps[j + 1] == ps[i]:
            j += 1
        r[o[i:j + 1]] = (i + j) / 2.0 + 1; i = j + 1
    npos = int(y.sum()); nneg = len(y) - npos
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)) if npos and nneg else None


def hh_fit(X, y):
    """logistic regression, float64 CPU, deterministic: standardised features (train mean / std, std floored at 1e-6); loss = mean BCE + HH_L2 * ||w||^2 (bias not penalised);
    full-batch L-BFGS (strong Wolfe, 500 iterations max). -> (mu, sd, w, b)."""
    torch.manual_seed(SEED_HH_FIT); Xt = torch.tensor(X, dtype=torch.float64); yt = torch.tensor(y, dtype=torch.float64)
    mu = Xt.mean(0); sd = Xt.std(0).clamp_min(1e-6); Z = (Xt - mu) / sd
    w = torch.zeros(Z.shape[1], dtype=torch.float64, requires_grad=True); b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1.0, max_iter=500, tolerance_grad=1e-10, tolerance_change=1e-14, history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad(); lo = torch.nn.functional.binary_cross_entropy_with_logits(Z @ w + b, yt) + HH_L2 * (w * w).sum(); lo.backward(); return lo
    with torch.enable_grad():
        opt.step(closure)
    return mu.numpy(), sd.numpy(), w.detach().numpy(), float(b.detach())


def hh_prob(H, mu, sd, w, b):
    return 1.0 / (1.0 + np.exp(-(((H.astype(np.float64) - mu) / sd) @ w + b)))


@torch.no_grad()
def part_halthead(m, g, sets, a, dev, bs):
    """§248 HALT-HEAD (HH_v1 §2-4): 6,000 fresh DF-stream items (Stream(SEED_HH_FIT + 1, excl = the eval sets incl. TEST2-DF), d ~ U{1..4} (rng SEED_HH_FIT)), the decoder's
    answer-position hidden at K' 0..8, label y = [pred(K') == pred(8)] pooled over K'; a logistic head (hh_fit) on 80% of the fit items (item-level split, seed), val on 20%;
    TEST2-DF d 1..6: halt at the first K' with p >= 0.90, else 8 (censored at K' 8 with the K' 8 answer). Output in eval_halt's format + head {train_acc, val_acc, val_auc}."""
    rng = np.random.default_rng(SEED_HH_FIT); kw = dict(frames=OW.FRAME_TRAIN, mul_s_p=law(a), **IO.OWKW)
    s_ = OW.Stream(SEED_HH_FIT + 1, excl=sets["excl"], **kw); ds = rng.integers(1, 5, size=HH_N)
    F = [AW.tok_aw(s_.draw(int(d)), a.get("aux_target", "value"), a.get("cue", "none"), a.get("write_src", "all")) for d in ds]
    t2sig = {it["sig"] for d in sets["test2"] for it in sets["test2"][d]}; assert not any(it["raw"]["sig"] in t2sig for it in F), "§248: a fit item is in TEST2-DF"
    Hf = np.zeros((HH_N, HH_KMAX + 1, 768), dtype=np.float32); Pf = np.zeros((HH_N, HH_KMAX + 1), dtype=np.int64)
    for d in range(1, 5):                                                   # one d per batch (cue inputs share one padding)
        idx = np.where(ds == d)[0]; Hd, Pd = hh_dump(m, [F[i] for i in idx], dev, bs); Hf[idx] = Hd; Pf[idx] = Pd
    perm = rng.permutation(HH_N); nv = int(round(HH_VAL * HH_N)); va, tr = perm[:nv], perm[nv:]
    lab = (Pf == Pf[:, [HH_KMAX]]).astype(np.float64)
    Xtr, ytr = Hf[tr].reshape(-1, 768), lab[tr].reshape(-1); Xva, yva = Hf[va].reshape(-1, 768), lab[va].reshape(-1)
    mu, sd, w, b = hh_fit(Xtr, ytr); ptr, pva = hh_prob(Xtr, mu, sd, w, b), hh_prob(Xva, mu, sd, w, b)
    out = {"head": {"train_acc": float(((ptr >= 0.5) == (ytr == 1)).mean()), "val_acc": float(((pva >= 0.5) == (yva == 1)).mean()), "val_auc": hh_auc(pva, yva),
                    "train_rows": int(len(ytr)), "val_rows": int(len(yva)), "train_items": int(len(tr)), "val_items": int(len(va)), "y_rate_train": float(ytr.mean()),
                    "fit_d_counts": {str(d): int((ds == d).sum()) for d in range(1, 5)}, "w_norm": float(np.linalg.norm(w)), "b": b}}
    T = tokd(sets, a, range(1, 7))
    for d in range(1, 7):
        y = np.array([it["ans"] for it in T[d]]); Hd, Pd = hh_dump(m, T[d], dev, bs); p = hh_prob(Hd.reshape(-1, 768), mu, sd, w, b).reshape(len(y), HH_KMAX + 1)
        hit = p >= HH_TAU; kh = np.where(hit.any(1), hit.argmax(1), HH_KMAX); ah = Pd[np.arange(len(y)), kh]
        out[str(d)] = {"n": len(y), "acc_k8": float((Pd[:, HH_KMAX] == y).mean()), "acc_halt": float((ah == y).mean()), "mean_khalt": float(kh.mean()),
                       "khalt_hist": {str(k): int((kh == k).sum()) for k in range(HH_KMAX + 1)}, "never_halt_n": int((~hit.any(1)).sum()),
                       "items": {"sig": [it["sig"] for it in T[d]], "p": p.round(6).tolist(), "pred": Pd.tolist(), "khalt": kh.tolist(), "ans": y.tolist()}}
    out["def"] = ("§248 HALT-HEAD (HH_v1): fit 6,000 fresh DF-stream items (seed 24750, d U{1..4}, excl the eval sets), features = the decoder's answer-position hidden (post ln_f) "
                  "at K' 0..8, y = [pred(K') == pred(8)]; logistic head (standardised, mean BCE + 1e-3 ||w||^2, L-BFGS fp64), 80 / 20 item split; TEST2-DF halt at the first "
                  "K' with p >= 0.90 else 8")
    return out
