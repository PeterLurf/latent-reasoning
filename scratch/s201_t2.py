"""THEORY §201 OVERTHINK-EXIT T2 (121eb9ef; plan Part V T2; reader theory/score_s201.py): halting rules scored OFFLINE from T1's CLEAN dumps
(<dump>/<run>_{train,test}.npz; scratch/s201_t1.py). Every fit / threshold uses TRAINING items only (2,000, n 2..8, train_items rng 19900).
Pass index t = 1..32: the features of row t come from the write that produced S_t (no look-ahead pass); the answer a(t) = argmax of the answer logits.
PRIMARY SHIRYAEV-HEAD: a logistic pi_t for 1[t >= n] on the features: answer max-prob, entropy, top-2 margin; answer-change flags at t-1 and t-2;
AUX-DONE bit and prob; beta_mean and retention per delta head; their one-pass differences; the additive and delta read relative increments. No t and
no additive norm. Standardised on train; L2 1e-4; LBFGS (fp64). Threshold A = p C_FA / (p C_FA + c_d): p = 1 / mean(n) (the geometric completion
hazard per pass, MLE), C_FA = 1 - acc over training item-passes t < n - 1, c_d = the mean over training item-passes t >= n of P(wrong at t+1 | right
at t) + 0.002. Stop at the first t with pi_t >= A; never -> t = 32 with a(32) (censored).
Descriptive (same scorer file, other keys): AUX-DONE / AUX-DONE+1; MIN-BETA (stop at t when beta_{t+1} > beta_t + d_b), RET-JUMP (rho_{t+1} - rho_t
>= d_r), CONF (max-prob >= th_c), TRM-HEAD (a logistic on 1[a_t = y] with the same features, stop at q >= tau): each threshold chosen on training items
(the pooled n 2..8 accuracy at the halt, censored at 32) over a fixed grid; answer2 / MOVED2 (halt173.halts); per-n AUROC of pi (done vs not done).
-> <run>/eval_s201_halt.json {primary, best_fixed_k_train, acc_by_n_kprime {K' 0..32: {n: acc}}, descriptive, auroc_by_n, threshold, fit}
    python scratch/s201_t2.py <run> [--dump_dir D]"""
import json
import os
import sys

import numpy as np
import torch

ROOT = "D:/Latent"; TMAX = 32
DUMP = sys.argv[sys.argv.index("--dump_dir") + 1] if "--dump_dir" in sys.argv else "C:/Users/peter/AppData/Local/Temp/claude/D--/6af9c751-a423-5604-9170-0c177c13d67f/scratchpad/s201_dump"


def feats(D):
    """(N, 33, F) features (row 0 unused) and the answer / correctness arrays."""
    lg = torch.tensor(D["logits"], dtype=torch.float64); p = torch.softmax(lg, -1); top2 = p.topk(2, -1).values
    ans = lg.argmax(-1).numpy(); mp = top2[..., 0]; ent = -(p * torch.log(p.clamp_min(1e-30))).sum(-1); mar = top2[..., 0] - top2[..., 1]
    ch1 = np.zeros(ans.shape); ch2 = np.zeros(ans.shape); ch1[:, 1:] = ans[:, 1:] != ans[:, :-1]; ch2[:, 2:] = ans[:, 1:-1] != ans[:, :-2]
    be = np.nan_to_num(D["beta"].astype(np.float64)); rt = np.nan_to_num(D["ret"].astype(np.float64)); dbe = np.zeros_like(be); drt = np.zeros_like(rt)
    dbe[:, 2:] = be[:, 2:] - be[:, 1:-1]; drt[:, 2:] = rt[:, 2:] - rt[:, 1:-1]
    X = np.concatenate([mp.numpy()[..., None], ent.numpy()[..., None], mar.numpy()[..., None], ch1[..., None], ch2[..., None], D["aux_bit"].astype(np.float64)[..., None],
                        np.nan_to_num(D["aux_prob"].astype(np.float64))[..., None], be, rt, dbe, drt, np.nan_to_num(D["ri_add"].astype(np.float64))[..., None],
                        np.nan_to_num(D["ri_del"].astype(np.float64))[..., None]], -1)
    return X, ans, (ans == D["t"][:, None]), mp.numpy()


def logistic(X, y, l2=1e-4):
    mu, sd = X.mean(0), X.std(0) + 1e-9; Xt = torch.tensor((X - mu) / sd); yt = torch.tensor(y, dtype=torch.float64)
    w = torch.zeros(X.shape[1] + 1, dtype=torch.float64, requires_grad=True); opt = torch.optim.LBFGS([w], max_iter=500, line_search_fn="strong_wolfe")

    def cl():
        opt.zero_grad(); z = Xt @ w[:-1] + w[-1]; l_ = torch.nn.functional.binary_cross_entropy_with_logits(z, yt) + l2 * (w[:-1] ** 2).sum(); l_.backward(); return l_
    opt.step(cl)
    W = w.detach().numpy(); return lambda Z: 1 / (1 + np.exp(-(((Z - mu) / sd) @ W[:-1] + W[-1])))


def halt_first(fire):
    """fire (N, 33) bool over t (row 0 ignored) -> (t_halt (N,), censored (N,))."""
    f = fire.copy(); f[:, 0] = False; hit = f.any(1); return np.where(hit, f.argmax(1), TMAX), ~hit


def summarise(th, cens, cor, n):
    a = cor[np.arange(len(th)), th]; out = {"acc_by_n": {}, "event_halt_mean_by_n": {}, "censored_by_n": {}, "halt_t_hist_by_n": {}}
    for v in sorted(set(n.tolist())):
        s = n == v; out["acc_by_n"][str(v)] = float(a[s].mean()); ev = s & ~cens
        out["event_halt_mean_by_n"][str(v)] = float(th[ev].mean()) if ev.any() else None; out["censored_by_n"][str(v)] = float(cens[s].mean())
        u, c = np.unique(th[s], return_counts=True); out["halt_t_hist_by_n"][str(v)] = {str(int(a_)): int(b_) for a_, b_ in zip(u, c)}
    return out


def pooled_train(th, cor, n):
    a = cor[np.arange(len(th)), th]; return float(np.mean([a[n == v].mean() for v in range(2, 9) if (n == v).any()]))


def auroc(s, y):
    y = y.astype(bool)
    if y.all() or (~y).all():
        return None
    r = s.argsort().argsort() + 1.0; return float((r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


def main():
    run = sys.argv[1]; Dtr = np.load(os.path.join(DUMP, run + "_train.npz")); Dte = np.load(os.path.join(DUMP, run + "_test.npz"))
    Xtr, atr, ctr, mptr = feats(Dtr); Xte, ate, cte, mpte = feats(Dte); ntr, nte = Dtr["n"], Dte["n"]; T = np.arange(TMAX + 1)
    rows = (slice(None), slice(1, None)); ytr = (T[None, :] >= ntr[:, None])
    head = logistic(Xtr[rows].reshape(-1, Xtr.shape[-1]), ytr[rows].reshape(-1))
    pi_tr = np.zeros(ytr.shape); pi_tr[:, 1:] = head(Xtr[rows].reshape(-1, Xtr.shape[-1])).reshape(len(ntr), TMAX)
    pi_te = np.zeros((len(nte), TMAX + 1)); pi_te[:, 1:] = head(Xte[rows].reshape(-1, Xte.shape[-1])).reshape(len(nte), TMAX)
    p_hat = 1.0 / float(ntr.mean())
    pre = T[None, :] < (ntr[:, None] - 1); pre[:, 0] = False; C_FA = float(1 - ctr[pre].mean())
    post = (T[None, :-1] >= ntr[:, None]) & (T[None, :-1] >= 1) & ctr[:, :-1]; c_d = float((~ctr[:, 1:][post]).mean()) + 0.002
    A = p_hat * C_FA / (p_hat * C_FA + c_d)
    th, cens = halt_first(pi_te >= A); primary = summarise(th, cens, cte, nte)
    primary["mean_passes_censored32"] = float(th.mean())
    kp = {str(k): {str(v): float(cte[nte == v, k].mean()) for v in sorted(set(nte.tolist()))} for k in range(TMAX + 1)}
    trp = {k: float(np.mean([ctr[ntr == v, k].mean() for v in range(3, 9)])) for k in range(1, TMAX + 1)}; best_k = max(trp, key=lambda k: (trp[k], -k))
    desc = {}
    th_, ce_ = halt_first(Dte["aux_bit"].astype(bool)); desc["AUX-DONE"] = summarise(th_, ce_, cte, nte)
    desc["AUX-DONE+1"] = summarise(np.minimum(th_ + (~ce_), TMAX), ce_, cte, nte)
    bmt, bme = np.nan_to_num(Dtr["beta"].mean(2).astype(np.float64)), np.nan_to_num(Dte["beta"].mean(2).astype(np.float64))
    rmt, rme = np.nan_to_num(Dtr["ret"].mean(2).astype(np.float64)), np.nan_to_num(Dte["ret"].mean(2).astype(np.float64))

    def nextrule(V, d):                                                    # stop at t when V_{t+1} - V_t > d (one look-ahead pass)
        f = np.zeros(V.shape, bool); f[:, 1:-1] = (V[:, 2:] - V[:, 1:-1]) > d; return f
    GRID = {"MIN-BETA": (bmt, bme, np.linspace(0.0, 0.2, 41)), "RET-JUMP": (rmt, rme, np.linspace(0.0, 0.2, 41))}
    for nm, (Vt, Ve, grid) in GRID.items():
        sc = {float(d): pooled_train(halt_first(nextrule(Vt, d))[0], ctr, ntr) for d in grid}; d_ = max(sc, key=lambda d: (sc[d], -d))
        desc[nm] = {**summarise(*halt_first(nextrule(Ve, d_)), cte, nte), "threshold": d_, "train_pooled": sc[d_]}
    sc = {float(c): pooled_train(halt_first(mptr >= c)[0], ctr, ntr) for c in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999)}; c_ = max(sc, key=lambda c: (sc[c], c))
    desc["CONF"] = {**summarise(*halt_first(mpte >= c_), cte, nte), "threshold": c_, "train_pooled": sc[c_]}
    trm = logistic(Xtr[rows].reshape(-1, Xtr.shape[-1]), ctr[rows].reshape(-1)); q_tr = np.zeros(ctr.shape); q_te = np.zeros(cte.shape)
    q_tr[:, 1:] = trm(Xtr[rows].reshape(-1, Xtr.shape[-1])).reshape(len(ntr), TMAX); q_te[:, 1:] = trm(Xte[rows].reshape(-1, Xte.shape[-1])).reshape(len(nte), TMAX)
    sc = {float(u): pooled_train(halt_first(q_tr >= u)[0], ctr, ntr) for u in (0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99)}; u_ = max(sc, key=lambda u: (sc[u], u))
    desc["TRM-HEAD"] = {**summarise(*halt_first(q_te >= u_), cte, nte), "threshold": u_, "train_pooled": sc[u_]}
    same1 = np.zeros(ate.shape, bool); same1[:, 1:] = ate[:, 1:] == ate[:, :-1]; a2 = np.zeros(ate.shape, bool); a2[:, 2:] = same1[:, 2:] & same1[:, 1:-1]
    desc["answer2"] = summarise(*halt_first(a2), cte, nte); mv = ate != ate[:, :1]; desc["MOVED2"] = summarise(*halt_first(a2 & mv), cte, nte)
    au = {str(v): auroc(pi_te[nte == v, 1:].reshape(-1), (T[None, 1:] >= v).repeat((nte == v).sum(), 0).reshape(-1)) for v in sorted(set(nte.tolist()))}
    out = {"primary": primary, "best_fixed_k_train": int(best_k), "best_fixed_k_train_pooled": trp[best_k], "acc_by_n_kprime": kp,
           "threshold": {"A": A, "p_hat": p_hat, "C_FA": C_FA, "c_d": c_d}, "descriptive": desc, "auroc_by_n": au,
           "fit": {"n_train_items": int(len(ntr)), "train_n_hist": {str(v): int((ntr == v).sum()) for v in sorted(set(ntr.tolist()))}, "n_features": int(Xtr.shape[-1]),
                   "features": "maxprob, entropy, top2 margin, change t-1, change t-2, aux bit, aux prob, beta x6, retention x6, d beta x6, d retention x6, rel inc add read, rel inc delta read",
                   "row_t_def": "row t = the write that produced S_t; no look-ahead (MIN-BETA / RET-JUMP use one: t+1)"}}
    json.dump(out, open(os.path.join(ROOT, "boxes", "runs_fsq", run, "eval_s201_halt.json"), "w", newline="\n"), indent=1)
    print(run, "T2 primary acc", {k: round(v, 3) for k, v in primary["acc_by_n"].items()}, "A %.3f" % A, "best K'", best_k, flush=True)


if __name__ == "__main__":
    main()
