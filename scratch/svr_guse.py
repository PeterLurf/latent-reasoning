"""SVR seed-stability draft rev 3, G-use (docs/THEORY_SEEDSTAB2_DRAFT_2026-10-14.md §4.3 + the G-use rule, 17c06299; Theory contract 2026-10-14; $0,
eval-only, final model.pt): is D7's natural separator USED by the 9 converged nested models? V_probe throughout (these models have no SVR head).
V_probe: a step_recover-protocol probe (nested170 dec_reads at the decoder's " =" query, fit 4,000 train_items rng 17021 / test 1,000 test items n <= 8
  rng 17022, the run's exclusion set) on the DELTA heads' reads only (post-gain, 384-d), POOLED over t = 2..min(8, K) (labels v_t = reductions[t-1]),
  standardised (mu, sigma), L2 1e-3, LBFGS 200. V = span{diag(1/sigma)(w_c - w_bar) : c in Z_11} (the dual vectors in read space); orthonormal basis Q
  by SVD (rank k = #singular values > 1e-6 max). Gate G-abl: the probe's pooled test accuracy on P_V-ablated reads = chance (1/11) +- 0.02.
(a) SUBSPACE ABLATION. r <- r - Q Q^T r on the concatenated post-gain delta read (= before the scalar gain e^{s_del}; s_del is a scalar) at EVERY
  read_fixed call (the think reads of every pass at all tokens and the answer decoder's read at all prompt tokens) - a process-local patched copy of
  deltamem.read_fixed (no module edit). Gate IDENT: the patched path with an EMPTY projector reproduces the original read_fixed's predictions on all
  items, bitwise. Outcome: answer accuracy at K' = 8 on 2,400 test items (n 3..8, 400 per n: for n = 3..8 in order, rng 18202 permutation of that n's
  test items [:400], one generator). Controls: (i) 5 Haar-random rank-k subspaces of the 384-d read space (rng 19950); (ii) 5 energy-matched ones:
  200 candidates (rng 19951), candidate j = span of k Gaussian vectors with covariance Sigma^beta, beta = (0.5, 1, 2)[j % 3]; keep the 5 with
  E(R) = tr(P_R Sigma) closest to E(V). Sigma = the pooled second moment of the post-gain delta read over all read sites (think passes 1..8 at the
  unpadded think tokens + the answer read at the unpadded prompt tokens) on the first 512 fit items. Paired bootstrap (2,000, rng 19970) over items.
  Delta_E = drop(V) - mean drop(R_E), Delta_rank = drop(V) - mean drop(R_rank). ABL-USED: Delta_E >= 0.10 and Delta_rank >= 0.10 and drop(V) above
  all 10 random drops and Delta_E's 95% lower bound > 0.05; ABL-NULL: Delta_E <= 0.03 and its upper bound <= 0.06; else ABL-UNCLEAR.
(b) DONOR SWAP at pass 2 (K = the run's think_k, as the registered answer path): after pass 2, Delta_h <- Delta_h + delta_h qhat_h^T with
  delta = P_V(r_d - r), r = Delta_2 qhat_eq(x) the item's own pre-gain delta read at its " =" query, r_d = the donor's at its own query; passes 3..K
  unchanged. Items: the 2,400 above; donors from the same n's 1,000 test items (rng 19960, one permutation per item): first a different-value donor
  (v_2^d != v_2) with the same (operator, original-tree height) at reduction 2, else any different-value donor of that n; the same-value control donor
  likewise with v_2^d == v_2 (another item). answer' = x's tree re-evaluated with reduction 2's node set to v_2^d; items kept iff answer' != answer (answer' undefined by a
  later division by 0 -> dropped, counted).
  F = P(answer = answer') under the different-value swap; F_null = the same rate under the same-value swap; disruption = own accuracy (no swap) - own
  accuracy under the same-value swap. FOLLOWS: F - F_null >= 0.15 and its 95% lower bound > 0.05; IGNORES: F - F_null <= 0.05; SWAP-DISRUPTIVE
  (void): disruption > 0.05. Use label: USED = ABL-USED or FOLLOWS; NOT-USED = ABL-NULL and IGNORES; else USE-UNCLEAR.
-> <run>/eval_svr_guse.json
    python scratch/svr_guse.py <run> [device cuda:1] [--bs 250]"""
import copy
import json
import os
import sys
import time

ROOT = "D:/Latent"; sys.path.insert(0, ROOT); os.chdir(ROOT)
os.environ.setdefault("BOXES_NB", "5")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

import boxes.deltamem as DM  # noqa: E402
import boxes.halt173 as H  # noqa: E402
import boxes.nested170 as N  # noqa: E402

DH = DM.DH; _ORIG_RF = DM.read_fixed; PROJ = {"on": False, "Q": None, "collect": None, "mask": None}


def read_fixed_p(mem, x, S, rec=None):
    """deltamem.read_fixed (READ_CLAMP / norms / ablate / rec all off on this path, asserted) with the delta read projected: r <- r - Q Q^T r."""
    if not PROJ["on"]:
        return _ORIG_RF(mem, x, S, rec)
    assert rec is None and DM.READ_CLAMP is None and mem.norms is None and mem.ablate is None
    H_ADD, H_DEL = mem.h_add, mem.h_del
    with torch.autocast("cuda", enabled=False):
        B, T, D = x.shape; a = mem.ln(x.float()); q = mem.qkv(a)[..., :D].view(B, T, H_ADD + H_DEL, DH)
        oa = torch.exp(mem.s_add) * DM.add_read(mem, Fn.elu(q[:, :, :H_ADD]) + 1, S[:, :H_ADD].float(), DM.state_z(mem, S))
        od = torch.exp(mem.s_del) * torch.einsum("bhvk,bthk->bthv", S[:, H_ADD:H_ADD + H_DEL].float(), Fn.normalize(q[:, :, H_ADD:], dim=-1))
        if PROJ["collect"] is not None:
            r = od.reshape(B, T, H_DEL * DH)[PROJ["mask"][:, :T]].double(); PROJ["collect"][0] += r.T @ r; PROJ["collect"][1] += r.shape[0]
        if PROJ["Q"] is not None:
            Q = PROJ["Q"]; r = od.reshape(B, T, H_DEL * DH); od = (r - (r @ Q) @ Q.T).view(B, T, H_DEL, DH)
        return x + (mem.fade * mem.wo(torch.cat([oa, od], 2).reshape(B, T, D))).to(x.dtype)


DM.read_fixed = read_fixed_p                                               # module-global lookup: gpt_forward_mem_r, deltas5.fwd_read, nested170 use DM.read_fixed


def dec_seqs(items):
    return [[N.EOT] + N.ids(it["syms"]) + [N.TID[" ="]] for it in items]


@torch.no_grad()
def answers(m, items, Kp, dev, ac, bs, swap=None):
    """nested170.inject_answers (think from the zero state, uncached, then dec_logits at ' =' over the 11 value tokens); swap(S, i0, chunk) after pass 2."""
    out = []
    for i in range(0, len(items), bs):
        ch = items[i:i + bs]; x, msk = m.think_inputs(ch, dev); sq = dec_seqs(ch); dm = torch.zeros(len(ch), max(len(s_) for s_ in sq), dtype=torch.bool, device=dev)
        for r_, s_ in enumerate(sq):
            dm[r_, :len(s_)] = True
        with ac():
            S = m.mem.zero_state(len(ch), dev)
            for t in range(1, Kp + 1):
                PROJ["mask"] = msk; S = m.write(S, x, msk)
                if swap is not None and t == 2:
                    S = swap(S, i, ch)
            PROJ["mask"] = dm; lg = m.dec_logits(S, ch, dev)
        out.append(lg[:, N.NUM_IDS].argmax(-1).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def qhat_eq(m, items, dev, ac):
    """the delta heads' unit queries at the decoder's ' =' position (blocks 0..layer-1 on [EOT, expr, ' =']; memory-independent) -> (b, H_DEL, DH)."""
    g = m.g; sq = dec_seqs(items); ids_, _ = N.pad(sq); x = torch.tensor(ids_, device=dev); pos = torch.tensor([len(q_) - 1 for q_ in sq], device=dev)
    with ac():
        h = g.embed(x) + DM.wpe_slice(g, x.shape[1])[None]
        for j in range(m.mem.layer):
            h = DM._block(g, j, h, h.shape[-1] // g.H)
    h6 = h[torch.arange(len(items), device=dev), pos].float(); D = h6.shape[-1]; HA, HD_ = m.mem.h_add, m.mem.h_del
    with torch.autocast("cuda", enabled=False):
        q = m.mem.qkv(m.mem.ln(h6))[..., :D].view(len(items), HA + HD_, DH)
    return Fn.normalize(q[:, HA:], dim=-1)


@torch.no_grad()
def state_after(m, items, t_, dev, ac, bs):
    out = []
    for i in range(0, len(items), bs):
        ch = items[i:i + bs]; x, msk = m.think_inputs(ch, dev)
        with ac():
            S = m.mem.zero_state(len(ch), dev)
            for _ in range(t_):
                PROJ["mask"] = msk; S = m.write(S, x, msk)
        out.append(S.float())
    return torch.cat(out)


def red2(root):
    """reduction 2 of nested170.reductions' order: (operator, original-tree height, value)."""
    nd = copy.deepcopy(root); hgt = {}

    def seth(x):
        hgt[id(x)] = 0 if x.op is None else 1 + max(seth(x.l), seth(x.r)); return hgt[id(x)]
    seth(nd); k = 0
    while nd.op is not None:
        path = N._leftmost_ready(nd); tgt = nd
        for p_ in path:
            tgt = getattr(tgt, p_)
        k += 1; op_, h_ = tgt.op, hgt[id(tgt)]; tgt.v = N.apply(tgt.op, tgt.l.v, tgt.r.v); tgt.op = tgt.l = tgt.r = None
        if k == 2:
            return op_, h_, tgt.v
    return None


def answer_cf(root, v2):
    """the root value with reduction 2's node set to v2 (same reduction order; later reductions recomputed); None when a later division by 0 makes the
    counterfactual undefined in Z_11 (such items are dropped, counted as n_undefined)."""
    nd = copy.deepcopy(root); k = 0
    while nd.op is not None:
        path = N._leftmost_ready(nd); tgt = nd
        for p_ in path:
            tgt = getattr(tgt, p_)
        k += 1
        if k != 2 and tgt.op == "/" and tgt.r.v == 0:
            return None
        tgt.v = v2 if k == 2 else N.apply(tgt.op, tgt.l.v, tgt.r.v); tgt.op = tgt.l = tgt.r = None
    return nd.v


def probe_fit(Xf, yf, dev, wd=1e-3, iters=200):
    mu, sd = Xf.mean(0), Xf.std(0) + 1e-6; Xf_ = torch.tensor((Xf - mu) / sd, device=dev, dtype=torch.float32); yf_ = torch.tensor(yf, device=dev, dtype=torch.long)
    with torch.enable_grad(), torch.autocast("cuda", enabled=False):
        W = torch.zeros(Xf.shape[1], N.P, device=dev, requires_grad=True); b = torch.zeros(N.P, device=dev, requires_grad=True)
        opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=iters, line_search_fn="strong_wolfe")

        def clos():
            opt.zero_grad(); l_ = Fn.cross_entropy(Xf_ @ W + b, yf_) + wd * (W ** 2).sum(); l_.backward(); return l_
        opt.step(clos)
    return mu, sd, W.detach().cpu().numpy().astype(np.float64), b.detach().cpu().numpy().astype(np.float64)


def probe_acc(X, y, mu, sd, W, b):
    return float(((((X - mu) / sd) @ W + b).argmax(1) == y).mean())


def boot(d, rng, n=2000):
    d = np.asarray(d, dtype=np.float64); idx = rng.integers(0, len(d), (n, len(d))); bs_ = d[idx].mean(1); return float(np.percentile(bs_, 2.5)), float(np.percentile(bs_, 97.5))


def orth(Mx):
    U, s, _ = np.linalg.svd(Mx, full_matrices=False); k = int((s > 1e-6 * s.max()).sum()); return U[:, :k]


def main():
    run = sys.argv[1]; dev = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else "cuda:1"; rd = os.path.join(ROOT, "boxes", "runs_fsq", run)
    kind, m, a = H.load_model(rd, dev); assert kind == "nested", kind
    bs = int(sys.argv[sys.argv.index("--bs") + 1]) if "--bs" in sys.argv else int(a.get("eval_bs") or 250)
    DM.FAST_ADD = bool(a.get("fast_add")); DM.CHUNK_DELTA = 0; K = int(a["think_k"]); KA = 8; t0 = time.time(); HA, HDl = int(m.mem.h_add), int(m.mem.h_del)
    assert m.mem.s_del.dim() == 0, "the projection before / after the gain is equivalent only for a scalar gain"
    amp = None if (not dev.startswith("cuda") or torch.cuda.get_device_capability(dev)[0] < 8) else torch.bfloat16
    ac = lambda: torch.autocast("cuda", dtype=amp, enabled=amp is not None)  # noqa: E731
    TE = N.test_set(int(a.get("per_n") or 1000)); EXCL = {tuple(it["syms"]) for it in TE if it["n"] >= int(a.get("excl_min_n", 1))}
    r_sp = np.random.default_rng(17300)
    for _ in range(int(a.get("serial_pool") or 2200)):
        r__, _t = N.generate(r_sp, 8); EXCL.add(tuple(N.render(r__)))
    tr = N.train_items(np.random.default_rng(17021), 4000, 8, EXCL); pool = [it for it in TE if it["n"] <= 8]
    te = [pool[i] for i in np.random.default_rng(17022).permutation(len(pool))[:1000]]
    out = {"run": run, "K": K, "K_ablation": KA, "bs": bs, "card": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu", "dtype": str(amp)}
    # ---- V_probe (pooled t = 2..min(8, K), delta slice of the post-gain reads)
    Rf, Rs = N.dec_reads(m, tr, K, dev, ac, bs), N.dec_reads(m, te, K, dev, ac, bs); sl = slice(HA * DH, (HA + HDl) * DH)
    vf = [N.reductions(it["root"]) for it in tr]; vs = [N.reductions(it["root"]) for it in te]; Xf, yf, Xs, ys = [], [], [], []
    for t in range(2, min(8, K) + 1):
        fi = [i for i, v in enumerate(vf) if len(v) >= t]; si = [i for i, v in enumerate(vs) if len(v) >= t]
        Xf.append(Rf[t - 1, fi][:, sl]); yf += [vf[i][t - 1] for i in fi]; Xs.append(Rs[t - 1, si][:, sl]); ys += [vs[i][t - 1] for i in si]
    Xf, Xs = np.concatenate(Xf).astype(np.float64), np.concatenate(Xs).astype(np.float64); yf, ys = np.array(yf), np.array(ys)
    mu, sd, W, b = probe_fit(Xf, yf, dev); Dual = (W - W.mean(1, keepdims=True)) / sd[:, None]; Qv = orth(Dual); k = Qv.shape[1]
    Xs_abl = Xs - (Xs @ Qv) @ Qv.T; acc_p, acc_abl = probe_acc(Xs, ys, mu, sd, W, b), probe_acc(Xs_abl, ys, mu, sd, W, b)
    maj = float((ys == np.bincount(yf, minlength=N.P).argmax()).mean())
    out["v_probe"] = {"rank": k, "acc_test_pooled": acc_p, "acc_test_ablated": acc_abl, "chance": 1 / N.P, "majority": maj, "n_fit": len(yf), "n_test": len(ys),
                      "gate_G_abl": bool(abs(acc_abl - 1 / N.P) <= 0.02)}
    print(run, "V_probe", out["v_probe"], flush=True)
    # ---- Sigma over all read sites (passes 1..8 + the answer read) on 512 fit items
    PROJ.update(on=True, Q=None, collect=[torch.zeros(HDl * DH, HDl * DH, dtype=torch.float64, device=dev), 0])
    answers(m, tr[:512], KA, dev, ac, bs); Sig = (PROJ["collect"][0] / PROJ["collect"][1]).cpu().numpy(); n_sites = PROJ["collect"][1]; PROJ.update(collect=None)
    ev, U = np.linalg.eigh(Sig); ev = np.clip(ev, 0, None); E = lambda Q_: float(np.trace(Q_.T @ Sig @ Q_))  # noqa: E731
    rng_h = np.random.default_rng(19950); R_rank = [orth(rng_h.standard_normal((HDl * DH, k))) for _ in range(5)]
    rng_e = np.random.default_rng(19951); cands = []
    for j in range(200):
        be_ = (0.5, 1.0, 2.0)[j % 3]; Q_ = orth((U * ev ** (be_ / 2)) @ U.T @ rng_e.standard_normal((HDl * DH, k))); cands.append((abs(E(Q_) - E(Qv)), j, be_, Q_))
    R_en = sorted(cands, key=lambda c_: c_[0])[:5]
    out["sigma"] = {"n_sites": int(n_sites), "E_V": E(Qv), "E_total": float(np.trace(Sig)), "E_rank": [E(Q_) for Q_ in R_rank],
                    "E_energy": [E(c_[3]) for c_ in R_en], "energy_cands": [{"j": c_[1], "beta": c_[2]} for c_ in R_en]}
    # ---- (a) ablation at K' = 8 on 2,400 test items
    r18 = np.random.default_rng(18202); A_it = []
    for n_ in range(3, 9):
        pn = [it for it in TE if it["n"] == n_]; A_it += [pn[i] for i in r18.permutation(len(pn))[:400]]
    gold = np.array([it["t"] for it in A_it]); PROJ.update(on=False, Q=None); base = answers(m, A_it, KA, dev, ac, bs)
    PROJ.update(on=True, Q=torch.zeros(HDl * DH, 0, device=dev)); ident = answers(m, A_it, KA, dev, ac, bs)
    out["gate_IDENT"] = {"equal_items": int((base == ident).sum()), "n": len(A_it), "pass": bool((base == ident).all())}; print(run, "IDENT", out["gate_IDENT"], flush=True)
    c0 = (base == gold).astype(float); drops = {}

    def abl(Q_):
        PROJ.update(on=True, Q=torch.tensor(Q_, device=dev, dtype=torch.float32)); p_ = answers(m, A_it, KA, dev, ac, bs); PROJ.update(on=False, Q=None)
        return c0 - (p_ == gold).astype(float)
    dV = abl(Qv); dR = [abl(Q_) for Q_ in R_rank]; dE = [abl(c_[3]) for c_ in R_en]
    rb = np.random.default_rng(19970); dEl = dV - np.mean(dE, 0); dRl = dV - np.mean(dR, 0)
    drops = {"base_acc": float(c0.mean()), "drop_V": float(dV.mean()), "drop_rank": [float(d_.mean()) for d_ in dR], "drop_energy": [float(d_.mean()) for d_ in dE],
             "Delta_E": float(dEl.mean()), "Delta_E_ci95": boot(dEl, rb), "Delta_rank": float(dRl.mean()), "Delta_rank_ci95": boot(dRl, rb),
             "drop_V_above_all_10": bool(dV.mean() > max([d_.mean() for d_ in dR + dE])),
             "base_acc_by_n": {str(n_): float(c0[i * 400:(i + 1) * 400].mean()) for i, n_ in enumerate(range(3, 9))},
             "drop_V_by_n": {str(n_): float(dV[i * 400:(i + 1) * 400].mean()) for i, n_ in enumerate(range(3, 9))}}
    lo, hi = drops["Delta_E_ci95"]
    drops["reading"] = ("ABL-USED" if (drops["Delta_E"] >= 0.10 and drops["Delta_rank"] >= 0.10 and drops["drop_V_above_all_10"] and lo > 0.05)
                        else "ABL-NULL" if (drops["Delta_E"] <= 0.03 and hi <= 0.06) else "ABL-UNCLEAR")
    out["ablation"] = drops; print(run, "ABL", {k_: v_ for k_, v_ in drops.items() if not k_.endswith("_by_n")}, flush=True)
    # ---- (b) donor swap after pass 2 (K passes)
    PROJ.update(on=False, Q=None); r60 = np.random.default_rng(19960); info = [red2(it["root"]) for it in A_it]; byn = {}
    for it in TE:
        if 3 <= it["n"] <= 8:
            byn.setdefault(it["n"], []).append(it)
    r2 = {id(it): red2(it["root"]) for v_ in byn.values() for it in v_}; S_it, Dd, Ds, acf = [], [], [], []; n_undef = 0
    for it, (op_, h_, v2) in zip(A_it, info):
        cand = byn[it["n"]]; order = r60.permutation(len(cand)); pick = {}
        for want in ("diff", "same"):
            ok_ = lambda c_: (r2[id(c_)][2] != v2) if want == "diff" else (r2[id(c_)][2] == v2 and c_["syms"] != it["syms"])  # noqa: E731
            m1 = [cand[j] for j in order if ok_(cand[j]) and r2[id(cand[j])][:2] == (op_, h_)]
            m2 = m1 or [cand[j] for j in order if ok_(cand[j])]; pick[want] = m2[0] if m2 else None
        if pick["diff"] is None or pick["same"] is None:
            continue
        a2 = answer_cf(it["root"], r2[id(pick["diff"])][2]); n_undef += a2 is None
        if a2 is not None and a2 != it["t"]:
            S_it.append(it); Dd.append(pick["diff"]); Ds.append(pick["same"]); acf.append(a2)
    acf = np.array(acf); goldS = np.array([it["t"] for it in S_it]); Qt = torch.tensor(Qv, device=dev, dtype=torch.float32)
    qi = qhat_eq(m, S_it, dev, ac)

    def donor_reads(D_):
        S2 = state_after(m, D_, 2, dev, ac, bs); return torch.einsum("bhvk,bhk->bhv", S2[:, HA:HA + HDl], qhat_eq(m, D_, dev, ac)).reshape(len(D_), -1)
    rD, rS = donor_reads(Dd), donor_reads(Ds)

    def mk_swap(rdon):
        def sw(S, i0, ch):
            S = S.float().clone(); q_ = qi[i0:i0 + len(ch)]; Sd = S[:, HA:HA + HDl]; r = torch.einsum("bhvk,bhk->bhv", Sd, q_).reshape(len(ch), -1)
            dl = ((rdon[i0:i0 + len(ch)] - r) @ Qt) @ Qt.T; S[:, HA:HA + HDl] = Sd + dl.view(len(ch), HDl, DH)[..., :, None] * q_[..., None, :]; return S
        return sw
    p0 = answers(m, S_it, K, dev, ac, bs); pd_ = answers(m, S_it, K, dev, ac, bs, swap=mk_swap(rD)); ps_ = answers(m, S_it, K, dev, ac, bs, swap=mk_swap(rS))
    fl = (pd_ == acf).astype(float) - (ps_ == acf).astype(float); flo, fhi = boot(fl, rb)
    sw = {"n_items": len(S_it), "n_undefined_cf": int(n_undef), "F": float((pd_ == acf).mean()), "F_null": float((ps_ == acf).mean()), "F_minus_null": float(fl.mean()), "F_minus_null_ci95": (flo, fhi),
          "own_acc_noswap": float((p0 == goldS).mean()), "own_acc_sameswap": float((ps_ == goldS).mean()), "own_acc_diffswap": float((pd_ == goldS).mean()),
          "donor_match_op_height": float(np.mean([r2[id(d_)][:2] == red2(it["root"])[:2] for d_, it in zip(Dd, S_it)]))}
    sw["disruption"] = sw["own_acc_noswap"] - sw["own_acc_sameswap"]
    sw["reading"] = ("SWAP-DISRUPTIVE" if sw["disruption"] > 0.05 else "FOLLOWS" if (sw["F_minus_null"] >= 0.15 and flo > 0.05)
                     else "IGNORES" if sw["F_minus_null"] <= 0.05 else "SWAP-UNCLEAR")
    out["swap"] = sw; print(run, "SWAP", sw, flush=True)
    ra, rs_ = drops["reading"], sw["reading"]
    out["use_label"] = "USED" if (ra == "ABL-USED" or rs_ == "FOLLOWS") else "NOT-USED" if (ra == "ABL-NULL" and rs_ == "IGNORES") else "USE-UNCLEAR"
    out["sec"] = round(time.time() - t0); out["def"] = "SVR G-use (THEORY_SEEDSTAB2 draft rev 3 §4.3, V_probe): see scratch/svr_guse.py's docstring"
    json.dump(out, open(os.path.join(rd, "eval_svr_guse.json"), "w", newline="\n"), indent=1); print(run, "USE", out["use_label"], out["sec"], "s", flush=True)


if __name__ == "__main__":
    main()
