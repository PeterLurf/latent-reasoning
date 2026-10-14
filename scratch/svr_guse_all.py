"""SVR G-use ALL-HEADS add-on (Theory 2026-10-14, DESCRIPTIVE, post hoc: added after s185_rms_s1's delta-only ABL-NULL; the registered rule and the
registered delta-only run (scratch/svr_guse.py) are unchanged). Why: the ADD slice holds the same values (ADD step_recover t2 0.66-0.81 in the converged
models), so a delta-only ablation / swap tests the delta copy's necessity while an additive copy can keep the answer right (U3).
(a) ALL ablation: V_probe_all = svr_guse's V_probe construction on the 768-d post-gain read [g_add read_add || g_del read_del] (pooled t = 2..min(8, K),
  dual vectors diag(1/sigma)(w_c - w_bar), SVD rank k); r_all <- r_all - Q Q^T r_all on the concatenated post-gain read at EVERY read_fixed call (both
  slices; think passes and the answer read). Gate IDENT (empty projector == original, bitwise) and G-abl as svr_guse. Controls: 5 Haar rank-k
  (rng 19950) and 5 energy-matched of 200 candidates (rng 19951, beta cycling 0.5 / 1 / 2) in 768-d, Sigma_all over the same read sites on the first
  512 fit items. Same 2,400 items, K' 8, bootstrap (rng 19970), the same reading thresholds (labelled descriptive).
(b) ALL swap (--swap): models WITHOUT add_norm: the additive read is e^{s_add} phi(q)^T A, so an exact edit exists (labelled "exact"): after pass 2, delta = P_Vall(r_d - r) on the 768-d post-gain read at q_eq (item / donor at their own queries), split into the ADD / DELTA
  slices. DELTA: Delta_h += (delta_del_h / e^{s_del}) qhat_h^T (svr_guse's rank-1 delta-rule edit). ADD: A_h += phi(q_h) delta_add_h^T / (e^{s_add}
  ||phi(q_h)||^2) - one extra additive write keyed at phi(q_eq), the minimal-Frobenius-norm A change that moves the read at q_eq by exactly delta_add_h.
  RMS models (Theory: R1, labelled "approx (direction-only, RMS)"): the read is e^{s_add} g_h u / rms(u), u = phi(q)^T A, so the ADD target
  t = r_add + delta_add is hit in direction only: u' = sign(e^{s_add} g_h) t rms(u) / rms(t), A_h += phi(q_h) (u' - u)^T / ||phi(q_h)||^2.
  Fidelity (median): cos(P_V(read' - r), P_V(r_d - r)) and ||P_V(read' - r)|| / ||P_V(r_d - r)||; max |read' - (r + delta)| (0 for exact).
  EDIT-SUCCESS (all 9): V_probe_all on the edited pass-2 read at q_eq predicts the donor's v_2; edit_success_rate, and F / F_null / F - F_null
  unconditional AND on edited items. edit_success < 0.5 -> INCONCLUSIVE. Same items / donors / readings as svr_guse (b).
-> "all_heads" in <run>/eval_svr_guse.json (the registered fields untouched)
    python scratch/svr_guse_all.py <run> [device cuda:0] [--bs 250] [--swap]"""
import json
import os
import sys
import time

sys.path.insert(0, "D:/Latent/scratch"); sys.path.insert(0, "D:/Latent")
import svr_guse as SG  # noqa: E402  (its module top level patches DM.read_fixed with the delta-only copy; replaced below)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

DM, H, N, DH, PROJ = SG.DM, SG.H, SG.N, SG.DH, SG.PROJ


def read_fixed_all(mem, x, S, rec=None):
    """deltamem.read_fixed with the 768-d post-gain read projected: r_all <- r_all - Q Q^T r_all (PROJ['Q'] now 768 x k)."""
    if not PROJ["on"]:
        return SG._ORIG_RF(mem, x, S, rec)
    assert rec is None and DM.READ_CLAMP is None and mem.norms is None and mem.ablate is None
    H_ADD, H_DEL = mem.h_add, mem.h_del
    with torch.autocast("cuda", enabled=False):
        B, T, D = x.shape; a = mem.ln(x.float()); q = mem.qkv(a)[..., :D].view(B, T, H_ADD + H_DEL, DH)
        oa = torch.exp(mem.s_add) * DM.add_read(mem, Fn.elu(q[:, :, :H_ADD]) + 1, S[:, :H_ADD].float(), DM.state_z(mem, S))
        od = torch.exp(mem.s_del) * torch.einsum("bhvk,bthk->bthv", S[:, H_ADD:H_ADD + H_DEL].float(), Fn.normalize(q[:, :, H_ADD:], dim=-1))
        o = torch.cat([oa, od], 2).reshape(B, T, D)
        if PROJ["collect"] is not None:
            r = o[PROJ["mask"][:, :T]].double(); PROJ["collect"][0] += r.T @ r; PROJ["collect"][1] += r.shape[0]
        if PROJ["Q"] is not None:
            Q = PROJ["Q"]; o = o - (o @ Q) @ Q.T
        return x + (mem.fade * mem.wo(o)).to(x.dtype)


DM.read_fixed = read_fixed_all


@torch.no_grad()
def q_eq(m, items, dev, ac):
    """(phi(q) of the additive heads, qhat of the delta heads) at the decoder's ' =' position."""
    g = m.g; sq = SG.dec_seqs(items); ids_, _ = N.pad(sq); x = torch.tensor(ids_, device=dev); pos = torch.tensor([len(q_) - 1 for q_ in sq], device=dev)
    with ac():
        h = g.embed(x) + DM.wpe_slice(g, x.shape[1])[None]
        for j in range(m.mem.layer):
            h = DM._block(g, j, h, h.shape[-1] // g.H)
    h6 = h[torch.arange(len(items), device=dev), pos].float(); D = h6.shape[-1]; HA, HD_ = m.mem.h_add, m.mem.h_del
    with torch.autocast("cuda", enabled=False):
        q = m.mem.qkv(m.mem.ln(h6))[..., :D].view(len(items), HA + HD_, DH)
    return Fn.elu(q[:, :HA]) + 1, Fn.normalize(q[:, HA:], dim=-1)


def read_at(m, S, pq, qd):
    """the post-gain 768-d read at (pq, qd) from state S, through deltamem.add_read (the run's own add_norm, e.g. RMS) - as _mem_read."""
    HA, HD_ = m.mem.h_add, m.mem.h_del
    ra = torch.exp(m.mem.s_add) * DM.add_read(m.mem, pq, S[:, :HA].float(), DM.state_z(m.mem, S))
    rd = torch.exp(m.mem.s_del) * torch.einsum("bhvk,bhk->bhv", S[:, HA:HA + HD_].float(), qd)
    return torch.cat([ra, rd], 1).reshape(S.shape[0], -1)


def main():
    run = sys.argv[1]; dev = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else "cuda:0"; rd = os.path.join(SG.ROOT, "boxes", "runs_fsq", run)
    fp = os.path.join(rd, "eval_svr_guse.json"); base_rec = json.load(open(fp)); assert "ablation" in base_rec, "run the registered svr_guse.py first"
    kind, m, a = H.load_model(rd, dev); assert kind == "nested", kind; do_swap = "--swap" in sys.argv
    bs = int(sys.argv[sys.argv.index("--bs") + 1]) if "--bs" in sys.argv else int(a.get("eval_bs") or 250)
    DM.FAST_ADD = bool(a.get("fast_add")); DM.CHUNK_DELTA = 0; K = int(a["think_k"]); KA = 8; t0 = time.time(); HA, HDl = int(m.mem.h_add), int(m.mem.h_del); DD = (HA + HDl) * DH
    assert m.mem.s_del.dim() == 0 and m.mem.s_add.dim() == 0
    amp = None if (not dev.startswith("cuda") or torch.cuda.get_device_capability(dev)[0] < 8) else torch.bfloat16
    ac = lambda: torch.autocast("cuda", dtype=amp, enabled=amp is not None)  # noqa: E731
    TE = N.test_set(int(a.get("per_n") or 1000)); EXCL = {tuple(it["syms"]) for it in TE if it["n"] >= int(a.get("excl_min_n", 1))}
    r_sp = np.random.default_rng(17300)
    for _ in range(int(a.get("serial_pool") or 2200)):
        r__, _t = N.generate(r_sp, 8); EXCL.add(tuple(N.render(r__)))
    tr = N.train_items(np.random.default_rng(17021), 4000, 8, EXCL); pool = [it for it in TE if it["n"] <= 8]
    te = [pool[i] for i in np.random.default_rng(17022).permutation(len(pool))[:1000]]
    out = {"label": "DESCRIPTIVE post-hoc add-on (Theory 2026-10-14, after s185_rms_s1's delta-only ABL-NULL); registered rule unchanged", "K": K, "K_ablation": KA,
           "bs": bs, "card": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu", "dtype": str(amp), "add_norm": a.get("add_norm") or None}
    Rf, Rs = N.dec_reads(m, tr, K, dev, ac, bs), N.dec_reads(m, te, K, dev, ac, bs)
    vf = [N.reductions(it["root"]) for it in tr]; vs = [N.reductions(it["root"]) for it in te]; Xf, yf, Xs, ys = [], [], [], []
    for t in range(2, min(8, K) + 1):
        fi = [i for i, v in enumerate(vf) if len(v) >= t]; si = [i for i, v in enumerate(vs) if len(v) >= t]
        Xf.append(Rf[t - 1, fi]); yf += [vf[i][t - 1] for i in fi]; Xs.append(Rs[t - 1, si]); ys += [vs[i][t - 1] for i in si]
    Xf, Xs = np.concatenate(Xf).astype(np.float64), np.concatenate(Xs).astype(np.float64); yf, ys = np.array(yf), np.array(ys)
    mu, sd, W, b = SG.probe_fit(Xf, yf, dev); Qv = SG.orth((W - W.mean(1, keepdims=True)) / sd[:, None]); k = Qv.shape[1]
    acc_p, acc_abl = SG.probe_acc(Xs, ys, mu, sd, W, b), SG.probe_acc(Xs - (Xs @ Qv) @ Qv.T, ys, mu, sd, W, b)
    out["v_probe_all"] = {"rank": k, "acc_test_pooled": acc_p, "acc_test_ablated": acc_abl, "chance": 1 / N.P, "gate_G_abl": bool(abs(acc_abl - 1 / N.P) <= 0.02),
                          "n_fit": len(yf), "n_test": len(ys), "dims": DD}
    print(run, "V_probe_all", out["v_probe_all"], flush=True)
    PROJ.update(on=True, Q=None, collect=[torch.zeros(DD, DD, dtype=torch.float64, device=dev), 0])
    SG.answers(m, tr[:512], KA, dev, ac, bs); Sig = (PROJ["collect"][0] / PROJ["collect"][1]).cpu().numpy(); n_sites = PROJ["collect"][1]; PROJ.update(collect=None)
    ev, U = np.linalg.eigh(Sig); ev = np.clip(ev, 0, None); E = lambda Q_: float(np.trace(Q_.T @ Sig @ Q_))  # noqa: E731
    rng_h = np.random.default_rng(19950); R_rank = [SG.orth(rng_h.standard_normal((DD, k))) for _ in range(5)]
    rng_e = np.random.default_rng(19951); cands = []
    for j in range(200):
        be_ = (0.5, 1.0, 2.0)[j % 3]; Q_ = SG.orth((U * ev ** (be_ / 2)) @ U.T @ rng_e.standard_normal((DD, k))); cands.append((abs(E(Q_) - E(Qv)), j, be_, Q_))
    R_en = sorted(cands, key=lambda c_: c_[0])[:5]
    out["sigma_all"] = {"n_sites": int(n_sites), "E_V": E(Qv), "E_total": float(np.trace(Sig)), "E_rank": [E(Q_) for Q_ in R_rank], "E_energy": [E(c_[3]) for c_ in R_en],
                        "E_V_add_slice": float(np.trace(Qv[:HA * DH].T @ Sig[:HA * DH, :HA * DH] @ Qv[:HA * DH])), "V_mass_in_add_slice": float((Qv[:HA * DH] ** 2).sum() / k)}
    r18 = np.random.default_rng(18202); A_it = []
    for n_ in range(3, 9):
        pn = [it for it in TE if it["n"] == n_]; A_it += [pn[i] for i in r18.permutation(len(pn))[:400]]
    gold = np.array([it["t"] for it in A_it]); PROJ.update(on=False, Q=None); base = SG.answers(m, A_it, KA, dev, ac, bs)
    PROJ.update(on=True, Q=torch.zeros(DD, 0, device=dev)); ident = SG.answers(m, A_it, KA, dev, ac, bs); PROJ.update(on=False, Q=None)
    out["gate_IDENT"] = {"equal_items": int((base == ident).sum()), "n": len(A_it), "pass": bool((base == ident).all())}; print(run, "IDENT", out["gate_IDENT"], flush=True)
    c0 = (base == gold).astype(float)

    def abl(Q_):
        PROJ.update(on=True, Q=torch.tensor(Q_, device=dev, dtype=torch.float32)); p_ = SG.answers(m, A_it, KA, dev, ac, bs); PROJ.update(on=False, Q=None)
        return c0 - (p_ == gold).astype(float)
    dV = abl(Qv); dR = [abl(Q_) for Q_ in R_rank]; dE = [abl(c_[3]) for c_ in R_en]
    rb = np.random.default_rng(19970); dEl = dV - np.mean(dE, 0); dRl = dV - np.mean(dR, 0)
    ab = {"base_acc": float(c0.mean()), "drop_V": float(dV.mean()), "drop_rank": [float(d_.mean()) for d_ in dR], "drop_energy": [float(d_.mean()) for d_ in dE],
          "Delta_E": float(dEl.mean()), "Delta_E_ci95": SG.boot(dEl, rb), "Delta_rank": float(dRl.mean()), "Delta_rank_ci95": SG.boot(dRl, rb),
          "drop_V_above_all_10": bool(dV.mean() > max([d_.mean() for d_ in dR + dE])),
          "drop_V_by_n": {str(n_): float(dV[i * 400:(i + 1) * 400].mean()) for i, n_ in enumerate(range(3, 9))}}
    lo, hi = ab["Delta_E_ci95"]
    ab["reading"] = ("ABL-USED" if (ab["Delta_E"] >= 0.10 and ab["Delta_rank"] >= 0.10 and ab["drop_V_above_all_10"] and lo > 0.05)
                     else "ABL-NULL" if (ab["Delta_E"] <= 0.03 and hi <= 0.06) else "ABL-UNCLEAR")
    out["ablation_all"] = ab; print(run, "ABL_ALL", {k_: v_ for k_, v_ in ab.items() if k_ != "drop_V_by_n"}, flush=True)
    if do_swap:
        an = a.get("add_norm") or None; assert an in (None, "rms"), "--swap: no add_norm (exact edit) or rms (R1) only"
        out["swap_all_edit"] = "exact" if an is None else "approx (direction-only, RMS)"
        Wt, bt, mut, sdt = (torch.tensor(v_, device=dev, dtype=torch.float32) for v_ in (W, b, mu, sd))
        r60 = np.random.default_rng(19960); byn = {}
        for it in TE:
            if 3 <= it["n"] <= 8:
                byn.setdefault(it["n"], []).append(it)
        r2 = {id(it): SG.red2(it["root"]) for v_ in byn.values() for it in v_}; S_it, Dd, Ds, acf = [], [], [], []; n_undef = 0
        for it in A_it:                                                     # the registered donor selection, verbatim (same rng stream)
            op_, h_, v2 = SG.red2(it["root"]); cand = byn[it["n"]]; order = r60.permutation(len(cand)); pick = {}
            for want in ("diff", "same"):
                ok_ = lambda c_: (r2[id(c_)][2] != v2) if want == "diff" else (r2[id(c_)][2] == v2 and c_["syms"] != it["syms"])  # noqa: E731
                m1 = [cand[j] for j in order if ok_(cand[j]) and r2[id(cand[j])][:2] == (op_, h_)]
                m2 = m1 or [cand[j] for j in order if ok_(cand[j])]; pick[want] = m2[0] if m2 else None
            if pick["diff"] is None or pick["same"] is None:
                continue
            a2 = SG.answer_cf(it["root"], r2[id(pick["diff"])][2]); n_undef += a2 is None
            if a2 is not None and a2 != it["t"]:
                S_it.append(it); Dd.append(pick["diff"]); Ds.append(pick["same"]); acf.append(a2)
        acf = np.array(acf); goldS = np.array([it["t"] for it in S_it]); Qt = torch.tensor(Qv, device=dev, dtype=torch.float32); pqi, qdi = q_eq(m, S_it, dev, ac)
        ga, gd = torch.exp(m.mem.s_add).float(), torch.exp(m.mem.s_del).float(); exact = [0.0]

        def donor_reads(D_):
            S2 = SG.state_after(m, D_, 2, dev, ac, bs); pq_, qd_ = q_eq(m, D_, dev, ac); return read_at(m, S2, pq_, qd_)
        rD, rS = donor_reads(Dd), donor_reads(Ds)

        def mk_swap(rdon, rec):
            """the pass-2 edit; rec collects per item: the V_probe_all class of the EDITED read at q_eq (EDIT-SUCCESS) and the R1 fidelity terms."""
            def sw(S, i0, ch):
                S = S.float().clone(); pq_, qd_ = pqi[i0:i0 + len(ch)], qdi[i0:i0 + len(ch)]; r = read_at(m, S, pq_, qd_)
                dl = ((rdon[i0:i0 + len(ch)] - r) @ Qt) @ Qt.T; da = dl[:, :HA * DH].view(len(ch), HA, DH); dd = dl[:, HA * DH:].view(len(ch), HDl, DH)
                n2 = pq_.pow(2).sum(-1, keepdim=True)
                if an is None:                                              # exact: the read at q_eq moves by exactly delta_add
                    du = da / ga
                else:                                                       # R1 (RMS): u' = sign(g_eff) t rms(u) / rms(t), t = r_add + delta_add (direction only)
                    u = torch.einsum("bhk,bhkv->bhv", pq_, S[:, :HA]); t_ = r[:, :HA * DH].view(len(ch), HA, DH) + da
                    rms = lambda z_: torch.sqrt(z_.pow(2).mean(-1, keepdim=True) + 1e-6)  # noqa: E731
                    sg = torch.sign(ga * m.mem.add_gain.float())[None, :, None]; du = sg * t_ * rms(u) / rms(t_) - u
                S[:, :HA] = S[:, :HA] + pq_[..., :, None] * (du / n2)[..., None, :]
                S[:, HA:HA + HDl] = S[:, HA:HA + HDl] + (dd / gd)[..., :, None] * qd_[..., None, :]
                r2_ = read_at(m, S, pq_, qd_); exact[0] = max(exact[0], float((r2_ - (r + dl)).abs().max()))
                pr = ((((r2_ - mut) / sdt) @ Wt + bt).argmax(1)).cpu().numpy(); a_ = ((r2_ - r) @ Qt) @ Qt.T
                cs = Fn.cosine_similarity(a_, dl, dim=1).cpu().numpy(); ra_ = (a_.norm(dim=1) / dl.norm(dim=1).clamp_min(1e-12)).cpu().numpy()
                rec.append((pr, cs, ra_)); return S
            return sw
        recD, recS = [], []
        p0 = SG.answers(m, S_it, K, dev, ac, bs); pd_ = SG.answers(m, S_it, K, dev, ac, bs, swap=mk_swap(rD, recD)); ps_ = SG.answers(m, S_it, K, dev, ac, bs, swap=mk_swap(rS, recS))
        prD = np.concatenate([x_[0] for x_ in recD]); csD = np.concatenate([x_[1] for x_ in recD]); raD = np.concatenate([x_[2] for x_ in recD])
        prS = np.concatenate([x_[0] for x_ in recS]); v2d = np.array([r2[id(d_)][2] for d_ in Dd]); v2s = np.array([r2[id(d_)][2] for d_ in Ds])
        ed = prD == v2d; fl = (pd_ == acf).astype(float) - (ps_ == acf).astype(float); flo, fhi = SG.boot(fl, rb)
        flc = fl[ed]; flo_c, fhi_c = SG.boot(flc, rb) if ed.sum() else (None, None)
        sw = {"edit": out["swap_all_edit"], "n_items": len(S_it), "n_undefined_cf": int(n_undef),
              "edit_success_rate": float(ed.mean()), "edit_success_rate_sameval": float((prS == v2s).mean()), "n_edited": int(ed.sum()),
              "F": float((pd_ == acf).mean()), "F_null": float((ps_ == acf).mean()), "F_minus_null": float(fl.mean()), "F_minus_null_ci95": (flo, fhi),
              "F_edited": float((pd_ == acf)[ed].mean()) if ed.sum() else None, "F_null_edited": float((ps_ == acf)[ed].mean()) if ed.sum() else None,
              "F_minus_null_edited": float(flc.mean()) if ed.sum() else None, "F_minus_null_edited_ci95": (flo_c, fhi_c),
              "own_acc_noswap": float((p0 == goldS).mean()), "own_acc_sameswap": float((ps_ == goldS).mean()), "own_acc_diffswap": float((pd_ == goldS).mean()),
              "fidelity_cos_median": float(np.median(csD)), "fidelity_norm_ratio_median": float(np.median(raD)), "max_abs_read_minus_target": exact[0]}
        sw["disruption"] = sw["own_acc_noswap"] - sw["own_acc_sameswap"]
        sw["reading"] = ("INCONCLUSIVE (edit_success < 0.5)" if sw["edit_success_rate"] < 0.5 else "SWAP-DISRUPTIVE" if sw["disruption"] > 0.05
                         else "FOLLOWS" if (sw["F_minus_null"] >= 0.15 and flo > 0.05) else "IGNORES" if sw["F_minus_null"] <= 0.05 else "SWAP-UNCLEAR")
        out["swap_all"] = sw; print(run, "SWAP_ALL", sw, flush=True)
    out["sec"] = round(time.time() - t0)
    rec = json.load(open(fp)); rec["all_heads"] = {**rec.get("all_heads", {}), **out}; json.dump(rec, open(fp, "w", newline="\n"), indent=1); print(run, "ALL_HEADS done", out["sec"], "s", flush=True)


if __name__ == "__main__":
    main()
