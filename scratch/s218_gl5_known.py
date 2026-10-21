"""§218 Note 218-17: G-L5 PLANT known-answer test on the untrained LangSLVA (G-L5's model: langvb_eval.untrained(cal_n 16); G-L5's pool / held-out
(Stream 22500, rng 22500, hold 1000, pool 200,000) and plant directions (plant_dirs on pool[:2000], t_list (0, 1, 2), rng 22550 - the same call G-L5 makes,
so dirs[1] is G-L5's u / eps). t = 1, PLANT-lin only, fp32, tf32 off. No A1 / DIRECT output is touched. One JSON per arm -> scratch/gate_s218/g_l5_known/.
  K1 PATH   untrained probe (no step): s_A - s_B on 512 held-out items with z = 1 and z = 0 (same items), eps x {1, 10, 100} -> mean |delta|; the gradient
            norm of the probe CE (z ~ rng 22553) on the trainable probe parameters (blocks 10-11 copies + beta / gam); a slot-row check that the planted
            rows o1_slot / o2_slot are real slots (spad) and that only those two rows of V_t change.
  K2 ORACLE logistic regression on [V_1[o1_slot], V_1[o2_slot]] (planted) -> z: pool[2000:22000] (20,000 train) / pool[22000:24000] (2,000 test),
            eps x {1, 3, 10}; features standardised on train; L2 (weight decay on w) tuned on a 16k / 4k split of train, refit on all 20k.
  K3 PROBE-LEARN  the registered recipe (Probe(m, 1e-4): AdamW 1e-4 wd 0.01, the step-0 init; pbs 64; batches np rng 22501, z rng 22551 drawn as G-L5's
            zdraw), PLANT-lin t 1 alone, eps x {1, 10, 100}, 2,000 steps each, held-out accuracy (z rng 22553 restarted per read) every 250 steps.
    python -B scratch/s218_gl5_known.py --device cuda:0 [--arms k1,k2,k3]
  Theory 10-21: --arms overfit32,k3anti,k3antixor,k2anti,k2antixor,k3diag (OVERFIT-32 first, then item 6 antisymmetric plant, then K3-DIAG items 1-5, then K2-anti; Note 218-18 waits on them).
  Theory 10-20: --arms k2 --k2_mults 1.75 --k2_name k2_oracle_eps175 (the planned Note 218-18 plant, same oracle)."""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, "D:/Latent"); os.chdir("D:/Latent")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import boxes.langvb_eval as E  # noqa: E402

OUT = "scratch/gate_s218/g_l5_known"


def save(name, o):
    os.makedirs(OUT, exist_ok=True); json.dump(o, open(os.path.join(OUT, name + ".json"), "w", newline="\n"), indent=1); print("WROTE", name, flush=True)


def scaled(d, mult):
    d2 = dict(d); d2["eps"] = d["eps"] * mult; return d2


def zd(r_, n, dev):                                                         # = part_probe.zdraw
    return torch.tensor(r_.integers(0, 2, size=n), device=dev), torch.tensor(r_.integers(0, 2, size=n), device=dev)


def k1(m, held, d1, dev, chunk, bs, mults=(1, 10, 100)):
    its = held[:512]; res = {"n_items": len(its), "eps_base": d1["eps"], "by_mult": {}}
    pr = E.Probe(m, 1e-4)
    for mult in mults:
        d = scaled(d1, mult); dl, dsum = [], 0.0
        for ch in E.chunks(range(len(its)), bs):
            sub = [its[j] for j in ch]; b, S = E.stored(m, sub, 1, dev, chunk); n = len(sub); one = torch.ones(n, dtype=torch.long, device=dev); zero = 0 * one
            with torch.no_grad():
                a1, b1, _ = E.probe_forward(m, pr, "lin", 1, sub, b, S, {1: d}, (one, zero), dev)
                a0, b0, _ = E.probe_forward(m, pr, "lin", 1, sub, b, S, {1: d}, (zero, zero), dev)
                dl += ((a1 - b1) - (a0 - b0)).abs().tolist()
                if mult == 1 and ch[0] == 0:                                # slot-row check on the first chunk
                    sp = b[4]; Vt = S[1]; V1 = E.plant_V(Vt, sub, one, d); V0 = E.plant_V(Vt, sub, zero, d)
                    rows_changed = ((V1 - V0).abs().amax(-1) > 0)            # (B, S)
                    o1 = torch.tensor([it["o1_slot"] for it in sub], device=dev); o2 = torch.tensor([it["o2_slot"] for it in sub], device=dev)
                    ar = torch.arange(n, device=dev); expect = torch.zeros_like(rows_changed); expect[ar, o1] = True; expect[ar, o2] = True
                    res["slot_check"] = {"V_shape": list(Vt.shape), "spad_shape": list(sp.shape), "o1_real": bool(sp[ar, o1].all()), "o2_real": bool(sp[ar, o2].all()),
                                         "o1_ne_o2": bool((o1 != o2).all()), "only_option_rows_change": bool((rows_changed == expect).all()),
                                         "n_slots_real_mean": float(sp.float().sum(1).mean()), "o1_slot_max": int(o1.max()), "o2_slot_max": int(o2.max())}
            del S
        # gradient of the probe CE at the untrained probe (no optimiser step), z from rng 22553 on the first 64 held items
        sub = its[:64]; b, S = E.stored(m, sub, 1, dev, chunk); z1, z2 = zd(np.random.default_rng(22553), len(sub), dev)
        pr.opt.zero_grad(set_to_none=True); sA, sB, y = E.probe_forward(m, pr, "lin", 1, sub, b, S, {1: d}, (z1, z2), dev)
        loss = F.cross_entropy(torch.stack([sA, sB], 1), y); loss.backward()
        ps = list(pr.P.parameters()) + [pr.beta, pr.gam]; gn = [float(p.grad.norm()) if p.grad is not None else None for p in ps]
        res["by_mult"][str(mult)] = {"mean_abs_delta": float(np.mean(dl)), "max_abs_delta": float(np.max(dl)), "min_abs_delta": float(np.min(dl)),
                                     "loss": float(loss), "grad_norm_total": float(np.sqrt(sum(g * g for g in gn if g is not None))),
                                     "grad_none": sum(g is None for g in gn), "grad_zero": sum(g == 0.0 for g in gn if g is not None), "n_param_tensors": len(gn),
                                     "acc_untrained_z1_vs_z0_sign": float(np.mean([v > 0 for v in dl]))}
        pr.opt.zero_grad(set_to_none=True); del S
        print("K1", mult, res["by_mult"][str(mult)], flush=True)
    res["K1_zero"] = all(v["mean_abs_delta"] <= 1e-6 for v in res["by_mult"].values())
    return res


def rows_V1(m, its, dev, chunk, bs):
    """V_1 at o1_slot / o2_slot, unplanted -> (N, 2, D) float64 cpu."""
    out = []
    for ch in E.chunks(range(len(its)), bs * 4):
        sub = [its[j] for j in ch]; b, S = E.stored(m, sub, 1, dev, chunk); ar = torch.arange(len(sub), device=dev)
        o1 = torch.tensor([it["o1_slot"] for it in sub], device=dev); o2 = torch.tensor([it["o2_slot"] for it in sub], device=dev)
        out.append(torch.stack([S[1][ar, o1], S[1][ar, o2]], 1).double().cpu()); del S
    return torch.cat(out, 0)


def logreg(Xtr, ytr, Xte, yte, l2, dev, iters=300):
    w = torch.zeros(Xtr.shape[1], dtype=torch.float64, device=dev, requires_grad=True); c = torch.zeros(1, dtype=torch.float64, device=dev, requires_grad=True)
    opt = torch.optim.LBFGS([w, c], lr=1, max_iter=iters, line_search_fn="strong_wolfe"); Xt, yt = Xtr.to(dev), ytr.to(dev)

    def clo():
        opt.zero_grad(); L = F.binary_cross_entropy_with_logits(Xt @ w + c, yt) + l2 * (w @ w); L.backward(); return L
    opt.step(clo)
    with torch.no_grad():
        return float(((Xte.to(dev) @ w + c > 0).double() == yte.to(dev)).double().mean())


def k2(m, pool, d1, dev, chunk, bs, mults=(1, 3, 10)):
    tr, te = pool[2000:22000], pool[22000:24000]; t0 = time.time()
    Rtr, Rte = rows_V1(m, tr, dev, chunk, bs), rows_V1(m, te, dev, chunk, bs); u = d1["u"].double().cpu()
    ztr = torch.tensor(np.random.default_rng(22571).integers(0, 2, len(tr)), dtype=torch.float64); zte = torch.tensor(np.random.default_rng(22572).integers(0, 2, len(te)), dtype=torch.float64)
    res = {"n_train": len(tr), "n_test": len(te), "rows_sec": round(time.time() - t0), "eps_base": d1["eps"], "z_rng": [22571, 22572], "by_mult": {}}
    # the derivation's quantities: noise std along u per option row, and the correlation of the two rows' projections
    p1, p2 = Rtr[:, 0] @ u, Rtr[:, 1] @ u
    res["proj_std_u"] = [float(p1.std()), float(p2.std())]; res["proj_corr_u"] = float(torch.corrcoef(torch.stack([p1, p2]))[0, 1])
    grid = [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    for mult in mults:
        e = d1["eps"] * mult
        Xtr = (Rtr + e * (2 * ztr - 1)[:, None, None] * u[None, None]).reshape(len(tr), -1); Xte = (Rte + e * (2 * zte - 1)[:, None, None] * u[None, None]).reshape(len(te), -1)
        mu, sd = Xtr.mean(0), Xtr.std(0).clamp_min(1e-8); Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
        val = {str(l2): logreg(Xtr[:16000], ztr[:16000], Xtr[16000:], ztr[16000:], l2, dev) for l2 in grid}
        best = max(grid, key=lambda l2: (val[str(l2)], l2)); acc = logreg(Xtr, ztr, Xte, zte, best, dev)
        # 1-d along u (the planted direction itself, no fit beyond the sign): the projection sum of both rows
        s_te = (Rte[:, 0] + Rte[:, 1]) @ u + 2 * e * (2 * zte - 1); thr = float(((Rtr[:, 0] + Rtr[:, 1]) @ u).mean())
        res["by_mult"][str(mult)] = {"eps": e, "val_acc_by_l2": val, "l2": best, "test_acc": acc, "u_proj_sum_acc": float(((s_te > thr).double() == zte).double().mean())}
        print("K2", mult, res["by_mult"][str(mult)]["test_acc"], res["by_mult"][str(mult)]["u_proj_sum_acc"], flush=True)
    return res


def k3(m, pool, held, d1, dev, chunk, bs, steps=2000, every=250, pbs=64, mults=(1, 10, 100), kind="lin"):
    res = {"recipe": "Probe(m, 1e-4) (AdamW lr 1e-4 wd 0.01, step-0 init); pbs 64; batches np rng 22501; z rng 22551 (zdraw); held-out 1000, z rng 22553 per read",
           "eps_base": d1["eps"], "by_mult": {}}
    for mult in mults:
        d = {1: scaled(d1, mult)}; pr = E.Probe(m, 1e-4); rb = np.random.default_rng(22501); rz = np.random.default_rng(22551); curve = []; t0 = time.time()
        for st in range(1, steps + 1):
            ix = rb.integers(0, len(pool), pbs); sub = [pool[j] for j in ix]; b, S = E.stored(m, sub, 1, dev, chunk); zs = zd(rz, pbs, dev)
            pr.opt.zero_grad(set_to_none=True); sA, sB, y = E.probe_forward(m, pr, kind, 1, sub, b, S, d, zs, dev)
            F.cross_entropy(torch.stack([sA, sB], 1), y).backward(); pr.opt.step(); del S
            if st % every == 0:
                rh = np.random.default_rng(22553); zz = zd(rh, len(held), dev); ok = []
                for ch in E.chunks(range(len(held)), bs):
                    sub = [held[j] for j in ch]; b, S = E.stored(m, sub, 1, dev, chunk)
                    with torch.no_grad():
                        sA, sB, y = E.probe_forward(m, pr, kind, 1, sub, b, S, d, (zz[0][ch], zz[1][ch]), dev); ok += ((sA > sB).long() == (1 - y)).long().tolist()
                    del S
                curve.append((st, float(np.mean(ok)))); print("K3", mult, st, curve[-1][1], round(time.time() - t0), "s", flush=True)
        first = np.mean([a for s_, a in curve if s_ <= 750]); last = np.mean([a for s_, a in curve if s_ > steps - 750])
        res["by_mult"][str(mult)] = {"eps": d[1]["eps"], "curve": curve, "final": curve[-1][1], "max": max(a for _, a in curve), "first750_mean": float(first),
                                     "last750_mean": float(last), "rising_005": bool(last - first >= 0.05), "sec": round(time.time() - t0)}
        del pr
        if dev.startswith("cuda"):
            torch.cuda.empty_cache()
    return res


def plant_V_anti(Vt, its, z, d, xor=False, z2=None):
    """Theory 10-21 item 6 (scratch only; Note 218-18 candidate): ANTISYMMETRIC plant - o1 row += eps (2z - 1) u (+ eps (2 z2 - 1) u2 for xor), o2 row -= the same."""
    V = Vt.clone(); B = V.shape[0]; ar = torch.arange(B, device=V.device)
    add = d["eps"] * (2 * z - 1).float()[:, None] * d["u"][None]
    if xor:
        add = add + d["eps"] * (2 * z2 - 1).float()[:, None] * d["u2"][None]
    j1 = torch.tensor([it["o1_slot"] for it in its], device=V.device); j2 = torch.tensor([it["o2_slot"] for it in its], device=V.device)
    V[ar, j1] = V[ar, j1] + add.to(V.dtype); V[ar, j2] = V[ar, j2] - add.to(V.dtype)
    return V


def k3anti(m, pool, held, d1, dev, chunk, bs, mults=(1, 10), steps=2000):
    """item 6: K3 (the registered probe recipe, PLANT-lin t 1, 2,000 steps, held-out every 250) with the ANTISYMMETRIC plant (E.plant_V patched in this
    process only), at eps x1 and x10; + K1-style mean |delta(sA - sB)| (z 1 vs 0) on 512 held items at the untrained probe."""
    orig = E.plant_V; E.plant_V = plant_V_anti
    try:
        r = k3(m, pool, held, d1, dev, chunk, bs, steps=steps, mults=mults)
        r["k1_anti"] = {str(mu): k1(m, held, d1, dev, chunk, bs, mults=(mu,))["by_mult"][str(mu)] for mu in mults}
    finally:
        E.plant_V = orig
    r["plant"] = plant_V_anti.__doc__; return r


def k2anti(m, pool, d1, dev, chunk, bs, mults=(1, 1.5, 2)):
    """K2-anti: the oracle under the antisymmetric plant: logistic on [r1, r2] and the 1-d difference projection (r1 - r2) . u (threshold = the train mean)."""
    tr, te = pool[2000:22000], pool[22000:24000]; Rtr, Rte = rows_V1(m, tr, dev, chunk, bs), rows_V1(m, te, dev, chunk, bs); u = d1["u"].double().cpu()
    ztr = torch.tensor(np.random.default_rng(22571).integers(0, 2, len(tr)), dtype=torch.float64); zte = torch.tensor(np.random.default_rng(22572).integers(0, 2, len(te)), dtype=torch.float64)
    res = {"n_train": len(tr), "n_test": len(te), "eps_base": d1["eps"], "by_mult": {}}; grid = [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    sgn = torch.tensor([1.0, -1.0], dtype=torch.float64)
    for mult in mults:
        e = d1["eps"] * mult
        Ptr = Rtr + e * (2 * ztr - 1)[:, None, None] * sgn[None, :, None] * u[None, None]; Pte = Rte + e * (2 * zte - 1)[:, None, None] * sgn[None, :, None] * u[None, None]
        Xtr, Xte = Ptr.reshape(len(tr), -1), Pte.reshape(len(te), -1); mu, sd = Xtr.mean(0), Xtr.std(0).clamp_min(1e-8); Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
        val = {str(l2): logreg(Xtr[:16000], ztr[:16000], Xtr[16000:], ztr[16000:], l2, dev) for l2 in grid}
        best = max(grid, key=lambda l2: (val[str(l2)], l2)); acc = logreg(Xtr, ztr, Xte, zte, best, dev)
        dtr = (Ptr[:, 0] - Ptr[:, 1]) @ u; dte = (Pte[:, 0] - Pte[:, 1]) @ u; thr = float(((Rtr[:, 0] - Rtr[:, 1]) @ u).mean())
        res["by_mult"][str(mult)] = {"eps": e, "l2": best, "test_acc": acc, "diff_proj_acc": float(((dte > thr).double() == zte).double().mean()), "val_acc_by_l2": val}
        print("K2ANTI", mult, acc, res["by_mult"][str(mult)]["diff_proj_acc"], flush=True)
    return res


def psha(pr):
    import hashlib
    h = hashlib.sha256()
    for k in sorted(pr.P.keys()):
        h.update(pr.P[k].detach().float().cpu().numpy().tobytes())
    for t_ in (pr.beta, pr.gam):
        h.update(t_.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()


def k3antixor(m, pool, held, d1, dev, chunk, bs, mults=(1, 10), steps=2000):
    """Theory 10-21: item 6 in XOR form - the registered K3 recipe, PLANT-xor t 1 (target z1 xor z2), with the ANTISYMMETRIC plant (o1 += eps (2z1 - 1) u
    + eps (2z2 - 1) u2, o2 -= the same; E.plant_V patched in this process only), eps x1 and x10, 2,000 steps, held-out every 250."""
    orig = E.plant_V; E.plant_V = plant_V_anti
    try:
        r = k3(m, pool, held, d1, dev, chunk, bs, steps=steps, mults=mults, kind="xor")
    finally:
        E.plant_V = orig
    r["plant"] = plant_V_anti.__doc__ + " (xor form)"; return r


def k2antixor(m, pool, d1, dev, chunk, bs, mults=(1, 1.5, 2)):
    """K2-anti-xor: under the antisymmetric xor plant, the sign-product oracle sign((r1 - r2).u - thr_u) x sign((r1 - r2).u2 - thr_u2) -> z1 xor z2
    (thresholds = the unplanted train means), and a learned reference: logreg on [pu, pu2, pu * pu2] (centred projections), L2 tuned on a 16k / 4k split."""
    tr, te = pool[2000:22000], pool[22000:24000]; Rtr, Rte = rows_V1(m, tr, dev, chunk, bs), rows_V1(m, te, dev, chunk, bs)
    u = d1["u"].double().cpu(); u2 = d1["u2"].double().cpu(); res = {"n_train": len(tr), "n_test": len(te), "eps_base": d1["eps"], "z_rng": [22573, 22574, 22575, 22576], "by_mult": {}}
    zt = [torch.tensor(np.random.default_rng(sd).integers(0, 2, n_), dtype=torch.float64) for sd, n_ in ((22573, len(tr)), (22574, len(tr)), (22575, len(te)), (22576, len(te)))]
    z1tr, z2tr, z1te, z2te = zt; ytr, yte = (z1tr != z2tr).double(), (z1te != z2te).double(); grid = [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    D0tr = Rtr[:, 0] - Rtr[:, 1]; D0te = Rte[:, 0] - Rte[:, 1]; tu, tu2 = float((D0tr @ u).mean()), float((D0tr @ u2).mean())
    for mult in mults:
        e = d1["eps"] * mult
        Dtr = D0tr + 2 * e * ((2 * z1tr - 1)[:, None] * u[None] + (2 * z2tr - 1)[:, None] * u2[None]); Dte = D0te + 2 * e * ((2 * z1te - 1)[:, None] * u[None] + (2 * z2te - 1)[:, None] * u2[None])
        pa, pb = Dte @ u - tu, Dte @ u2 - tu2; oracle = float((((pa * pb) < 0).double() == yte).double().mean())
        def feats(D_):
            x1, x2 = D_ @ u - tu, D_ @ u2 - tu2; return torch.stack([x1, x2, x1 * x2], 1)
        Ftr, Fte = feats(Dtr), feats(Dte); mu, sd = Ftr.mean(0), Ftr.std(0).clamp_min(1e-8); Ftr, Fte = (Ftr - mu) / sd, (Fte - mu) / sd
        val = {str(l2): logreg(Ftr[:16000], ytr[:16000], Ftr[16000:], ytr[16000:], l2, dev) for l2 in grid}
        best = max(grid, key=lambda l2: (val[str(l2)], l2)); acc = logreg(Ftr, ytr, Fte, yte, best, dev)
        res["by_mult"][str(mult)] = {"eps": e, "sign_product_oracle_acc": oracle, "logreg_product_acc": acc, "l2": best, "val_acc_by_l2": val}
        print("K2ANTIXOR", mult, oracle, acc, flush=True)
    return res


def overfit32(m, pool, d1, dev, chunk, mult=10, steps=300, every=10):
    """Theory / Audit 10-21 item 0: OVERFIT-32 - the registered probe (Probe(m, 1e-4)) trained on 32 FIXED items (pool[24000:24032], fixed z rng 22581)
    as one full batch per step, eps x10, train accuracy + CE every 10 steps; with the registered SYMMETRIC plant and the ANTISYMMETRIC plant (item 6).
    Also Audit's param-identity check: sha(probe ParameterDict + beta + gam) at the end of training == at the start of the read, and the read uses pr.src."""
    its = pool[24000:24032]; z1 = torch.tensor(np.random.default_rng(22581).integers(0, 2, 32), device=dev); z2 = torch.zeros_like(z1)
    d = {1: scaled(d1, mult)}; res = {"eps": d[1]["eps"], "mult": mult, "steps": steps, "items": "pool[24000:24032]", "z_rng": 22581, "z_mean": float(z1.float().mean())}
    b, S = E.stored(m, its, 1, dev, chunk); orig = E.plant_V
    try:
        for name, fn in (("symmetric", orig), ("antisymmetric", plant_V_anti)):
            E.plant_V = fn; pr = E.Probe(m, 1e-4); rec = []
            for st in range(1, steps + 1):
                pr.opt.zero_grad(set_to_none=True); sA, sB, y = E.probe_forward(m, pr, "lin", 1, its, b, S, d, (z1, z2), dev)
                L = F.cross_entropy(torch.stack([sA, sB], 1), y); acc = float(((sA > sB).long() == (1 - y)).float().mean())
                if st % every == 0 or st == 1:
                    rec.append({"step": st, "train_ce": float(L), "train_acc_before_step": acc, "mean_abs_sA_sB": float((sA - sB).abs().mean())})
                    print("OVERFIT32", name, st, round(float(L), 4), acc, flush=True)
                L.backward(); pr.opt.step()
            sha_end = psha(pr); reads = {"P": 0, "g": 0}; w0 = pr.src.w
            def w_hook(k, _w0=w0):
                reads["P" if k.replace(".", "_") in pr.src.P else "g"] += 1; return _w0(k)
            pr.src.w = w_hook; sha_read = psha(pr)
            with torch.no_grad():
                sA, sB, y = E.probe_forward(m, pr, "lin", 1, its, b, S, d, (z1, z2), dev)
            res[name] = {"curve": rec, "final_train_acc": float(((sA > sB).long() == (1 - y)).float().mean()), "param_sha_end_train": sha_end,
                         "param_sha_at_read": sha_read, "param_identity_ok": sha_end == sha_read, "read_weights_from_probe_copy": reads["P"], "read_weights_from_trunk": reads["g"]}
            print("OVERFIT32", name, "final", res[name]["final_train_acc"], "identity", res[name]["param_identity_ok"], reads, flush=True)
            del pr
    finally:
        E.plant_V = orig
    del S
    return res


def k3diag(m, pool, held, d1, dev, chunk, bs, steps=500, every=25, pbs=64, mult=10):
    """Theory 10-21 (Note 218-18 waits on it): K3-DIAG at eps x10, t 1, PLANT-lin, 500 steps, log every 25: (1) train-batch CE / acc on the batch the step
    trained on (before the step) + held-out acc (first 256 held items, z rng 22553; all 1,000 at the end); (2) plant-z == target-z asserted per item in
    train and held-out (sign of the planted u-component at o1 / o2 == 2 z - 1; target index == 1 - z), 8 items logged (z, target, sA, sB); (3) grad norm and
    update norm per group (blocks 10-11, beta, gam; beta0 is a frozen tensor) + AdamW membership + the weights pr.scores actually reads (Src.w hooked);
    control A: a linear head on [V_1[o1_slot], V_1[o2_slot]] (planted), same AdamW 1e-4 wd 0.01 / bs 64 / batch + z streams; control B: the coda probe
    trained on the FLIPPED target (y instead of 1 - y), read with the registered accuracy (learning + correct eval -> ~0.0)."""
    d = {1: scaled(d1, mult)}; u = d[1]["u"]; out = {"eps": d[1]["eps"], "mult": mult, "steps": steps, "every": every}

    def zchk(V, Vt, its, z, tgt):
        ar = torch.arange(len(its), device=dev); o1 = torch.tensor([it["o1_slot"] for it in its], device=dev); o2 = torch.tensor([it["o2_slot"] for it in its], device=dev)
        s1 = torch.sign((V - Vt)[ar, o1] @ u); s2 = torch.sign((V - Vt)[ar, o2] @ u); want = (2 * z - 1).float()
        return bool((s1 == want).all() and (s2 == want).all() and (tgt == (1 - z)).all())

    def held_acc(pr, n, flip=False):
        rh = np.random.default_rng(22553); zz = zd(rh, len(held), dev); ok = []; zok = True
        for ch in E.chunks(range(n), bs):
            sub = [held[j] for j in ch]; b, S = E.stored(m, sub, 1, dev, chunk); z1 = zz[0][ch]
            with torch.no_grad():
                sA, sB, y = E.probe_forward(m, pr, "lin", 1, sub, b, S, d, (z1, zz[1][ch]), dev)
                V = E.plant_V(S[1], sub, z1, d[1]); zok &= zchk(V, S[1], sub, z1, y)
                ok += ((sA > sB).long() == (1 - y)).long().tolist()
            del S
        return float(np.mean(ok)), zok

    def run_coda(flip):
        pr = E.Probe(m, 1e-4); rb = np.random.default_rng(22501); rz = np.random.default_rng(22551); rec = []
        groups = {"blocks_10_11": list(pr.P.parameters()), "beta": [pr.beta], "gam": [pr.gam]}
        in_opt = {id(p_) for g_ in pr.opt.param_groups for p_ in g_["params"]}
        memb = {k: [id(p_) in in_opt for p_ in v] for k, v in groups.items()}
        reads = {"from_P": 0, "from_g": 0, "keys_P": set(), "keys_g": set()}; w0 = pr.src.w
        def w_hook(k, _w0=w0):
            kk = k.replace(".", "_")
            if kk in pr.src.P:
                reads["from_P"] += 1; reads["keys_P"].add(kk)
            else:
                reads["from_g"] += 1; reads["keys_g"].add(k)
            return _w0(k)
        pr.src.w = w_hook; logged8 = None; zok_all = True
        for st in range(1, steps + 1):
            ix = rb.integers(0, len(pool), pbs); sub = [pool[j] for j in ix]; b, S = E.stored(m, sub, 1, dev, chunk); zs = zd(rz, pbs, dev)
            pr.opt.zero_grad(set_to_none=True); sA, sB, y = E.probe_forward(m, pr, "lin", 1, sub, b, S, d, zs, dev)
            tgt = (1 - y) if flip else y
            with torch.no_grad():
                V = E.plant_V(S[1], sub, zs[0], d[1]); zok_all &= zchk(V, S[1], sub, zs[0], y)
                tr_acc = float(((sA > sB).long() == (1 - y)).float().mean())
            loss = F.cross_entropy(torch.stack([sA, sB], 1), tgt); loss.backward()
            gn = {k: float(np.sqrt(sum(float(p_.grad.norm()) ** 2 for p_ in v if p_.grad is not None))) for k, v in groups.items()}
            before = {k: [p_.detach().clone() for p_ in v] for k, v in groups.items()}
            pr.opt.step()
            un = {k: float(np.sqrt(sum(float((p_.detach() - b_).norm()) ** 2 for p_, b_ in zip(v, before[k])))) for k, v in groups.items()}; del before
            if logged8 is None:
                logged8 = [{"z": int(zs[0][q]), "target": int(y[q]), "sA": float(sA[q]), "sB": float(sB[q])} for q in range(8)]
            if st % every == 0 or st == 1:
                ha, hz = held_acc(pr, 256); zok_all &= hz
                rec.append({"step": st, "train_ce": float(loss), "train_acc": tr_acc, "held256_acc": ha, "grad_norm": gn, "update_norm": un})
                print("K3DIAG", "flip" if flip else "coda", st, round(float(loss), 4), round(tr_acc, 3), round(ha, 3), {k: round(v, 4) for k, v in gn.items()}, flush=True)
            del S
        fa, fz = held_acc(pr, len(held)); zok_all &= fz
        res = {"curve": rec, "held1000_final": fa, "z_plant_eq_target_all": bool(zok_all), "items8_first_step": logged8, "opt_membership": memb,
               "n_param_tensors": {k: len(v) for k, v in groups.items()}, "src_reads": {"from_P": reads["from_P"], "from_g": reads["from_g"],
               "keys_P": sorted(reads["keys_P"]), "keys_g_sample": sorted(reads["keys_g"])[:40]}}
        del pr
        if dev.startswith("cuda"):
            torch.cuda.empty_cache()
        return res

    out["coda"] = run_coda(False)
    out["control_B_flipped"] = run_coda(True)
    # control A: linear head on the planted option rows, same optimiser / stream
    head = torch.nn.Linear(2 * u.numel(), 1).to(dev); opt = torch.optim.AdamW(head.parameters(), lr=1e-4, weight_decay=0.01)
    rb = np.random.default_rng(22501); rz = np.random.default_rng(22551); recA = []

    def rowsP(sub, z1):
        b, S = E.stored(m, sub, 1, dev, chunk); V = E.plant_V(S[1], sub, z1, d[1]); ar = torch.arange(len(sub), device=dev)
        o1 = torch.tensor([it["o1_slot"] for it in sub], device=dev); o2 = torch.tensor([it["o2_slot"] for it in sub], device=dev)
        X = torch.cat([V[ar, o1], V[ar, o2]], 1).float(); del S; return X

    def heldA(n):
        rh = np.random.default_rng(22553); zz = zd(rh, len(held), dev); ok = []
        for ch in E.chunks(range(n), bs * 4):
            sub = [held[j] for j in ch]; X = rowsP(sub, zz[0][ch])
            with torch.no_grad():
                ok += ((head(X).squeeze(-1) > 0).long() == zz[0][ch]).long().tolist()
        return float(np.mean(ok))
    for st in range(1, steps + 1):
        ix = rb.integers(0, len(pool), pbs); sub = [pool[j] for j in ix]; zs = zd(rz, pbs, dev); X = rowsP(sub, zs[0])
        opt.zero_grad(set_to_none=True); L = F.binary_cross_entropy_with_logits(head(X).squeeze(-1), zs[0].float()); L.backward(); opt.step()
        if st % every == 0 or st == 1:
            recA.append({"step": st, "train_bce": float(L), "held256_acc": heldA(256)}); print("K3DIAG ctrlA", st, round(float(L), 4), recA[-1]["held256_acc"], flush=True)
    out["control_A_linear_head"] = {"curve": recA, "held1000_final": heldA(len(held)), "features": "raw [V_1[o1_slot], V_1[o2_slot]] planted (no standardisation)"}
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--arms", default="k1,k2,k3"); ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=16); ap.add_argument("--k3_steps", type=int, default=2000); ap.add_argument("--pool_n", type=int, default=200000); ap.add_argument("--k2_mults", default="1,3,10"); ap.add_argument("--k2_name", default="k2_oracle")
    a = ap.parse_args(); dev = a.device; torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False; t0 = time.time()
    m, info = E.untrained(dev, 16)
    excl = E.PS.exclusion_sigs(); s = E.PS.Stream(22500, excl); rng = np.random.default_rng(22500); Dseq = rng.integers(3, 9, size=1000 + a.pool_n)
    held = [E.ev(s.draw(int(D))) for D in Dseq[:1000]]; pool = [E.ev(s.draw(int(D))) for D in Dseq[1000:]]
    dirs = E.plant_dirs(m, pool, [0, 1, 2], dev, a.chunk); d1 = dirs[1]
    meta = {"note": "Note 218-17", "model": "langvb_eval.untrained(cal_n 16)", "untrained_cal": info, "card": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu",
            "dtype": "fp32", "tf32": False, "pool_n": len(pool), "held_n": len(held), "eps": d1["eps"], "eps_rel": d1["eps_rel"], "vnorm": d1["vnorm"],
            "eig_40_60": d1["eig"][39:60], "eig_50": d1["eig"][49], "setup_sec": round(time.time() - t0), "commit": os.popen("git rev-parse --short HEAD").read().strip()}
    print("SETUP", meta["setup_sec"], "s eps", d1["eps"], flush=True)
    arms = a.arms.split(",")
    if "k1" in arms:
        r = k1(m, held, d1, dev, a.chunk, a.bs); save("k1_path", {**meta, **r, "rule": "K1 = 0 to float precision (mean |delta| <= 1e-6 at every eps) -> BUG"})
    if "k2" in arms:
        r = k2(m, pool, d1, dev, a.chunk, a.bs, tuple(float(v) for v in a.k2_mults.split(","))); save(a.k2_name, {**meta, **r})
    if "overfit32" in arms:
        r = overfit32(m, pool, d1, dev, a.chunk); save("overfit32", {**meta, **r})
    if "k3anti" in arms:
        r = k3anti(m, pool, held, d1, dev, a.chunk, a.bs); save("k3_anti", {**meta, **r})
    if "k3antixor" in arms:
        r = k3antixor(m, pool, held, d1, dev, a.chunk, a.bs); save("k3_anti_xor", {**meta, **r})
    if "k2antixor" in arms:
        r = k2antixor(m, pool, d1, dev, a.chunk, a.bs); save("k2_anti_xor", {**meta, **r})
    if "k2anti" in arms:
        r = k2anti(m, pool, d1, dev, a.chunk, a.bs); save("k2_anti", {**meta, **r})
    if "k3diag" in arms:
        r = k3diag(m, pool, held, d1, dev, a.chunk, a.bs); save("k3_diag", {**meta, **r})
    if "k3" in arms:
        r = k3(m, pool, held, d1, dev, a.chunk, a.bs, steps=a.k3_steps); save("k3_probe_learn", {**meta, **r})


if __name__ == "__main__":
    main()
