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
    python -B scratch/s218_gl5_known.py --device cuda:0 [--arms k1,k2,k3]"""
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


def k1(m, held, d1, dev, chunk, bs):
    its = held[:512]; res = {"n_items": len(its), "eps_base": d1["eps"], "by_mult": {}}
    pr = E.Probe(m, 1e-4)
    for mult in (1, 10, 100):
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


def k2(m, pool, d1, dev, chunk, bs):
    tr, te = pool[2000:22000], pool[22000:24000]; t0 = time.time()
    Rtr, Rte = rows_V1(m, tr, dev, chunk, bs), rows_V1(m, te, dev, chunk, bs); u = d1["u"].double().cpu()
    ztr = torch.tensor(np.random.default_rng(22571).integers(0, 2, len(tr)), dtype=torch.float64); zte = torch.tensor(np.random.default_rng(22572).integers(0, 2, len(te)), dtype=torch.float64)
    res = {"n_train": len(tr), "n_test": len(te), "rows_sec": round(time.time() - t0), "eps_base": d1["eps"], "z_rng": [22571, 22572], "by_mult": {}}
    # the derivation's quantities: noise std along u per option row, and the correlation of the two rows' projections
    p1, p2 = Rtr[:, 0] @ u, Rtr[:, 1] @ u
    res["proj_std_u"] = [float(p1.std()), float(p2.std())]; res["proj_corr_u"] = float(torch.corrcoef(torch.stack([p1, p2]))[0, 1])
    grid = [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    for mult in (1, 3, 10):
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


def k3(m, pool, held, d1, dev, chunk, bs, steps=2000, every=250, pbs=64):
    res = {"recipe": "Probe(m, 1e-4) (AdamW lr 1e-4 wd 0.01, step-0 init); pbs 64; batches np rng 22501; z rng 22551 (zdraw); held-out 1000, z rng 22553 per read",
           "eps_base": d1["eps"], "by_mult": {}}
    for mult in (1, 10, 100):
        d = {1: scaled(d1, mult)}; pr = E.Probe(m, 1e-4); rb = np.random.default_rng(22501); rz = np.random.default_rng(22551); curve = []; t0 = time.time()
        for st in range(1, steps + 1):
            ix = rb.integers(0, len(pool), pbs); sub = [pool[j] for j in ix]; b, S = E.stored(m, sub, 1, dev, chunk); zs = zd(rz, pbs, dev)
            pr.opt.zero_grad(set_to_none=True); sA, sB, y = E.probe_forward(m, pr, "lin", 1, sub, b, S, d, zs, dev)
            F.cross_entropy(torch.stack([sA, sB], 1), y).backward(); pr.opt.step(); del S
            if st % every == 0:
                rh = np.random.default_rng(22553); zz = zd(rh, len(held), dev); ok = []
                for ch in E.chunks(range(len(held)), bs):
                    sub = [held[j] for j in ch]; b, S = E.stored(m, sub, 1, dev, chunk)
                    with torch.no_grad():
                        sA, sB, y = E.probe_forward(m, pr, "lin", 1, sub, b, S, d, (zz[0][ch], zz[1][ch]), dev); ok += ((sA > sB).long() == (1 - y)).long().tolist()
                    del S
                curve.append((st, float(np.mean(ok)))); print("K3", mult, st, curve[-1][1], round(time.time() - t0), "s", flush=True)
        first = np.mean([a for s_, a in curve if s_ <= 750]); last = np.mean([a for s_, a in curve if s_ > steps - 750])
        res["by_mult"][str(mult)] = {"eps": d[1]["eps"], "curve": curve, "final": curve[-1][1], "max": max(a for _, a in curve), "first750_mean": float(first),
                                     "last750_mean": float(last), "rising_005": bool(last - first >= 0.05), "sec": round(time.time() - t0)}
        del pr
        if dev.startswith("cuda"):
            torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--arms", default="k1,k2,k3"); ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=16); ap.add_argument("--k3_steps", type=int, default=2000); ap.add_argument("--pool_n", type=int, default=200000)
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
        r = k2(m, pool, d1, dev, a.chunk, a.bs); save("k2_oracle", {**meta, **r})
    if "k3" in arms:
        r = k3(m, pool, held, d1, dev, a.chunk, a.bs, steps=a.k3_steps); save("k3_probe_learn", {**meta, **r})


if __name__ == "__main__":
    main()
