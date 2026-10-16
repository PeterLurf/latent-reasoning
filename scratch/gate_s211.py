"""§211 VERSION-BANK Stage 0 gates (Theory pin 6: diff E_t against the red team's reference). -> scratch/gate_s211/<gate>.json
    python scratch/gate_s211.py et      # boxes.vbank.recode_targets vs scratch/s211_ref_slot_targets.slot_targets on 100k items, n 1-16"""
import json
import os
import sys

sys.path.insert(0, "D:/Latent"); sys.path.insert(0, "D:/Latent/scratch"); os.chdir("D:/Latent")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F_  # noqa: E402

import boxes.nested170 as N  # noqa: E402
import boxes.vbank as VB  # noqa: E402


def gate_et(n_items=100_000, seed=20902, n_max=16):
    import s211_ref_slot_targets as R
    rng = np.random.default_rng(seed); bad = 0; vers = 0; first = None
    for i in range(n_items):
        root, _ = N.generate(rng, int(rng.integers(1, n_max + 1))); ref = R.slot_targets(root)
        mine = [["V" if c == VB.VOID else VB.CLS[c] for c in e] for e in VB.recode_targets(root)]
        vers += len(ref)
        if mine != ref:
            bad += 1; first = first if first is not None else i
    return {"items": n_items, "seed": seed, "n_range": [1, n_max], "versions": vers, "diffs": bad, "first_diff_item": first, "pass": bad == 0,
            "reference": "scratch/s211_ref_slot_targets.py (red team, sha256 ebf25d3b... of the source file)"}



# ---- Part 3 gates (VBankN): G-FP function preservation, c = 1e4 == CARRY bitwise, masked-K (batched/padded == per item), G1-cal + ratio test
def _items(n_items, seed, n_lo=1, n_hi=8):
    rng = np.random.default_rng(seed); out = []
    for _ in range(n_items):
        r, y = N.generate(rng, int(rng.integers(n_lo, n_hi + 1))); out.append({"syms": N.render(r), "t": y, "root": r, "n": None})
    return out


def _model(mode, dt, dev="cpu", seed=21100):
    from boxes import s5
    torch.manual_seed(seed); g = s5.load_gpt2(dev); m = VB.VBankN(g, mode).to(dev)
    return m.double() if dt == torch.float64 else m


def gate_fp(dev="cpu"):
    """draft §5.5: CARRY (and PA at c = 1e4) at K = 1 with the bank masked == GPT-2's logits at ' =' (fp64, <= 1e-9 rel)."""
    out = {}
    for mode in ("carry", "pa"):
        m = _model(mode, torch.float64, dev); m.bank_off = True; its = _items(6, 21101); x, pad, lens = VB.batch_x(its, dev)
        with torch.no_grad():
            r = m.rollout(x, pad, 1, coda_rows={1: torch.arange(len(its))}); z = m.answer_logits(r["coda"][1][1], lens)
            h, _ = m.g(m.g.embed(x)); zr = m.g.logits(h[torch.arange(len(its)), lens - 1])
        out[mode] = float((z - zr).abs().max() / zr.abs().max())
    out["pass"] = all(v <= 1e-9 for v in out.values()); return out


def gate_carry_bitwise(dev="cpu"):
    """pin 1: PURE-ANNEAL at c = 1e4 == CARRY bitwise (every stored version V_1..V_12 and the coda logits at every t), bank ON (beta -5, ALiBi gamma),
    fp32 and fp64; plus c = 0 differs (the gather is live)."""
    out = {}
    for dt in (torch.float32, torch.float64):
        m = _model("pa", dt, dev); its = _items(6, 21102); x, pad, lens = VB.batch_x(its, dev); rows = {t: torch.arange(len(its)) for t in range(13)}
        with torch.no_grad():
            m.c_anneal.fill_(1e4); ra = m.rollout(x, pad, 12, coda_rows=rows)
            m.mode = "carry"; rc = m.rollout(x, pad, 12, coda_rows=rows); m.mode = "pa"
            m.c_anneal.fill_(0.0); r0 = m.rollout(x, pad, 12, coda_rows=rows); m.c_anneal.fill_(1e4)
        eqV = all(torch.equal(a[pad], c[pad]) for a, c in zip(ra["V"], rc["V"])); eqZ = all(torch.equal(ra["coda"][t][1][pad], rc["coda"][t][1][pad]) for t in rows)
        d0 = max(float((a[pad] - c[pad]).abs().max()) for a, c in zip(r0["V"][2:], rc["V"][2:]))
        out[str(dt)] = {"V_bitwise": eqV, "coda_bitwise": eqZ, "c0_maxdiff_V2plus": d0}
    out["pass"] = all(v["V_bitwise"] and v["coda_bitwise"] and v["c0_maxdiff_V2plus"] > 1e-6 for v in out.values()); return out


def gate_maskk(dev="cpu"):
    """draft §9 item 5: the batched, right-padded path (per-row coda subsets) == each item alone, unpadded (fp64, real slots; V_1..V_8 and coda logits
    at t in its window), both modes, bank ON with c = 0 (PA) so padding enters every softmax."""
    out = {}
    for mode in ("carry", "pa"):
        m = _model(mode, torch.float64, dev); m.c_anneal.fill_(0.0); its = _items(5, 21103); x, pad, lens = VB.batch_x(its, dev)
        rows = {t: torch.tensor([i for i in range(len(its)) if (i + t) % 2 == 0]) for t in range(1, 9)}; worst = 0.0
        with torch.no_grad():
            rb = m.rollout(x, pad, 8, coda_rows=rows)
            for i in range(len(its)):
                xi, pi, li = VB.batch_x([its[i]], dev); ri = m.rollout(xi, pi, 8, coda_rows={t: torch.tensor([0]) for t in range(1, 9)}); L_ = int(li[0])
                for t in range(9):
                    worst = max(worst, float((rb["V"][t][i, :L_] - ri["V"][t][0, :L_]).abs().max() / ri["V"][t][0, :L_].abs().max()))
                for t, rr in rows.items():
                    if i in rr.tolist():
                        k_ = rr.tolist().index(i); za = m.answer_logits(rb["coda"][t][1][k_:k_ + 1], lens[i:i + 1]); zb = m.answer_logits(ri["coda"][t][1], li)
                        worst = max(worst, float((za - zb).abs().max() / zb.abs().max()))
        out[mode] = worst
    out["pass"] = all(v <= 1e-9 for k, v in out.items()); return out


def gate_cal(dev="cuda:0"):
    """pin 3 G1-cal: calibrate beta0 then beta on 64 items n <= 8, K = 12 (fp32), then the ratio test."""
    m = _model("pa", torch.float32, dev); its = _items(64, 21104); x, pad, lens = VB.batch_x(its, dev)
    rep = m.calibrate(x, pad, lens); rt = m.ratio_test(x, pad, lens)
    return {"calibration": rep, "ratio_test": rt, "pass": rep["max_m99"] <= 0.02 + 1e-6 and rt["pass"]}



def _train_items(n_items, seed, n_lo, n_hi):
    its = _items(n_items, seed, n_lo, n_hi)
    for it in its:
        it["n"] = sum(1 for x_ in it["syms"] if x_.strip() in N.OPS)
    return its


def gate_gradflow(dev="cuda:0"):
    """REDTEAM F9a / §211 gates: (1) every parameter group gets a nonzero gradient from vb_loss (PA at c = 0 and c = 1e4, CARRY; MCT on for one PA run);
    (2) dL / d(bank K, V) != 0 for every bank-reading layer (versions >= 1); (3) with the carry / gather input stop-gradiented (sg_input) and rec_w 0,
    dL_ans / dV_1 != 0 on items with n >= 3 (their answer windows start at t >= 3, so V_1 reaches L_ans only through bank reads). fp32."""
    ac = lambda: torch.autocast("cuda", enabled=False)  # noqa: E731
    its = _train_items(8, 21105, 3, 5); Ks = [min(it["n"] + 2, 12) for it in its]; out = {}
    for tag, mode, c, tf in (("pa_c0", "pa", 0.0, 0.0), ("pa_c1e4", "pa", 1e4, 0.0), ("carry", "carry", 1e4, 0.0), ("pa_c0_mct", "pa", 0.0, 0.5)):
        m = _model(mode, torch.float32, dev); m.c_anneal.fill_(c); m.zero_grad(set_to_none=True)
        L, st, r = VB.vb_loss(m, its, Ks, dev, ac, 1.0, tf, np.random.default_rng(0), keep=True)
        bank = {i: [t_ for s_ in range(1, len(r["kv"][i])) for t_ in r["kv"][i][s_]] for i in VB.LAYERS}
        gb = torch.autograd.grad(L, [t_ for i in VB.LAYERS for t_ in bank[i]], retain_graph=True, allow_unused=True)
        gbn, k_ = {}, 0
        for i in VB.LAYERS:
            n_ = len(bank[i]); gbn[str(i)] = float(sum(float(g_.norm()) for g_ in gb[k_:k_ + n_] if g_ is not None)); k_ += n_
        L.backward(); zero = []
        for nm, p_ in m.named_parameters():
            if p_.grad is None or float(p_.grad.norm()) == 0:
                zero.append(nm)
        if tf == 0 and mode == "pa":
            zb0 = m.beta0.grad[4:].abs().sum().item() if m.beta0.grad is not None else 0.0
        out[tag] = {"zero_grad_params": zero, "bank_kv_grad_norm": gbn, "answer_ce": st["answer_ce"], "rec_ce": st["rec_ce"],
                    "beta0_core_grad": float(m.beta0.grad[:4].norm()), "beta0_coda_grad": float(m.beta0.grad[4:].norm()), "beta_grad": float(m.beta.grad.norm()),
                    "gam_grad": float(m.gam.grad.norm()), **({"tf_frac": st["tf_n"] / max(1, st["tf_tot"])} if tf else {})}
        m2 = _model(mode, torch.float32, dev); m2.c_anneal.fill_(c)
        L2, _, r2 = VB.vb_loss(m2, its, Ks, dev, ac, 0.0, 0.0, None, sg_input=True, keep=True)
        out[tag]["sg_dLans_dV1"] = float(torch.autograd.grad(L2, r2["V"][1])[0].norm())
    ok = True
    for tag, v in out.items():
        allowed = {"gq_w", "gq_b", "gk_w", "gk_b", "gln_w", "gln_b", "gamma_q"} if tag == "pa_c1e4" else set()   # c = 1e4: a one-hot softmax, d/dlogits = 0 exactly
        ok = ok and not [z for z in v["zero_grad_params"] if z not in allowed] and all(x_ > 0 for x_ in v["bank_kv_grad_norm"].values())
        ok = ok and v["sg_dLans_dV1"] > 0 and v["beta0_core_grad"] > 0 and v["beta_grad"] > 0 and v["gam_grad"] > 0
    out["pass"] = bool(ok); return out


def gate_mct(dev="cuda:0"):
    """MCT W1 / W3 (R3; pin 7): W1, R(MCT(V_s)) decodes E_s at every real slot (<= 3 steps), on V_1..V_8 of 32 items (n 3..8, init model, fp32);
    the share of slots needing > 1 step and ||dV|| / ||V|| per step logged. W3: an edit of V_s along R's row space (rec.weight rows, per slot, norm 0.1 rms)
    vs one of equal norm in its null space: the relative change of V_{s+1} for each (row-space change > 0 is the requirement; the ratio is reported)."""
    m = _model("pa", torch.float32, dev); its = _train_items(32, 21106, 3, 8); x, pad, lens = VB.batch_x(its, dev); E = VB.rec_targets(its, 8, x.shape[1], dev)
    out = {"W1": {}, "W3": {}}
    with torch.no_grad():
        r = m.rollout(x, pad, 8)
        for s_ in range(1, 9):
            Vm, ms = m.mct(r["V"][s_], E[:, s_], pad); dec = m.rec_logits(Vm).argmax(-1); ok_ = ((dec == E[:, s_].clamp_min(0)) | ~pad).all(1)
            need = ms["steps"][pad]; out["W1"][str(s_)] = {"all_slots_ok_items": float(ok_.float().mean()), "slot_acc": float((dec == E[:, s_])[pad].float().mean()),
                                                          "frac_multi_step": float((need > 1).float().sum() / max(1, int((need >= 1).sum()))), "rel": ms["rel"]}
        W = m.rec.weight.float(); P_row = torch.linalg.pinv(W) @ W; gen = torch.Generator(device=dev).manual_seed(21107)
        for s_ in (2, 4, 6):
            z = torch.randn(r["V"][s_].shape, device=dev, generator=gen); dr = z @ P_row.T; dn = z - dr
            sc = 0.1 * r["V"][s_].pow(2).mean(-1, keepdim=True).sqrt()
            dr = dr / dr.norm(dim=-1, keepdim=True) * sc * 768 ** 0.5; dn = dn / dn.norm(dim=-1, keepdim=True) * sc * 768 ** 0.5
            ch = {}
            for nm, d_ in (("row", dr), ("null", dn)):
                rr = m.rollout(x, pad, s_ + 1, edit=lambda t, V, d_=d_, s_=s_: V + d_ * pad[..., None] if t == s_ else None)
                ch[nm] = float((rr["V"][s_ + 1] - r["V"][s_ + 1])[pad].norm() / r["V"][s_ + 1][pad].norm())
            out["W3"][str(s_)] = {**ch, "row_over_null": ch["row"] / ch["null"] if ch["null"] > 0 else None}
    out["pass"] = all(v["all_slots_ok_items"] == 1.0 for v in out["W1"].values()) and all(v["row"] > 1e-4 for v in out["W3"].values()); return out



def gate_g1b(dev="cuda:0", runs=("s211_g1b_pa", "s211_g1b_carry", "s211_g1b_pamct"), steps=(0, 1000, 2000)):
    """G1b (REDTEAM F5; draft §9 item 3) on the 2k-step smokes' kept models: 256 items (n 1..8, seed 21108), fp32. Per kept step: P(number) at ' =' from
    the coda after pass t (t 0..8), cos(V_t, V_{t-1}) over real slots (t 1..8), answer CE at K' = n (pooled n 1..8) and at K' = 1 on n = 1.
    drift = P(num | K' = 1) - P(num | K' = 8). PASS per run: drift(2k) < drift(0) and CE(K' = n) falls 0 -> 1k -> 2k."""
    import json as _j
    from boxes import s5
    its = _train_items(256, 21108, 1, 8); x, pad, lens = VB.batch_x(its, dev); y = torch.tensor(N.NUM_IDS[[it["t"] for it in its]], device=dev)
    nn_ = torch.tensor([it["n"] for it in its], device=dev); NUM = torch.tensor(N.NUM_IDS, device=dev); out = {}
    for run in runs:
        d = os.path.join("boxes/runs_fsq", run); a = _j.load(open(os.path.join(d, "args.json"))); g = s5.load_gpt2(dev); m = VB.VBankN(g, a["mode"]).to(dev); m.eval(); o = {}
        for st in steps:
            m.load_state_dict(torch.load(os.path.join(d, "ckpt_model_%d.pt" % st), map_location=dev, weights_only=False)["model"])
            with torch.no_grad():                                           # 64-item chunks (fp32 on 8 GB)
                pnl, cel, csl = {t: [] for t in range(9)}, {t: [] for t in range(9)}, {t: [] for t in range(1, 9)}
                for c0 in range(0, len(its), 64):
                    sl = slice(c0, c0 + 64); xc, pc, lc = VB.batch_x(its[sl], dev)
                    r = m.rollout(xc, pc, 8, coda_rows={t: torch.arange(xc.shape[0], device=dev) for t in range(9)})
                    for t in range(9):
                        lg = m.answer_logits(r["coda"][t][1], lc); pnl[t].append(torch.softmax(lg, -1)[:, NUM].sum(-1)); cel[t].append(F_.cross_entropy(lg, y[sl], reduction="none"))
                    for t in range(1, 9):
                        csl[t].append(F_.cosine_similarity(r["V"][t], r["V"][t - 1], dim=-1)[pc])
                pn = {str(t): float(torch.cat(pnl[t]).mean()) for t in range(9)}; ce_t = {t: torch.cat(cel[t]) for t in range(9)}
                cos = {str(t): float(torch.cat(csl[t]).mean()) for t in range(1, 9)}
                ce_kn = float(torch.stack([ce_t[int(n)][i] for i, n in enumerate(nn_.tolist())]).mean()); ce_k1_n1 = float(ce_t[1][nn_ == 1].mean())
            o[str(st)] = {"p_num_by_pass": pn, "cos_by_pass": cos, "ce_kn": ce_kn, "ce_k1_n1": ce_k1_n1, "drift": pn["1"] - pn["8"], "c": float(m.c_anneal)}
        o["pass"] = o[str(steps[-1])]["drift"] < o[str(steps[0])]["drift"] and o[str(steps[2])]["ce_kn"] < o[str(steps[1])]["ce_kn"] < o[str(steps[0])]["ce_kn"]
        out[run] = o
    out["pass"] = all(v["pass"] for k, v in out.items() if k != "pass"); return out



def gate_verkeep(dev="cpu"):
    """note 211-14 (eval-only ver_keep hook): (a) ver_keep = all-True == ver_keep None bitwise (fp32 and fp64, PA c 0 and CARRY, 10 passes, every V_t and
    coda output); (b) BANK-ABLATE (versions 1..t-2 masked) runs finite and differs from intact at t >= 3."""
    out = {}
    for dt in (torch.float32, torch.float64):
        for mode in ("pa", "carry"):
            m = _model(mode, dt, dev); m.c_anneal.fill_(0.0); its = _items(5, 21109, 3, 8); x, pad, lens = VB.batch_x(its, dev); rows = {t: torch.arange(len(its)) for t in range(11)}
            with torch.no_grad():
                r0 = m.rollout(x, pad, 10, coda_rows=rows)
                m.ver_keep = lambda t, nv: torch.ones(nv, dtype=torch.bool); r1 = m.rollout(x, pad, 10, coda_rows=rows)
                m.ver_keep = lambda t, nv: (torch.arange(nv) == 0) | (torch.arange(nv) >= t - 1); r2 = m.rollout(x, pad, 10, coda_rows=rows); m.ver_keep = None
            eq = all(torch.equal(a[pad], b[pad]) for a, b in zip(r0["V"], r1["V"])) and all(torch.equal(r0["coda"][t][1][pad], r1["coda"][t][1][pad]) for t in rows)
            fin = all(bool(torch.isfinite(v).all()) for v in r2["V"]); dif = float((r2["V"][10] - r0["V"][10])[pad].abs().max())
            out["%s_%s" % (mode, str(dt)[6:])] = {"all_true_bitwise": eq, "ablate_finite": fin, "ablate_maxdiff_V10": dif}
    out["pass"] = all(v["all_true_bitwise"] and v["ablate_finite"] and v["ablate_maxdiff_V10"] > 0 for v in out.values()); return out



def gate_hooks2(dev="cpu"):
    """note 212-3 eval-only hooks: (b) ver_sub = the item's own stored V_1..V_o == intact bitwise; (c) prefix = [] == intact; (d) age_floor 2 changes V_t
    (t >= 3) and stays finite; fp32 + fp64, PA (c 0) + CARRY, 10 passes."""
    out = {}
    for dt in (torch.float32, torch.float64):
        for mode in ("pa", "carry"):
            m = _model(mode, dt, dev); m.c_anneal.fill_(0.0); its = _items(5, 21110, 3, 8); x, pad, lens = VB.batch_x(its, dev); rows = {t: torch.arange(len(its)) for t in range(11)}
            with torch.no_grad():
                r0 = m.rollout(x, pad, 10, coda_rows=rows); rs = m.rollout(x, pad, 10, coda_rows=rows, ver_sub=[r0["S"][s_] for s_ in (1, 2, 3)])
                rp = m.rollout(x, pad, 10, coda_rows=rows, prefix=[]); m.age_floor = 2.0; rf = m.rollout(x, pad, 10, coda_rows=rows); m.age_floor = None
            eq = lambda a, b: all(torch.equal(u[pad], v[pad]) for u, v in zip(a["V"], b["V"])) and all(torch.equal(a["coda"][t][1][pad], b["coda"][t][1][pad]) for t in rows)  # noqa: E731
            out["%s_%s" % (mode, str(dt)[6:])] = {"ver_sub_self_bitwise": eq(r0, rs), "prefix_empty_bitwise": eq(r0, rp),
                                                  "floor_finite": all(bool(torch.isfinite(v).all()) for v in rf["V"]), "floor_maxdiff_V10": float((rf["V"][10] - r0["V"][10])[pad].abs().max())}
    out["pass"] = all(v["ver_sub_self_bitwise"] and v["prefix_empty_bitwise"] and v["floor_finite"] and v["floor_maxdiff_V10"] > 0 for v in out.values()); return out


if __name__ == "__main__":
    g = sys.argv[1]; r = {"et": gate_et, "fp": gate_fp, "carry_bitwise": gate_carry_bitwise, "maskk": gate_maskk, "cal": gate_cal, "gradflow": gate_gradflow, "mct": gate_mct, "g1b": gate_g1b, "verkeep": gate_verkeep, "hooks2": gate_hooks2}[g](); json.dump(r, open("scratch/gate_s211/%s.json" % g, "w", newline="\n"), indent=1); print(json.dumps(r))
