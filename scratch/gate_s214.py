"""§214 D4-B local gates (D4_v1 §7 at 10187c1b; the scratch cell's list: G-P0, P2, P3, P5, P9, P10, P14 here; P6 / P7 / P8 / P11 on the host; P15 after landing).
  p0   factory / loader: pause "A" adds exactly one key (pause_vec, zeros); "none" adds none; VE.load strict on D3-T / D3-F / N1 / N2 (no pause key); a
       pause model's state_dict round-trips strictly through make_model(..., pause="A").
  p2   target map / R1 window fixtures (vb_loss internals on hand-built schedules): rho(t) = t - #pauses <= t; E_{rho(t)} (HOLD at every pause pass);
       pass n + P is an R (lastR); the R1 coda rows are exactly t in [n + P, K] per item; with pause_vec = 0 a paused rollout == the plain one (bitwise).
  p3   20k logged draws through the trainer's own draw logic (N.train_items(rng 1, bs 64, n <= 8, EXCL), rng_k, rng_p = seed + 21300, pause_schedule)
       vs exact_realised.json A_lastR (pass weights): P(F = 0 | t >= 9) 0.491 +- 0.01, = 0 at t >= 17; I(t; rho) 0.99 +- 0.03; I(t; M) 0.046 +- 0.005;
       item |corr(P, n)| <= 0.02; zero trailing pauses; P(n = 1) 0.017 +- 0.003.
  p5   plain path (--pause none): the pre-edit vbank (boxes/_vbank_pre_sd.py) vs boxes/vbank.py, pa / carry / slva, 3 steps, CPU: log + calibration bitwise.
  p9   pi only via rollout(pause=...): grep - "pause=" is passed only by vbank.vb_loss; no eval module passes it; input_edit never set in vbank.main.
  p10  gradflow on an augmented batch (SLVA, c 0, N1-free random init): d L / d pause_vec != 0; SLVA / core / R / coda grads nonzero; ast: vb_loss,
       rollout, pause_schedule carry no no_grad decorator.
  p14  slva.mass_stats on paused items uses E_{rho(t)} (a fixture: off-newest counts equal a hand-mapped reference).
    python scratch/gate_s214.py p2 --device cuda:1"""
import argparse
import ast
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from contextlib import nullcontext

sys.path.insert(0, "D:/Latent")
import numpy as np  # noqa: E402
import torch  # noqa: E402

import boxes.nested170 as N  # noqa: E402
import boxes.vbank as VB  # noqa: E402
import boxes.vbank_eval as VE  # noqa: E402
from boxes import s5  # noqa: E402

OUT = "D:/Latent/scratch/gate_s214"; RUNS = "boxes/runs_fsq"


@torch.no_grad()
def g_p0(dev):
    o = {}; ma = VB.make_model(s5.load_gpt2(dev), "slva", "train", "A"); mn = VB.make_model(s5.load_gpt2(dev), "slva", "train", "none")
    ka, kn = set(ma.state_dict()), set(mn.state_dict()); o["extra_keys_A"] = sorted(ka - kn); o["extra_keys_none_vs_A"] = sorted(kn - ka)
    o["pause_vec_zero"] = bool((ma.pause_vec == 0).all()) and tuple(ma.pause_vec.shape) == (768,)
    for c in ("s213_d3t_s1", "s213_d3f_s1", "s211_vb_carry_s1", "s211_vb_pa_s1"):
        m, a = VE.load(os.path.join(RUNS, c), dev); o["load_" + c] = [type(m).__name__, hasattr(m, "pause_vec")]; del m
    mb = VB.make_model(s5.load_gpt2(dev), "slva", "train", "A"); ma.pause_vec.data.fill_(0.5); mb.load_state_dict(ma.state_dict())
    o["roundtrip_strict"] = bool(torch.equal(mb.pause_vec, ma.pause_vec))
    o["pass"] = (o["extra_keys_A"] == ["pause_vec"] and o["extra_keys_none_vs_A"] == [] and o["pause_vec_zero"] and o["roundtrip_strict"]
                 and all(v == [("SLVA" if "d3" in k else "VBankN"), False] for k, v in o.items() if k.startswith("load_")))
    return o


def g_p2(dev):
    o = {}; torch.manual_seed(0); m = VB.make_model(s5.load_gpt2(dev), "slva", "train", "A").to(dev); m.c_anneal.fill_(0.0)
    TE = N.test_set(1000); its = [it for it in TE if it["n"] == 3][:2] + [it for it in TE if it["n"] == 5][:2]
    pz = [[True, False, True, False, False], None, [False, True, False, True, True, False, False, False], None]   # n 3 P 2; plain; n 5 P 3; plain
    assert all(z is None or (sum(1 for q in z if not q) == it["n"] and not z[-1]) for z, it in zip(pz, its))
    Ks = [7, 4, 10, 6]; x, pad, lens = VB.batch_x(its, dev); E = VB.rec_targets(its, max(Ks), x.shape[1], dev); ok_rho = True; ok_hold = True
    for b, z in enumerate(pz):
        if z is None:
            continue
        rho = 0
        for t in range(1, len(z) + 1):
            if not z[t - 1]:
                rho += 1
            ok_hold &= (not z[t - 1]) or rho == (t - sum(z[:t]))
        ok_rho &= rho == its[b]["n"]
    L, st, r = VB.vb_loss(m, its, Ks, dev, nullcontext, 1.0, keep=True, pz=pz)
    rows = defaultdict(list)
    for t, (rr, _) in r["coda"].items():
        for b in rr.tolist():
            rows[b].append(t)
    want = {b: list(range((its[b]["n"] + (sum(z) if z else 0)), Ks[b] + 1)) for b, z in enumerate(pz)}
    o["r1_windows"] = {str(b): sorted(rows[b]) for b in rows}; o["r1_windows_expected"] = {str(b): v for b, v in want.items()}
    o["r1_window_exact"] = all(sorted(rows[b]) == want[b] for b in want); o["rho_counts_n"] = bool(ok_rho); o["lastR"] = all(z is None or not z[-1] for z in pz)
    # E_{rho(t)} at a pause pass equals E at the previous pass (HOLD) - recompute the mapped E as vb_loss does and compare to a hand map
    Kb = max(Ks); hand = []
    for b, z in enumerate(pz):
        seq = [0]
        for t in range(1, Kb + 1):
            seq.append(seq[-1] + (0 if (z is not None and t <= len(z) and z[t - 1]) else 1))
        hand.append([min(v, Kb) for v in seq])
    Eh = torch.stack([E[b, hand[b]] for b in range(len(its))])
    pm = torch.zeros(len(its), Kb + 1, dtype=torch.bool, device=dev); rho_t = torch.arange(Kb + 1, device=dev)[None].repeat(len(its), 1)
    for b, z in enumerate(pz):
        if z is not None:
            pm[b, 1:len(z) + 1] = torch.tensor(z, device=dev); rho_t[b] = rho_t[b] - torch.cumsum(pm[b].long(), 0)
    Em = E.gather(1, rho_t.clamp(0, Kb)[:, :, None].expand(-1, -1, x.shape[1]))
    o["E_map_eq_hand"] = bool(torch.equal(Em, Eh)); o["hold_at_pause"] = all(torch.equal(Em[b, t], Em[b, t - 1]) for b, z in enumerate(pz) if z for t in range(1, len(z) + 1) if z[t - 1])
    with torch.no_grad():                                                   # pause_vec = 0: a paused rollout == the plain one, bitwise
        m.eval(); r0 = m.rollout(x, pad, Kb, coda_rows={Kb: torch.arange(len(its), device=dev)}); r1 = m.rollout(x, pad, Kb, coda_rows={Kb: torch.arange(len(its), device=dev)}, pause=pm)
        o["pi0_paused_eq_plain"] = bool(all(torch.equal(a_, b_) for a_, b_ in zip(r0["V"], r1["V"])) and torch.equal(r0["coda"][Kb][1], r1["coda"][Kb][1]))
        m.pause_vec.data.fill_(0.1); r2 = m.rollout(x, pad, Kb, coda_rows={Kb: torch.arange(len(its), device=dev)}, pause=pm)
        o["pi_moves_paused_only"] = bool(not torch.equal(r2["V"][Kb][0], r0["V"][Kb][0]) and torch.equal(r2["V"][Kb][1], r0["V"][Kb][1]))
    o["n_paused_logged"] = st["n_paused"]
    o["pass"] = o["r1_window_exact"] and o["rho_counts_n"] and o["lastR"] and o["E_map_eq_hand"] and o["hold_at_pause"] and o["pi0_paused_eq_plain"] and o["pi_moves_paused_only"] and o["n_paused_logged"] == 2
    return o


def g_p3(dev, n_items=20000, bs=64):
    """the trainer's draw logic, repeated in the same rng order (seed 1): items (rng), U (rng_k), coin / P / D / positions (rng_p)."""
    rng = np.random.default_rng(1); rng_k = np.random.default_rng(1 + 21100); rng_p = np.random.default_rng(1 + 21300)
    TE = N.test_set(1000); EXCL = {tuple(it["syms"]) for it in TE}; r_sp = np.random.default_rng(17300)
    for _ in range(2200):
        EXCL.add(tuple(N.render(N.generate(r_sp, 8)[0])))
    jR, jM, jF, unf, allp = defaultdict(float), defaultdict(float), defaultdict(float), defaultdict(float), defaultdict(float); Pn, trail, n1, cnt = [], 0, 0, 0
    recs = []                                                               # (aug, t, rho, M, F) per pass, for the exact-share reweighting
    while cnt < n_items:
        its = N.train_items(rng, bs, 8, EXCL); Ks = [min(it["n"] + int(rng_k.integers(0, 5)), 20) for it in its]; pz = None
        if rng_p.random() < 0.5:
            pz = [VB.pause_schedule(rng_p, it["n"], 8) for it in its]; Ks = [k_ + sum(z) for k_, z in zip(Ks, pz)]
        for b, it in enumerate(its):
            n = it["n"]; K = Ks[b]; z = pz[b] if pz else [False] * n; cnt += 1; n1 += n == 1
            if pz:
                Pn.append((sum(z), n)); trail += bool(z[-1])
            rho = 0
            for t in range(1, K + 1):
                M = int(t <= len(z) and z[t - 1]); F = int(rho >= n); jR[(t, rho)] += 1; jM[(t, M)] += 1; jF[(t, F)] += 1; allp[t] += 1; unf[t] += not F
                recs.append((pz is not None, t, rho, M, F))
                if not M and rho < n:
                    rho += 1
    def mi(j):
        tot = sum(j.values()); pa, pb = defaultdict(float), defaultdict(float)
        for (a_, b_), v in j.items():
            pa[a_] += v; pb[b_] += v
        return sum(v / tot * math.log2(v * tot / (pa[a_] * pb[b_])) for (a_, b_), v in j.items() if v > 0)
    ge = lambda d, ts: sum(v for t, v in d.items() if t >= ts)  # noqa: E731
    P_, n_ = np.array([p for p, _ in Pn]), np.array([n for _, n in Pn])
    o = {"items": cnt, "P_F0_t_ge9": ge(unf, 9) / ge(allp, 9), "P_F0_t_ge17": ge(unf, 17) / max(1, ge(allp, 17)), "I_t_rho": mi(jR), "I_t_M": mi(jM), "I_t_F": mi(jF),
         "corr_P_n": float(np.corrcoef(P_, n_)[0, 1]), "trailing_pauses": trail, "P_n1": n1 / cnt}
    # descriptive: the same draws reweighted to an exact 0.5 plain / augmented ITEM share (the per-step coin's batch clustering makes the realised share noisy)
    sh = np.mean([a_ for a_, *_ in recs]) if recs else 0.5; ia = sum(1 for z in [None] for _ in [0]); na_items = sum(1 for p_ in Pn); share_items = na_items / cnt
    wa, wp = 0.5 / share_items, 0.5 / (1 - share_items); jR2, jM2, unf2, all2 = defaultdict(float), defaultdict(float), defaultdict(float), defaultdict(float)
    for a_, t, rho, M, F in recs:
        w_ = wa if a_ else wp; jR2[(t, rho)] += w_; jM2[(t, M)] += w_; all2[t] += w_; unf2[t] += w_ * (not F)
    o["reweighted_to_share_0p5"] = {"aug_item_share_realised": share_items, "P_F0_t_ge9": ge(unf2, 9) / ge(all2, 9), "I_t_rho": mi(jR2), "I_t_M": mi(jM2)}
    ref = json.load(open("D:/Latent/docs/research_d4_2026-10-16/scripts/proofs_d4/exact_realised.json"))["A_lastR"]["pass"]
    o["ref"] = {"P_F0_t_ge9": ref["P_unfinished_input_given_t_ge"]["9"], "I_t_rho": ref["I_t_rho"], "I_t_M": ref["I_t_M"]}
    o["pass"] = (abs(o["P_F0_t_ge9"] - 0.491) <= 0.01 and o["P_F0_t_ge17"] == 0.0 and abs(o["I_t_rho"] - 0.99) <= 0.03 and abs(o["I_t_M"] - 0.046) <= 0.005
                 and abs(o["corr_P_n"]) <= 0.02 and trail == 0 and abs(o["P_n1"] - 0.017) <= 0.003)
    return o


def g_p5(dev):
    o = {}; base = "D:/Latent/scratch/_tmp/s214_plain"
    for mode in ("pa", "carry", "slva"):
        logs = {}
        for tag, mod in (("pre", "boxes._vbank_pre_sd"), ("new", "boxes.vbank")):
            out = "%s/%s_%s" % (base, mode, tag); os.makedirs(out, exist_ok=True)
            if os.path.exists(os.path.join(out, "log.jsonl")):
                os.remove(os.path.join(out, "log.jsonl"))
            cmd = [sys.executable, "-m", mod, "--mode", mode, "--steps", "48000", "--stop_at", "3", "--log_every", "1", "--bs", "16", "--micro", "1", "--device", "cpu", "--out", out]
            r = subprocess.run(cmd, cwd="D:/Latent", capture_output=True, text=True, env={**os.environ, "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}); assert r.returncode == 0, r.stderr[-2000:]
            logs[tag] = ([json.loads(l_) for l_ in open(os.path.join(out, "log.jsonl"))], json.load(open(os.path.join(out, "calibration.json"))))
        keys = ("loss", "answer_ce", "rec_ce", "gn_preclip", "c", "carry_mass", "slva_gn", "slva_offnewest_chg", "slva_offnewest_copy")
        o[mode] = {"log_bitwise": [[r_.get(k) for k in keys] for r_ in logs["pre"][0]] == [[r_.get(k) for k in keys] for r_ in logs["new"][0]],
                   "calibration_equal": logs["pre"][1] == logs["new"][1], "n_log": len(logs["new"][0])}
    o["pass"] = all(o[md]["log_bitwise"] and o[md]["calibration_equal"] and o[md]["n_log"] == 3 for md in ("pa", "carry", "slva"))
    return o


def g_p9(dev):
    o = {}; hits = {}
    for f in ("boxes/vbank.py", "boxes/vbank_eval.py", "boxes/vbank_diag.py", "boxes/vbank_halt.py", "boxes/slva.py"):
        src = open(os.path.join("D:/Latent", f), encoding="utf-8").read()
        hits[f] = [ln.strip()[:120] for ln in src.splitlines() if "pause=" in ln and "rollout(" in ln and not ln.strip().startswith("def ")]
    o["rollout_pause_callers"] = hits
    main_src = open("D:/Latent/boxes/vbank.py", encoding="utf-8").read().split("def main():")[1]
    o["input_edit_set_in_main"] = "input_edit =" in main_src or "input_edit=" in main_src
    o["pass"] = (len(hits["boxes/vbank.py"]) == 1 and "pause=pmask" in hits["boxes/vbank.py"][0] and all(not v for k, v in hits.items() if k != "boxes/vbank.py")
                 and not o["input_edit_set_in_main"])
    return o


def g_p10(dev):
    o = {}; torch.manual_seed(0); m = VB.make_model(s5.load_gpt2(dev), "slva", "train", "A").to(dev); m.c_anneal.fill_(0.0); m.train()
    its = [it for it in N.test_set(1000) if 3 <= it["n"] <= 6][:8]; rng_p = np.random.default_rng(5); pz = [VB.pause_schedule(rng_p, it["n"], 8) for it in its]
    Ks = [it["n"] + sum(z) + 2 for it, z in zip(its, pz)]
    L = VB.vb_loss(m, its, Ks, dev, nullcontext, 1.0, pz=pz)[0]; L.backward()
    g = lambda nm: float(dict(m.named_parameters())[nm].grad.norm()) if dict(m.named_parameters())[nm].grad is not None else None  # noqa: E731
    o["grad"] = {k: g(k) for k in ("pause_vec", "slva_wq", "slva_wk", "rec.weight", "g.p.h_7_attn_c_attn_weight", "g.p.h_10_attn_c_attn_weight", "beta")}
    tree = ast.parse(open("D:/Latent/boxes/vbank.py", encoding="utf-8").read()); dec = {}
    for nd in ast.walk(tree):
        if isinstance(nd, ast.FunctionDef) and nd.name in ("vb_loss", "rollout", "pause_schedule"):
            dec[nd.name] = [ast.unparse(d) for d in nd.decorator_list]
    o["decorators"] = dec
    o["pass"] = all(v is not None and v > 0 and math.isfinite(v) for v in o["grad"].values()) and all(v == [] for v in dec.values())
    return o


@torch.no_grad()
def g_p14(dev):
    from boxes.slva import mass_stats
    m = VB.make_model(s5.load_gpt2(dev), "slva", "train", "A").to(dev); m.c_anneal.fill_(0.0); m.eval()
    its = [it for it in N.test_set(1000) if it["n"] == 4][:3]; pz = [[False, True, False, True, False, False], None, [True, False, False, False, False]]
    Ks = [8, 6, 7]; x, pad, lens = VB.batch_x(its, dev); Kb = max(Ks); pm = torch.zeros(3, Kb + 1, dtype=torch.bool, device=dev)
    for b, z in enumerate(pz):
        if z:
            pm[b, 1:len(z) + 1] = torch.tensor(z, device=dev)
    m.slva_log = []; m.rollout(x, pad, Kb, pause=pm); logs = list(m.slva_log); got = mass_stats(m, its, Ks, dev, pz=pz)
    E = VB.rec_targets(its, Kb, x.shape[1], dev); ref = {"chg": [0.0, 0], "copy": [0.0, 0]}; Kt = torch.tensor(Ks, device=dev)
    for t, nw in logs:
        for b in range(3):
            z = pz[b]; rho = lambda tt: tt - (sum(z[:tt]) if z else 0)  # noqa: E731
            Et, Ep = E[b, min(rho(t), Kb)], E[b, min(rho(t - 1), Kb)]; live = pad[b] & (t <= Kt[b]); off = 1.0 - nw[b].float()
            for k_, sel in (("chg", (Et != Ep) & live), ("copy", (Et == Ep) & (Et != VB.VOID) & live)):
                ref[k_][0] += float(off[sel].sum()); ref[k_][1] += int(sel.sum())
    o = {"got": got, "ref": ref}
    o["pass"] = got["chg"][1] == ref["chg"][1] and got["copy"][1] == ref["copy"][1] and abs(got["chg"][0] - ref["chg"][0]) < 1e-3 and abs(got["copy"][0] - ref["copy"][0]) < 1e-3
    return o


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("gate"); ap.add_argument("--device", default="cuda:1"); a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False; os.makedirs(OUT, exist_ok=True); os.chdir("D:/Latent")
    r = {"p0": g_p0, "p2": g_p2, "p3": g_p3, "p5": g_p5, "p9": g_p9, "p10": g_p10, "p14": g_p14}[a.gate](a.device)
    r["device"] = torch.cuda.get_device_name(a.device) if a.device.startswith("cuda") else "cpu"
    json.dump(r, open(os.path.join(OUT, "%s.json" % a.gate), "w", newline="\n"), indent=1); print(a.gate, "PASS" if r["pass"] else "FAIL", json.dumps(r)[:1400], flush=True)


if __name__ == "__main__":
    main()
