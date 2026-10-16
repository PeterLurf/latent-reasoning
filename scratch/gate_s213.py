"""§213 D3 implementation gates (D3_v1.md §6 S0-4), boxes/slva.py + the vbank trainer / loader edits. -> scratch/gate_s213/<gate>.json
  d0    arch / loader: SLVA param names slva_* (no PA gather params), mode "slva", anneals True; make_model types; N2's model.pt into SLVA strict -> fails;
        N1's model.pt into SLVA strict=False -> missing = slva_* only, unexpected = none; vbank_eval.load(N2) still VBankN (arch absent).
  d1    CARRY identity: N1's weights in SLVA and in VBankN carry; c 1e4: rollout K 12 bitwise (fp32: every V_t + coda logits at K) and <= 1e-9 rel (fp64);
        t = 1 bitwise at c in {1e4, 20, 0}; both table arms (the table only matters in training).
  d2    padding: c 0, fp64: items alone vs in a padded batch, every real slot of V_K + answer logits <= 1e-9 rel; no NaN anywhere (padded slots included).
  d5    hook parity at c 1e4 (SLVA vs CARRY, fp32 bitwise): ver_keep (vbank_eval.ablate_keep), v0_off_after 2, FOREIGN ver_sub, FOREIGN prefix (vs CARRY
        whose first pass input is N(F_o): INSERT keeps CARRY's carry at V_0 by design, SLVA / PA read F_o as the newest), age_floor 0.1,
        slot_mask (vowel slots {3, 5}), input_edit; SLVA-only hooks run finite: gather_splice 8, gather_log, gather_split_log (cross 0), gather_hist_ablate
        self / other; gather_qmod asserts. c 0: masked versions' SLVA weight exactly 0 (keep / off_v0 / self ablate) and > 0 unmasked.
  d6    gradflow at c 0 (N1 weights + SLVA init): R1-only (rec_w 0) and R1 + R2 -> every slva_* param grad nonzero and finite, R2 changes them; at c 1e4
        the slva_* grads are exactly 0 (Thm 1(b)); ast: SLVA.gather / gkey carry no decorator.
  plain pa / carry trainer: the pre-edit vbank (boxes/_vbank_pre_d3.py, 400b98ea + 27245354 hooks, before the §213 trainer edits) vs boxes/vbank.py,
        --stop_at 3 --log_every 1 --bs 16 on the same card: log.jsonl loss / answer_ce / rec_ce / gn_preclip and calibration.json bitwise (subprocess).
  g1b   the 2k-step smokes' kept models (s213_g1b_T / _F): drift(2k) < drift(0), CE(K' = n) falls 0 -> 1k -> 2k (gate_s211 g1b's definition).
    python scratch/gate_s213.py d1 --device cuda:1"""
import argparse
import ast
import json
import math
import os
import subprocess
import sys

sys.path.insert(0, "D:/Latent")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import boxes.nested170 as N  # noqa: E402
import boxes.vbank as VB  # noqa: E402
import boxes.vbank_eval as VE  # noqa: E402
from boxes import s5  # noqa: E402
from boxes.slva import SLVA  # noqa: E402

N1 = "boxes/runs_fsq/s211_vb_carry_s1"; N2 = "boxes/runs_fsq/s211_vb_pa_s1"; OUT = "D:/Latent/scratch/gate_s213"


def pair(dev, dtype=torch.float32, table="train"):
    """(SLVA, VBankN carry) sharing N1's trained weights (SLVA's slva_* at init)."""
    sd = torch.load(os.path.join(N1, "model.pt"), map_location=dev, weights_only=True)
    c = VB.VBankN(s5.load_gpt2(dev), "carry").to(dev); c.load_state_dict(sd)
    s = SLVA(s5.load_gpt2(dev), table).to(dev); mk = s.load_state_dict(sd, strict=False)
    assert all(k.startswith("slva_") for k in mk.missing_keys) and not mk.unexpected_keys, mk
    for m in (c, s):
        m.eval(); m.to(dtype)
    return s, c


def items(n_list, k, seed=21300):
    TE = N.test_set(1000); r = np.random.default_rng(seed); out = []
    for n in n_list:
        pool = [it for it in TE if it["n"] == n]; out += [pool[i] for i in r.permutation(len(pool))[:k]]
    return out


def rel(a, b):
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-30))


@torch.no_grad()
def g_d0(dev):
    o = {}; s = VB.make_model(s5.load_gpt2(dev), "slva", "train"); c = VB.make_model(s5.load_gpt2(dev), "carry"); p = VB.make_model(s5.load_gpt2(dev), "pa")
    nm = [n for n, _ in s.named_parameters()]
    o["types"] = [type(s).__name__, type(c).__name__, type(p).__name__]; o["slva_names"] = sorted({n.split(".")[0] for n in nm if n.startswith("slva_")})
    o["no_pa_gather_params"] = not any(n.split(".")[0] in ("gq_w", "gq_b", "gk_w", "gk_b", "gln_w", "gln_b", "gamma_q") for n in nm)
    o["mode_anneals"] = [s.mode, s.anneals, c.anneals, p.anneals]
    try:
        s.load_state_dict(torch.load(os.path.join(N2, "model.pt"), map_location=dev, weights_only=True)); o["n2_into_slva_strict_fails"] = False
    except RuntimeError as e:
        o["n2_into_slva_strict_fails"] = True; o["n2_err_head"] = str(e)[:160]
    try:
        VB.make_model(s5.load_gpt2(dev), "carry", "frozen"); o["frozen_needs_slva"] = False
    except AssertionError:
        o["frozen_needs_slva"] = True
    m2, a2 = VE.load(N2, dev); o["ve_load_n2"] = type(m2).__name__
    o["pass"] = (o["types"] == ["SLVA", "VBankN", "VBankN"] and len(o["slva_names"]) == 8 and o["no_pa_gather_params"] and o["mode_anneals"] == ["slva", True, False, True]
                 and o["n2_into_slva_strict_fails"] and o["frozen_needs_slva"] and o["ve_load_n2"] == "VBankN")
    return o


@torch.no_grad()
def g_d1(dev):
    o = {}; its = items([3, 8, 12], 4); K = 12
    for table in ("train", "frozen"):
        for dt, tag in ((torch.float32, "fp32"), (torch.float64, "fp64")):
            s, c = pair(dev, dt, table); x, pad, lens = VB.batch_x(its, dev); rows = torch.arange(len(its), device=dev)
            s.c_anneal.fill_(1e4); rs = s.rollout(x, pad, K, coda_rows={K: rows}); rc = c.rollout(x, pad, K, coda_rows={K: rows})
            ls, lc = s.answer_logits(rs["coda"][K][1], lens), c.answer_logits(rc["coda"][K][1], lens)
            if tag == "fp32":
                o["%s_fp32_bitwise_all_V_and_logits" % table] = bool(all(torch.equal(a, b) for a, b in zip(rs["V"], rc["V"])) and torch.equal(ls, lc))
            else:
                o["%s_fp64_rel_max" % table] = max(max(rel(a, b) for a, b in zip(rs["V"], rc["V"])), rel(ls, lc))
            for cv in (1e4, 20.0, 0.0):
                s.c_anneal.fill_(cv); r1 = s.rollout(x, pad, 1, coda_rows={1: rows}); q1 = c.rollout(x, pad, 1, coda_rows={1: rows})
                o["%s_%s_t1_bitwise_c%g" % (table, tag, cv)] = bool(torch.equal(r1["V"][1], q1["V"][1]) and torch.equal(r1["coda"][1][1], q1["coda"][1][1]))
            del s, c, rs, rc, r1, q1; torch.cuda.empty_cache()
    o["pass"] = all(v for k, v in o.items() if "bitwise" in k) and all(v <= 1e-9 for k, v in o.items() if k.endswith("fp64_rel_max"))
    return o


@torch.no_grad()
def g_d2(dev):
    s, _ = pair(dev, torch.float64); s.c_anneal.fill_(0.0); s.slva_gamma.data.normal_(0, 0.3, generator=torch.Generator(device=dev).manual_seed(1)) if False else None
    torch.manual_seed(21301)
    for p_ in (s.slva_beta0, s.slva_gamma):                                  # non-trivial SLVA biases
        p_.data += 0.3 * torch.randn_like(p_)
    its = items([2, 9, 14], 3); K = 8; x, pad, lens = VB.batch_x(its, dev); rows = torch.arange(len(its), device=dev)
    r = s.rollout(x, pad, K, coda_rows={K: rows}); lg = s.answer_logits(r["coda"][K][1], lens); o = {"no_nan_all_slots": bool(all(torch.isfinite(v).all() for v in r["V"]))}
    worst = 0.0
    for i, it in enumerate(its):
        x1, p1, l1 = VB.batch_x([it], dev); r1 = s.rollout(x1, p1, K, coda_rows={K: torch.arange(1, device=dev)}); L = int(l1[0])
        worst = max(worst, rel(r["V"][K][i, :L], r1["V"][K][0, :L]), rel(lg[i], s.answer_logits(r1["coda"][K][1], l1)[0]))
    o["padded_vs_alone_rel_max"] = worst; o["n_pad_slots"] = int((~pad).sum()); o["pass"] = o["no_nan_all_slots"] and worst <= 1e-9 and o["n_pad_slots"] > 0
    return o


@torch.no_grad()
def g_d5(dev):
    s, c = pair(dev); s.c_anneal.fill_(1e4); its = items([6, 10], 8); x, pad, lens = VB.batch_x(its, dev); K = 10; rows = torch.arange(len(its), device=dev); o = {}
    S = torch.zeros_like(pad); S[:, [3, 5]] = True

    def both(**kw):
        outs = []
        for m in (s, c):
            setup = kw.get("setup")
            if setup:
                setup(m)
            try:
                r = m.rollout(x, pad, K, coda_rows={K: rows}, **{k: v for k, v in kw.items() if k != "setup"})
            finally:
                m.ver_keep = m.age_floor = m.slot_mask = m.input_edit = None
            outs.append((r["V"], m.answer_logits(r["coda"][K][1], lens)))
        return bool(all(torch.equal(a, b) for a, b in zip(outs[0][0], outs[1][0])) and torch.equal(outs[0][1], outs[1][1]))
    F_ = [torch.randn_like(x, dtype=torch.float32)[..., None].expand(-1, -1, 768).contiguous() * 0.0 + s.rollout(x, pad, 2)["V"][2] for _ in range(1)]
    o["keep"] = both(setup=lambda m: setattr(m, "ver_keep", VE.ablate_keep)); o["v0_off_after"] = both(v0_off_after=2)
    o["foreign_ver_sub"] = both(ver_sub=F_)
    # FOREIGN INSERT: CARRY keeps its carry at V_0 by design, SLVA (as PA) reads the inserted entry as its newest version -> the comparator is CARRY with
    # its first pass input (t = o + 1) set to N(F_o) via input_edit (rms0 recomputed from V_0 with rollout's own formula)
    V0_ = c.rollout(x, pad, 0)["V"][0]; rms0_ = V0_.pow(2).mean(-1, keepdim=True).sqrt(); fin_ = c.nrm(F_[-1], rms0_); o_ = len(F_)
    rs_ = s.rollout(x, pad, K, coda_rows={K: rows}, prefix=F_)
    c.input_edit = lambda t, U: fin_.to(U.dtype) if t == o_ + 1 else U
    try:
        rc_ = c.rollout(x, pad, K, coda_rows={K: rows}, prefix=F_)
    finally:
        c.input_edit = None
    o["foreign_prefix_vs_carry_from_F"] = bool(all(torch.equal(a_, b_) for a_, b_ in zip(rs_["V"], rc_["V"])) and torch.equal(s.answer_logits(rs_["coda"][K][1], lens), c.answer_logits(rc_["coda"][K][1], lens))); o["age_floor"] = both(setup=lambda m: setattr(m, "age_floor", 0.1))
    o["slot_mask"] = both(setup=lambda m: setattr(m, "slot_mask", lambda t, V: S if t >= 1 else torch.zeros_like(S)))
    o["input_edit"] = both(setup=lambda m: setattr(m, "input_edit", lambda t, U: torch.cat([U[:, :1], U[:, :-1]], 1) if t >= 2 else U))
    for nm_, st_ in (("gather_splice", lambda m: setattr(m, "gather_splice", 8)), ("hist_self", lambda m: setattr(m, "gather_hist_ablate", "self")),
                     ("hist_other", lambda m: setattr(m, "gather_hist_ablate", "other"))):
        st_(s); s.gather_log = []; s.gather_split_log = []
        try:
            r = s.rollout(x, pad, K, coda_rows={K: rows})
        finally:
            s.gather_splice = None; s.gather_hist_ablate = None
        o["slva_%s_finite" % nm_] = bool(torch.isfinite(s.answer_logits(r["coda"][K][1], lens)).all())
        o["slva_%s_split_cross_max" % nm_] = float(max(float(z[3].abs().max()) for z in s.gather_split_log)); s.gather_log = s.gather_split_log = None
    s.gather_qmod = 4
    try:
        s.rollout(x, pad, 3); o["qmod_asserts"] = False
    except AssertionError:
        o["qmod_asserts"] = True
    s.gather_qmod = None
    # c 0: the masked versions' weight is exactly 0
    s.c_anneal.fill_(0.0); r = s.rollout(x, pad, 6); t = 6; rms0 = r["V"][0].pow(2).mean(-1, keepdim=True).sqrt(); gk = [s.gkey(V) for V in r["S"][:t]]; NV = [s.nrm(V, rms0) for V in r["S"][:t]]
    orig = torch.softmax

    def wts(**kw):
        cap = {}

        def spy(z, d):
            w = orig(z, d); cap["w"] = w; return w
        torch.softmax = spy
        try:
            s.gather(t, gk, NV, pad, **kw)
        finally:
            torch.softmax = orig
        return cap["w"]                                                     # (B, T, H, t)
    w0 = wts(); wk = wts(keep=VE.ablate_keep(t, t)); wv = wts(off_v0=True); s.gather_hist_ablate = "self"; ws = wts(); s.gather_hist_ablate = None
    mid = slice(1, t - 1)
    o["c0_unmasked_mid_mass_min"] = float(w0[..., mid].sum(-1).min()); o["c0_keep_mid_mass_max"] = float(wk[..., mid].abs().max())
    o["c0_offv0_v0_mass_max"] = float(wv[..., 0].abs().max()); o["c0_self_mid_mass_max"] = float(ws[..., mid].abs().max())
    o["pass"] = (all(o[k] for k in ("keep", "v0_off_after", "foreign_prefix_vs_carry_from_F", "foreign_ver_sub", "age_floor", "slot_mask", "input_edit", "qmod_asserts"))
                 and all(v for k, v in o.items() if k.endswith("_finite")) and all(v == 0.0 for k, v in o.items() if k.endswith("split_cross_max"))
                 and o["c0_unmasked_mid_mass_min"] > 0 and o["c0_keep_mid_mass_max"] == 0.0 and o["c0_offv0_v0_mass_max"] == 0.0 and o["c0_self_mid_mass_max"] == 0.0)
    return o


def g_d6(dev):
    from contextlib import nullcontext
    o = {}; its = [it for it in N.test_set(1000) if 3 <= it["n"] <= 5][:8]; Ks = [it["n"] + 2 for it in its]
    for cv, tag in ((0.0, "c0"), (1e4, "c1e4")):
        res = {}
        for rw in (0.0, 1.0):
            s, _ = pair(dev); s.train(); s.c_anneal.fill_(cv)
            for p_ in s.parameters():
                p_.requires_grad_(True)
            L = VB.vb_loss(s, its, Ks, dev, nullcontext, rw)[0]; L.backward()
            res[rw] = {n: (p_.grad.detach().clone() if p_.grad is not None else None) for n, p_ in s.named_parameters() if n.startswith("slva_")}; del s
        o[tag + "_R1_norms"] = {n: (None if g is None else float(g.norm())) for n, g in res[0.0].items()}
        o[tag + "_R1R2_norms"] = {n: (None if g is None else float(g.norm())) for n, g in res[1.0].items()}
        o[tag + "_R2_changes"] = all(g is not None and not torch.equal(g, res[0.0][n]) for n, g in res[1.0].items()) if tag == "c0" else None
    tree = ast.parse(open("D:/Latent/boxes/slva.py", encoding="utf-8").read()); dec = {}
    for nd in ast.walk(tree):
        if isinstance(nd, ast.ClassDef) and nd.name == "SLVA":
            for f_ in nd.body:
                if isinstance(f_, ast.FunctionDef):
                    dec[f_.name] = [ast.unparse(d) for d in f_.decorator_list]
    o["decorators"] = dec
    c0 = o["c0_R1_norms"], o["c0_R1R2_norms"]; c4 = o["c1e4_R1R2_norms"]
    o["pass"] = (all(v is not None and v > 0 and math.isfinite(v) for d in c0 for v in d.values()) and o["c0_R2_changes"]
                 and all(v is None or v == 0.0 for v in c4.values()) and dec.get("gather") == [] and dec.get("gkey") == [])
    return o


def g_plain(dev):
    o = {}; base = "D:/Latent/scratch/_tmp/s213_plain"
    for mode in ("pa", "carry"):
        logs = {}
        for tag, mod in (("pre", "boxes._vbank_pre_d3"), ("new", "boxes.vbank")):
            out = "%s/%s_%s" % (base, mode, tag); os.makedirs(out, exist_ok=True)
            for f_ in ("log.jsonl",):
                if os.path.exists(os.path.join(out, f_)):
                    os.remove(os.path.join(out, f_))
            cmd = [sys.executable, "-m", mod, "--mode", mode, "--steps", "48000", "--stop_at", "3", "--log_every", "1", "--bs", "16", "--micro", "1", "--device", dev, "--out", out]
            r = subprocess.run(cmd, cwd="D:/Latent", capture_output=True, text=True); assert r.returncode == 0, r.stderr[-2000:]
            logs[tag] = ([json.loads(l_) for l_ in open(os.path.join(out, "log.jsonl"))], json.load(open(os.path.join(out, "calibration.json"))))
        keys = ("loss", "answer_ce", "rec_ce", "gn_preclip", "c", "carry_mass")
        same = [[r_.get(k) for k in keys] for r_ in logs["pre"][0]] == [[r_.get(k) for k in keys] for r_ in logs["new"][0]]
        o[mode] = {"log_bitwise": same, "calibration_equal": logs["pre"][1] == logs["new"][1], "n_log": len(logs["new"][0]),
                   "loss_new": [r_.get("loss") for r_ in logs["new"][0]]}
    o["pass"] = all(o[md]["log_bitwise"] and o[md]["calibration_equal"] and o[md]["n_log"] == 3 for md in ("pa", "carry"))
    return o


@torch.no_grad()
def g_g1b(dev, runs=("s213_g1b_T", "s213_g1b_F"), steps=(0, 1000, 2000)):
    sys.path.insert(0, "D:/Latent/scratch")
    import gate_s211 as G
    its = G._train_items(256, 21108, 1, 8); y = torch.tensor(N.NUM_IDS[[it["t"] for it in its]], device=dev); nn_ = torch.tensor([it["n"] for it in its], device=dev)
    NUM = torch.tensor(N.NUM_IDS, device=dev); out = {}
    for run in runs:
        d = os.path.join("D:/Latent/boxes/runs_fsq", run); a = json.load(open(os.path.join(d, "args.json"))); m = VB.make_model(s5.load_gpt2(dev), a["mode"], a["table"]).to(dev); m.eval(); o = {}
        for st in steps:
            m.load_state_dict(torch.load(os.path.join(d, "ckpt_model_%d.pt" % st), map_location=dev, weights_only=False)["model"])
            pnl, cel = {t: [] for t in range(9)}, {t: [] for t in range(9)}
            for c0 in range(0, len(its), 64):
                sl = slice(c0, c0 + 64); xc, pc, lc = VB.batch_x(its[sl], dev); r = m.rollout(xc, pc, 8, coda_rows={t: torch.arange(xc.shape[0], device=dev) for t in range(9)})
                for t in range(9):
                    lg = m.answer_logits(r["coda"][t][1], lc); pnl[t].append(torch.softmax(lg, -1)[:, NUM].sum(-1)); cel[t].append(F.cross_entropy(lg, y[sl], reduction="none"))
            pn = {str(t): float(torch.cat(pnl[t]).mean()) for t in range(9)}; ce_t = {t: torch.cat(cel[t]) for t in range(9)}
            ce_kn = float(torch.stack([ce_t[int(n)][i] for i, n in enumerate(nn_.tolist())]).mean())
            o[str(st)] = {"p_num_by_pass": pn, "ce_kn": ce_kn, "drift": pn["1"] - pn["8"], "c": float(m.c_anneal)}
        o["pass"] = o[str(steps[-1])]["drift"] < o[str(steps[0])]["drift"] and o[str(steps[2])]["ce_kn"] < o[str(steps[1])]["ce_kn"] < o[str(steps[0])]["ce_kn"]
        out[run] = o
    out["pass"] = all(v["pass"] for k, v in out.items() if k != "pass"); return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("gate"); ap.add_argument("--device", default="cuda:1"); a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False; os.makedirs(OUT, exist_ok=True)
    r = {"d0": g_d0, "d1": g_d1, "d2": g_d2, "d5": g_d5, "d6": g_d6, "plain": g_plain, "g1b": g_g1b}[a.gate](a.device)
    r["device"] = torch.cuda.get_device_name(a.device) if a.device.startswith("cuda") else "cpu"
    json.dump(r, open(os.path.join(OUT, "%s.json" % a.gate), "w", newline="\n"), indent=1); print(a.gate, "PASS" if r["pass"] else "FAIL", json.dumps(r)[:1500], flush=True)


if __name__ == "__main__":
    main()
