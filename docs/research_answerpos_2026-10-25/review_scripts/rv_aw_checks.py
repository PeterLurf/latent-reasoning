"""REVIEW_AW $0 checks (CPU, 2 threads, idle priority). Base GPT-2 only; no trained model is loaded; aw_gate.py is NOT run.
  A  OC: exact binomial G-AW0 / LP2; aw_oc's P-2 table re-simulated with another RNG seed (+ a SOURCE-DRIFT truth).
  B  cost: aw_oc's measured-input cost model re-run; family guard with launch overheads; ST-AN inside vs after P-2 (expected spend).
  C  gamma schedule arithmetic (inject_ow line 352) for write_mix (2000, 6000) and (0, 0).
  D  D2's structure on base GPT-2 + DeltaMem as InjectOW builds it: after ONE write from the zero state, the read at each decoder position is
     a per-HEAD scalar times v_h (additive: phi(q).phi(k) > 0; delta: beta k^.q^, any sign); reports the scalars' spread over positions / items.
    python -B docs/research_answerpos_2026-10-25/review_scripts/rv_aw_checks.py   -> rv_aw_checks.json
"""
import contextlib
import importlib.util
import io
import json
import math
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2"); os.environ.setdefault("MKL_NUM_THREADS", "2"); os.environ.setdefault("BOXES_NB", "5")
try:
    import psutil
    psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS if os.name == "nt" else 19)
except Exception:  # noqa: BLE001
    pass
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__)); AWD = os.path.dirname(HERE); ROOT = os.path.abspath(os.path.join(AWD, "..", ".."))
sys.path.insert(0, ROOT)
OUT = os.path.join(HERE, "rv_aw_checks.json")


def binom_ge(n, p, k):
    return float(sum(math.comb(n, j) * p ** j * (1 - p) ** (n - j) for j in range(k, n + 1)))


def part_a():
    r = {"G_AW0_exact_n600_bar080": {str(t): binom_ge(600, t, 480) for t in (1.0, 0.855, 0.82, 0.80, 0.78, 0.75)},
         "LP2_exact_neff495_bar080": {str(t): binom_ge(495, t, 396) for t in (0.90, 0.85, 0.82, 0.80, 0.75, 0.40)}}
    sp = importlib.util.spec_from_file_location("aw_oc_copy", os.path.join(AWD, "aw_oc.py")); mod = importlib.util.module_from_spec(sp); sp.loader.exec_module(mod)
    mod.OUT = os.path.join(HERE, "rv_aw_oc_reseed.json"); mod.rng = np.random.default_rng(777)
    with contextlib.redirect_stdout(io.StringIO()):
        mod.main()
    re_ = json.load(open(mod.OUT)); orig = json.load(open(os.path.join(AWD, "aw_oc.json")))
    r["P2_PASS_reseed_vs_orig"] = {k: [round(v["PASS"], 4), round(orig["P2_pilot"][k]["PASS"], 4)] for k, v in re_["P2_pilot"].items()}
    g2 = mod.lp2(0.35); g3, C = mod.lp3(0.10, 0.10); g5 = mod.lp5(0.10, 0.0); gm = mod.b(0.10, 1000) <= C + 0.10; gs = mod.swapk(0.18, 0.18)
    r["SOURCE_DRIFT_truth"] = {"LP2": float(g2.mean()), "PASS": float((g2 & g3 & g5 & gm & gs).mean()),
                               "note": "same binding-gate pattern as no_effect / NOT-STORED; only WS-READOUT separates it"}
    r["cost_reseed_identical"] = re_["cost"] == orig["cost"]
    return r, orig["cost"]


def part_b(cost):
    g = {k: cost[k]["guard_x1.3"] for k in ("P2", "ST_AN_attribution", "P4", "Stage1")}; hi = {k: cost[k]["usd_low_high"][1] for k in g}
    lo = {k: cost[k]["usd_low_high"][0] for k in g}; fam = round(sum(g.values()), 2)
    launches = {"P2": 1, "ST_AN_attribution": 1, "P4": 2, "Stage1": 2}; bad = 0.26   # §233: refused-admission BAD $0.24, BAD $0.28
    p = {"G": 0.97, "LP2": 0.65, "rest": 0.30, "P4": 0.5}; pp2 = p["G"] * p["LP2"] * p["rest"]
    e_after = p["G"] * hi["P2"] + pp2 * (hi["ST_AN_attribution"] + hi["P4"] + p["P4"] * hi["Stage1"])
    e_inside = p["G"] * (hi["P2"] + hi["ST_AN_attribution"]) + pp2 * (hi["P4"] + p["P4"] * hi["Stage1"])
    return {"guards": g, "family_guard": fam, "headroom_to_30": round(30 - fam, 2),
            "family_guard_plus_launch_overheads_0.26_each": round(fam + bad * sum(launches.values()), 2),
            "family_high_no_guard": round(sum(hi.values()), 2), "family_low": round(sum(lo.values()), 2),
            "P_P2": round(pp2, 3), "expected_spend_ST_AN_after_pass": round(e_after, 2), "expected_spend_ST_AN_inside_P2": round(e_inside, 2),
            "P2_plus_ST_AN_guard": round(g["P2"] + g["ST_AN_attribution"], 2), "P2_with_one_UNREADABLE_rerun_guard": round(2 * g["P2"], 2)}


def part_c():
    def gam(step, s, e, sc=1.0):
        return 0.0 if step <= s * sc else min(1.0, (step - s * sc) / max(1.0, (e - s) * sc))
    return {"ramp_2000_6000": {st: gam(st, 2000, 6000) for st in (1, 2000, 2001, 4000, 6000)}, "flat_0_0": {st: gam(st, 0, 0) for st in (1, 2, 6000)}}


def part_d():
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(2)
    import boxes.deltamem as DM
    import boxes.inject_ow as IO
    import boxes.ow_words as OW
    from boxes.nested170 import EOT, pad
    g = IO.load_trunk("cpu", {"scratch": ""}); g.eval(); torch.manual_seed(1); m = IO.InjectOW(g, 6, "top", 6, 6, None); mem = m.mem; DM.FAST_ADD = True
    HA, HD, DH = mem.h_add, mem.h_del, DM.DH; D = mem.D
    s = OW.Stream(99173, frames=OW.FRAME_TRAIN, **IO.OWKW); its = [s.draw(2) for _ in range(24)]
    x1s = [[EOT] + IO.ENC.encode(" " + it["text"] + " " + OW.QUESTION.format(u=it["unit"], A=it["names"]["spine"][1])) + IO.SFX for it in its]
    decs = [IO.tok_ow(it)["dec"] for it in its]
    with torch.no_grad():
        ids_, _ = pad(x1s); x = torch.tensor(ids_); pos = torch.tensor([len(q) - 1 for q in x1s]); B = len(its); ar = torch.arange(B); cap = {}
        S0 = mem.zero_state(B, "cpu"); DM.gpt_forward_mem_r(g, g.embed(x), mem, S0, cap=cap)
        h6, hw = cap["h6"][ar, pos][:, None], cap["hw"][ar, pos][:, None]
        mem.gamma.fill_(1.0); S1 = DM.write_seq(mem, h6, hw, S0, None)
        u = mem.ln_w(hw.float()); _, k, v = mem.qkv(u).split(D, -1); k = k.view(B, HA + HD, DH); v = v.view(B, HA + HD, DH)
        gg = mem.gate(u)[:, 0]; be = 2 * torch.sigmoid(gg[:, HD:])
        idd, _ = pad(decs); xd = torch.tensor(idd); T = xd.shape[1]; valid = torch.tensor([[t < len(q) for t in range(T)] for q in decs])
        hd6 = DM.gpt_lower(g, g.embed(xd), mem.layer); a = mem.ln(hd6.float()); q = mem.qkv(a)[..., :D].view(B, T, HA + HD, DH)
        pq = F.elu(q[:, :, :HA]) + 1; pk = F.elu(k[:, :HA]) + 1
        c_add = torch.einsum("bthk,bhk->bth", pq, pk)
        c_del = be[:, None, :] * torch.einsum("bthk,bhk->bth", F.normalize(q[:, :, HA:], dim=-1), F.normalize(k[:, HA:], dim=-1))
        ra = DM.add_read(mem, pq, S1[:, :HA].float()); rd = torch.einsum("bhvk,bthk->bthv", S1[:, HA:HA + HD].float(), F.normalize(q[:, :, HA:], dim=-1))
        err_a = float((ra - c_add[..., None] * v[:, None, :HA]).abs().max() / ra.abs().max())
        err_d = float((rd - c_del[..., None] * v[:, None, HA:]).abs().max() / rd.abs().max())
        last = torch.tensor([len(q_) - 1 for q_ in decs]); ca = c_add[valid]; cd = c_del[valid]; cl = c_add[ar, last]
        cv_pos = [float((c_add[i, :len(decs[i])].std(0) / c_add[i, :len(decs[i])].mean(0)).mean()) for i in range(B)]
        out = {"items": B, "rel_err_read_eq_scalar_times_v_add": err_a, "rel_err_read_eq_scalar_times_v_del": err_d,
               "add_scalar_min": float(ca.min()), "add_scalar_max": float(ca.max()), "add_scalar_cv_over_positions_mean": float(np.mean(cv_pos)),
               "add_scalar_cv_over_items_at_dec_colon_mean": float((cl.std(0) / cl.mean(0)).mean()),
               "add_per_head_max_over_min_at_dec_colon_mean": float((cl.max(1).values / cl.min(1).values).mean()),
               "del_scalar_frac_negative": float((cd < 0).float().mean()), "del_scalar_min": float(cd.min()), "del_scalar_max": float(cd.max()),
               "note": "base GPT-2 init (askname not loaded): structure only. The read is 12 per-head scalars (per position, per item) times v_h, not one scalar times v"}
    return out


def main():
    res = {}
    res["A"], cost = part_a(); res["B"] = part_b(cost); res["C"] = part_c(); res["D"] = part_d()
    with open(OUT, "w", newline="\n") as f:
        json.dump(res, f, indent=1); f.write("\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    sys.exit(main())
