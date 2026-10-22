"""§224 FIXED-HOLD $0 gates (FH_v1 §6 at 232e84ab; Audit approval 4e56c8ca). The code under test is TREE/boxes/vbank.py + vbank_diag.py (the §224
working copy, a code-only copy of boxes/ under scratch/_tmp with ROOT pointed at D:/Latent) until it is committed; after commit run with --tree D:/Latent.
Each part writes scratch/gate_s224/<part>.json with a "pass" key.
  k1  gradflow (fp64 CPU; D3-T s1 weights, c 0; 8 test items n 3..8; T_far 13): L_far and L_FP (lambda 1) each give nonzero finite grads on the core
      blocks 6 / 9 and slva_wq (gain / slva_wk act only on the detached versions < T_far: reported); L_far also on coda block 10 (L_FP reads V_T_far before the coda: none expected there); every far-rollout V_t with
      t < T_far has requires_grad False and V_T_far True; the far prefix V_t equal a plain no-grad rollout's bitwise; for a plain vb_loss rollout of the
      same items autograd.grad(L_FP, its V_0..V_{n+2}) is None for every t (exactly zero); ast: far_branch / vb_loss / main / rollout undecorated.
  k2  anchor (fp64 CPU): r_anchor recomputed here per item from the returned rollout; L_FP == mean(g * r); r == 0 exactly when V_T_far := V_{n_i+2}
      (rollout wrapper); L_FP unchanged within 1e-12 rel when every V_t is scaled by 3.7; with the gold of 3 items shifted (+1 mod 11) L_FP ==
      mean(g * (not shifted) * r) (wrong items masked).
  k3  gate share == a separate no-grad rollout to T_far - 1 (coda argmax over the 11 value tokens == gold), per item and as a share.
  k4  plain path, HEAD (pre-FH; cwd D:/Latent) vs TREE, no new flags, CPU: 3 steps bs 16 log_every 1 for slva base, slva FT (init_from D3-T s1, lr 3e-5,
      cur 8, c 0,1), pa, carry -> log + calibration bitwise, same file list; rollout(grad_from=None) fp64 CPU (D3-T s1, 6 items, K 10, codas 4 / 8 / 10):
      every V / S / coda tensor bitwise.
  k5  T_far: 20,000 draws of default_rng(seed + 7100).integers(13, 49) -> support exactly 13..48, chi-square vs uniform < 66.62 (df 35, p 0.001); a 3-step CPU FT run with
      --far --log_every 1 logs far_t_mean == the first 3 draws of default_rng(1 + 7100) (one T_far per step).
  k6  CurGate isolation: a 12-step CPU FT fixture (cur 3,5,8 so CurGate records every step; --far --fh --fh_dose 3,6, far_t 13,14, far_n 4, bs 16,
      ckpt at 12; 1 thread) run twice with far_branch wrapped to offset L_far (and far_ce) by +100 vs 0 (a constant: identical gradients) -> CurGate
      state (12 recorded answer_ce values) and model state at the step-12 ckpt identical, logged far_ce differs by 100, lambda-set lines equal.
    python scratch/gate_s224.py k1 k2 k3 [--tree DIR]"""
import ast
import json
import math
import os
import subprocess
import sys
from contextlib import nullcontext

TREE = "D:/Latent/scratch/_tmp/fh224_tree"
for i, v in enumerate(sys.argv):
    if v == "--tree":
        TREE = sys.argv[i + 1]
PRE = "D:/Latent"
sys.path.insert(0, TREE)
import numpy as np  # noqa: E402
import torch  # noqa: E402

import boxes.nested170 as N  # noqa: E402
import boxes.vbank as VB  # noqa: E402
import boxes.vbank_eval as VE  # noqa: E402

OUT = "D:/Latent/scratch/gate_s224"; RUNS = "D:/Latent/boxes/runs_fsq"; D3T = RUNS + "/s213_d3t_s1"; TMP = "D:/Latent/scratch/_tmp/s224_gate"
ENV = {**os.environ, "OMP_NUM_THREADS": "3", "MKL_NUM_THREADS": "3"}
torch.set_num_threads(3)


def model64():
    m, _ = VE.load(D3T, "cpu"); m.double()
    for p_ in m.parameters():
        p_.requires_grad_(True)
    m.c_anneal.fill_(0.0); m.train(); return m


def items(per=(3, 4, 5, 6, 7, 8, 5, 6)):
    TE = N.test_set(1000); cnt = {}; out = []
    for n in per:
        k = cnt.get(n, 0); out.append([it for it in TE if it["n"] == n][k]); cnt[n] = k + 1
    return out


def r_anchor(r, its, T):
    """an independent recompute of r_anchor per item (fp64 when the rollout is)."""
    x, pad, lens = VB.batch_x(its, "cpu"); out = []
    for b, it in enumerate(its):
        A = r["V"][it["n"] + 2][b].detach(); V = r["V"][T][b].detach(); pm = pad[b][:, None].to(A.dtype)
        out.append(float((((V - A) ** 2) * pm).sum() / ((A ** 2) * pm).sum()))
    return np.array(out)


def gate_sep(m, its, T):
    x, pad, lens = VB.batch_x(its, "cpu"); rows = torch.arange(len(its))
    with torch.no_grad():
        r = m.rollout(x, pad, T - 1, coda_rows={T - 1: rows})
        a = m.answer_logits(r["coda"][T - 1][1], lens)[:, torch.tensor(N.NUM_IDS)].argmax(-1).numpy()
    return (a == np.array([it["t"] for it in its])).astype(float)


def k1():
    o = {}; m = model64(); its = items(); T = 13; P = dict(m.named_parameters())
    names = ["g.p.h_6_attn_c_attn_weight", "g.p.h_9_mlp_c_fc_weight", "g.p.h_10_attn_c_attn_weight", "gain", "slva_wq", "slva_wk", "beta", "rec.weight"]
    Lf, Lp, st, r = VB.far_branch(m, its, T, "cpu", nullcontext); o["stats"] = st
    gr = lambda L: [None if g_ is None else float(g_.norm()) for g_ in torch.autograd.grad(L, [P[k] for k in names], retain_graph=True, allow_unused=True)]   # noqa: E731
    o["grad_L_far"] = dict(zip(names, gr(Lf))); o["grad_L_FP"] = dict(zip(names, gr(Lp)))
    o["prefix_requires_grad"] = [bool(r["V"][t].requires_grad) for t in range(T)]; o["V_T_requires_grad"] = bool(r["V"][T].requires_grad)
    x, pad, lens = VB.batch_x(its, "cpu")
    with torch.no_grad():
        rp = m.rollout(x, pad, T)
    o["prefix_bitwise_vs_plain"] = all(bool(torch.equal(r["V"][t], rp["V"][t])) for t in range(T))
    L, _, rpl = VB.vb_loss(m, its, [it["n"] + 2 for it in its], "cpu", nullcontext, 1.0, keep=True)
    tmax = max(it["n"] for it in its) + 2
    gv = torch.autograd.grad(Lp, [rpl["V"][t] for t in range(tmax + 1)], retain_graph=True, allow_unused=True); o["plain_V_grad_from_L_FP"] = [g_ is None for g_ in gv]
    src = open(os.path.join(TREE, "boxes", "vbank.py"), encoding="utf-8").read(); dec = {}
    for nd in ast.walk(ast.parse(src)):
        if isinstance(nd, ast.FunctionDef) and nd.name in ("far_branch", "vb_loss", "main", "rollout"):
            dec[nd.name] = [ast.unparse(d) for d in nd.decorator_list]
    o["decorators"] = dec
    nz = lambda d, ks: all(d[k] is not None and d[k] > 0 and math.isfinite(d[k]) for k in ks)   # noqa: E731
    core = ["g.p.h_6_attn_c_attn_weight", "g.p.h_9_mlp_c_fc_weight", "slva_wq"]   # gain / slva_wk: reported (they act only on bank versions < T_far: detached)
    o["pass"] = (nz(o["grad_L_far"], core + ["g.p.h_10_attn_c_attn_weight"]) and nz(o["grad_L_FP"], core) and o["grad_L_FP"]["g.p.h_10_attn_c_attn_weight"] in (None, 0.0)
                 and not any(o["prefix_requires_grad"]) and o["V_T_requires_grad"] and o["prefix_bitwise_vs_plain"] and all(o["plain_V_grad_from_L_FP"])
                 and all(v == [] for v in dec.values()) and st["fh_gate_share"] > 0)
    return o


def k2():
    o = {}; m = model64(); its = items(); T = 13
    Lf, Lp, st, r = VB.far_branch(m, its, T, "cpu", nullcontext); rr = r_anchor(r, its, T); g = gate_sep(m, its, T)
    o["L_FP"] = float(Lp); o["mean_g_r_recomputed"] = float(np.mean(g * rr)); o["rr"] = rr.tolist(); o["g"] = g.tolist()
    o["eq_formula"] = abs(o["L_FP"] - o["mean_g_r_recomputed"]) <= 1e-12 * max(1.0, abs(o["L_FP"]))
    orig = m.rollout

    def wrap(fn):
        def w(*a, **k):
            r_ = orig(*a, **k); fn(r_, a[2]); return r_
        return w

    def anchor_eq(r_, T_):
        r_["V"][T_] = torch.stack([r_["V"][it["n"] + 2][b] for b, it in enumerate(its)])
    m.rollout = wrap(anchor_eq); _, Lp0, st0, _ = VB.far_branch(m, its, T, "cpu", nullcontext); o["L_FP_anchor_equal"] = float(Lp0); o["r_far_anchor_equal"] = st0["fh_r_far"]

    def scale(r_, T_):
        for t in range(len(r_["V"])):
            r_["V"][t] = r_["V"][t] * 3.7
    m.rollout = wrap(scale); _, Lps, _, _ = VB.far_branch(m, its, T, "cpu", nullcontext); o["L_FP_scaled"] = float(Lps)
    o["scale_rel_diff"] = abs(float(Lps) - float(Lp)) / max(abs(float(Lp)), 1e-300); m.rollout = orig
    sh = [0, 3, 5]; its2 = [dict(it, t=(it["t"] + 1) % N.P) if b in sh else it for b, it in enumerate(its)]
    _, Lp2, st2, _ = VB.far_branch(m, its2, T, "cpu", nullcontext); keep = np.array([0.0 if b in sh else 1.0 for b in range(len(its))])
    o["L_FP_shifted"] = float(Lp2); o["expected_shifted"] = float(np.mean(g * keep * rr)); o["gate_share_shifted"] = st2["fh_gate_share"]
    o["pass"] = (o["eq_formula"] and o["L_FP_anchor_equal"] == 0.0 and o["r_far_anchor_equal"] == 0.0 and o["scale_rel_diff"] <= 1e-12
                 and abs(o["L_FP_shifted"] - o["expected_shifted"]) <= 1e-12 * max(1.0, abs(o["expected_shifted"])) and sum(g) >= 6 and o["L_FP_shifted"] < o["L_FP"])
    return o


def k3():
    o = {}; m = model64(); its = items(); T = 13
    _, _, st, r = VB.far_branch(m, its, T, "cpu", nullcontext); g = gate_sep(m, its, T); x, pad, lens = VB.batch_x(its, "cpu")
    gi = (m.answer_logits(r["coda"][T - 1][1], lens)[:, torch.tensor(N.NUM_IDS)].argmax(-1).numpy() == np.array([it["t"] for it in its])).astype(float)
    o.update(share_far_branch=st["fh_gate_share"], share_separate=float(g.mean()), per_item_far=gi.tolist(), per_item_separate=g.tolist())
    o["pass"] = o["share_far_branch"] == o["share_separate"] and o["per_item_far"] == o["per_item_separate"]; return o


def run_trainer(cwd, out, extra, runner=None, env=None):
    os.makedirs(out, exist_ok=True)
    for f_ in os.listdir(out):
        os.remove(os.path.join(out, f_))
    cmd = ([sys.executable, "-B", "-m", "boxes.vbank"] if runner is None else [sys.executable, "-B"] + runner) + extra + ["--device", "cpu", "--out", out]
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env=env or ENV); assert r.returncode == 0, (cmd, r.stderr[-3000:])
    return [json.loads(l_) for l_ in open(os.path.join(out, "log.jsonl"))], sorted(os.listdir(out))


FT = ["--mode", "slva", "--init_from", D3T + "/model.pt", "--steps", "12000", "--lr", "3e-5", "--cur", "8", "--c_sched", "0,1"]


def k4():
    o = {}
    cfgs = {"slva_base": ["--mode", "slva"], "slva_ft": FT, "pa": ["--mode", "pa"], "carry": ["--mode", "carry"]}
    for name, ex in cfgs.items():
        res = {}
        for tag, cwd in (("pre", PRE), ("new", TREE)):
            out = "%s/k4_%s_%s" % (TMP, name, tag); ex_ = ex + (["--steps", "48000"] if "--steps" not in ex else [])
            lg, files = run_trainer(cwd, out, ex_ + ["--stop_at", "3", "--log_every", "1", "--bs", "16", "--micro", "1"])
            res[tag] = (lg, json.load(open(os.path.join(out, "calibration.json"))), files)
        nosec = lambda lg: [{k: v for k, v in r_.items() if k != "sec"} for r_ in lg]  # noqa: E731  (sec = wall time)
        o[name] = {"log_bitwise": nosec(res["pre"][0]) == nosec(res["new"][0]), "calibration_equal": res["pre"][1] == res["new"][1],
                   "n_log": len(res["new"][0]), "files_equal": res["pre"][2] == res["new"][2], "files_new": res["new"][2]}
    dump = ("import sys, torch, json; sys.path.insert(0, '.'); import boxes.vbank_eval as VE, boxes.vbank as VB, boxes.nested170 as N\n"
            "m, _ = VE.load('%s', 'cpu'); m.double(); TE = N.test_set(1000); its = [it for it in TE if it['n'] in (2, 5, 8)][:6]\n"
            "x, pad, lens = VB.batch_x(its, 'cpu'); rows = torch.arange(len(its))\n"
            "with torch.no_grad():\n    r = m.rollout(x, pad, 10, coda_rows={4: rows, 8: rows, 10: rows})\n"
            "torch.save({'V': r['V'], 'S': r['S'], 'coda': {k: v[1] for k, v in r['coda'].items()}}, sys.argv[1])\n") % D3T
    pt = {}
    for tag, cwd in (("pre", PRE), ("new", TREE)):
        pt[tag] = "%s/k4_rollout_%s.pt" % (TMP, tag); os.makedirs(TMP, exist_ok=True)
        r = subprocess.run([sys.executable, "-B", "-c", dump, pt[tag]], cwd=cwd, capture_output=True, text=True, env=ENV); assert r.returncode == 0, r.stderr[-2000:]
    a, b = torch.load(pt["pre"]), torch.load(pt["new"])
    o["rollout_fp64_bitwise"] = (all(torch.equal(u, v) for u, v in zip(a["V"], b["V"])) and all(torch.equal(u, v) for u, v in zip(a["S"], b["S"]))
                                 and a["coda"].keys() == b["coda"].keys() and all(torch.equal(a["coda"][k], b["coda"][k]) for k in a["coda"]) and a["V"][0].dtype == torch.float64)
    o["pass"] = o["rollout_fp64_bitwise"] and all(o[k]["log_bitwise"] and o[k]["calibration_equal"] and o[k]["n_log"] == 3 and o[k]["files_equal"] for k in cfgs)
    return o


def k5():
    o = {}; d = np.random.default_rng(1 + 7100).integers(13, 49, 20000); u, c = np.unique(d, return_counts=True)
    e = len(d) / len(u); o["support"] = [int(u.min()), int(u.max()), len(u)]; o["chi2"] = float(((c - e) ** 2 / e).sum()); o["chi2_crit_df35_p001"] = 66.62
    lg, _ = run_trainer(TREE, TMP + "/k5", FT + ["--far", "--stop_at", "3", "--log_every", "1", "--bs", "16", "--micro", "1", "--far_n", "4"])
    o["logged_t_far"] = [r_.get("far_t_mean") for r_ in lg if "loss" in r_]; o["expected"] = [int(v) for v in np.random.default_rng(1 + 7100).integers(13, 49, 3)]
    o["far_fields"] = sorted(k for k in lg[-1] if k.startswith(("far_", "fh_")))
    o["pass"] = o["support"] == [13, 48, 36] and o["chi2"] < o["chi2_crit_df35_p001"] and o["logged_t_far"] == o["expected"]; return o


def k6():
    o = {}; runner = "D:/Latent/scratch/s224_k6_runner.py"; ck = {}
    for off in ("0", "100"):
        out = TMP + "/k6_off" + off
        ft = [v if v != "8" else "3,5,8" for v in FT]                     # a multi-stage curriculum: CurGate.update records every step's answer_ce
        lg, _ = run_trainer(TREE, out, ft + ["--far", "--fh", "--fh_dose", "3,6", "--far_t", "13,14", "--far_n", "4", "--bs", "16", "--micro", "1",
                                              "--stop_at", "12", "--ckpt_every", "12", "--log_every", "3"], runner=[runner, TREE, off],
                            env={**ENV, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})   # 1 thread: the CPU embedding backward is not bitwise across runs at 3 threads
        c_ = torch.load(os.path.join(out, "ckpt_last.pt"), weights_only=False); ck[off] = (c_["gate"], c_["model"], c_["fh_state"], lg)
    o["gate_equal"] = ck["0"][0] == ck["100"][0]; o["gate"] = ck["0"][0]
    o["model_bitwise"] = all(torch.equal(ck["0"][1][k], ck["100"][1][k]) for k in ck["0"][1])
    fc = lambda lg: [r_["far_ce"] for r_ in lg if "far_ce" in r_]  # noqa: E731
    o["far_ce_diff"] = [b - a for a, b in zip(fc(ck["0"][3]), fc(ck["100"][3]))]
    lam = lambda lg: [r_ for r_ in lg if "fh_lambda_set" in r_]  # noqa: E731
    o["lambda_set"] = [lam(ck["0"][3]), lam(ck["100"][3])]; o["fh_state"] = {k: v for k, v in ck["0"][2].items() if k != "ratios"}
    o["pass"] = (o["gate_equal"] and len(o["gate"]["cur"]) == 12 and o["model_bitwise"] and len(o["far_ce_diff"]) == 4 and all(abs(d - 100) < 1e-6 for d in o["far_ce_diff"])
                 and len(o["lambda_set"][0]) == 1 and o["lambda_set"][0] == o["lambda_set"][1] and ck["0"][2] == ck["100"][2])
    return o


def main():
    os.makedirs(OUT, exist_ok=True)
    for part in [a for a in sys.argv[1:] if a in ("k1", "k2", "k3", "k4", "k5", "k6")]:
        res = {"k1": k1, "k2": k2, "k3": k3, "k4": k4, "k5": k5, "k6": k6}[part](); res["tree"] = TREE
        json.dump(res, open(os.path.join(OUT, part + ".json"), "w", newline="\n"), indent=1, default=str); print(part, "pass", res["pass"], flush=True)


if __name__ == "__main__":
    main()
