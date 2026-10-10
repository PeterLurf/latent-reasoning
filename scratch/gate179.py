"""THEORY §179 READOUT pre-launch gates (259c40cd + note 179-1): boxes/inject169.py --blind_decoder / --p0_evals (+ halt173's blind load).
  (1) the blind decoder sees no problem token: dec_pre is [" ="] only; with a fixed nonzero memory, dec_logits / greedy_ans are bitwise equal for two
      different problem sets with the same answers (and differ for the seeing decoder - the test can fail)
  (2) step-0 function preservation: a freshly built model's blind decoder logits are bitwise equal with a random memory and a zero memory (W_o = 0, fade 0)
  (3) gradflow (blind, fade path p = 0, 4x2 -> 8 pair passes; W_o perturbed, fade 1, gamma 0.5): every DeltaMem parameter, pgain, the trunk, and the
      memory after passes 1 and 2 get nonzero gradient from the decoder loss
  (4) flags off = pre-§179 inject169 bitwise: 6 CPU steps (1 thread) of the fade + pass-index path and of the K2 path; log losses and every eval.json
      field (except timings and the new blind_decoder key) equal
  (5) smoke of the cell flags (CPU, sched_scale 0.002): --blind_decoder --p0_evals 12000,24000 --fade 24000,48000 --stop_at 48 -> s179.acc_p0_by_step
      {"12000", "24000"} with acc / acc_by_column / len_err / digit_err; the seeing twin likewise; halt173 --dir177 loads the blind run
    python scratch/gate179.py <pre-179 inject169 copy>   -> boxes/runs_fsq/s179_gates/gate179.json"""
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
os.environ.setdefault("BOXES_NB", "5")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

import boxes.inject169 as I  # noqa: E402
from boxes import s5  # noqa: E402
from boxes.fixedchunk import DT, EQ, product_digits  # noqa: E402
from boxes.multcol import gen_operands  # noqa: E402

GD = os.path.join(ROOT, "boxes", "runs_fsq", "s179_gates"); SP = os.environ.get("GATE179_TMP", os.path.join(GD, "tmp"))


def build(blind, seed=1):
    torch.manual_seed(seed); m = I.Inject(s5.load_gpt2("cpu"), 6, "top"); m.enable_pass_index(); m.blind = blind; return m


def run(args, out, script=None):
    if os.environ.get("GATE179_REUSE") and os.path.exists(os.path.join(out, "eval.json")):   # a finished run from an earlier (timed-out) gate pass
        return 0, "reused"
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONPATH=ROOT)
    cmd = [sys.executable, "-u"] + ([script] if script else ["-m", "boxes.inject169"]) + args + ["--device", "cpu", "--out", out]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=ROOT); return r.returncode, r.stdout[-300:] + r.stderr[-600:]


def flat(d, pre=""):
    o = {}
    for k, v in d.items():
        if isinstance(v, dict):
            o.update(flat(v, pre + k + "."))
        else:
            o[pre + k] = v
    return o


def main():
    torch.set_num_threads(3); out = {}; ok = True; old = sys.argv[1]; r_ = np.random.default_rng(5)
    A1, B1 = gen_operands(r_, 6, 4), gen_operands(r_, 6, 3); A2, B2 = gen_operands(r_, 6, 4), gen_operands(r_, 6, 3)
    ans = torch.tensor(DT[product_digits(A1, B1)])
    # (1)
    g1 = {}
    for blind in (True, False):
        m = build(blind); m.eval()
        with torch.no_grad():
            m.mem.wo.weight.normal_(0, 0.02); m.mem.fade.fill_(1.0); S = torch.randn(6, 12, 64, 64) * 0.05
            l1, l2 = m.dec_logits(S, A1, B1, ans), m.dec_logits(S, A2, B2, ans); g1_ = m.greedy_ans(S, A1, B1, 7), m.greedy_ans(S, A2, B2, 7)
        g1["blind" if blind else "seeing"] = {"prefix_len": int(m.dec_pre(A1, B1).shape[1]), "prefix_is_eq": bool((m.dec_pre(A1, B1) == EQ).all()),
                                               "logits_equal_across_problems": bool(torch.equal(l1, l2)), "greedy_equal": bool(np.array_equal(*g1_))}
    out["g1"] = g1; out["g1_pass"] = (g1["blind"]["prefix_len"] == 1 and g1["blind"]["prefix_is_eq"] and g1["blind"]["logits_equal_across_problems"]
                                      and g1["blind"]["greedy_equal"] and not g1["seeing"]["logits_equal_across_problems"]); ok &= out["g1_pass"]
    # (2)
    m = build(True); m.eval()
    with torch.no_grad():
        la, lb = m.dec_logits(torch.randn(6, 12, 64, 64), A1, B1, ans), m.dec_logits(m.mem.zero_state(6, "cpu"), A1, B1, ans)
    out["g2"] = {"wo_zero": bool(m.mem.wo.weight.abs().max() == 0), "fade": float(m.mem.fade), "logits_equal_random_vs_zero_memory": bool(torch.equal(la, lb))}
    out["g2_pass"] = out["g2"]["logits_equal_random_vs_zero_memory"]; ok &= out["g2_pass"]
    # (3)
    m = build(True); m.train()
    with torch.no_grad():
        m.mem.wo.weight.normal_(0, 0.02); m.mem.fade.fill_(1.0); m.mem.gamma.fill_(0.5); m.pgain.normal_(0, 0.01)
    kept = []; w_ = I.Inject.write_mixed

    def wm(self, S, ids, last, mask, t):
        S2 = w_(self, S, ids, last, mask, t); S2.retain_grad(); kept.append(S2); return S2
    I.Inject.write_mixed = wm
    try:
        A, B = gen_operands(r_, 4, 4), gen_operands(r_, 4, 2); S = m.think_fade(m.mem.zero_state(4, "cpu"), A, B, 0.0, np.random.default_rng(0))
        a_ = torch.tensor(DT[product_digits(A, B)]); lg = m.dec_logits(S, A, B, a_); Fn.cross_entropy(lg.reshape(-1, lg.shape[-1]), a_.reshape(-1)).backward()
    finally:
        I.Inject.write_mixed = w_
    mz = [n_ for n_, p_ in m.mem.named_parameters() if p_.requires_grad and (p_.grad is None or float(p_.grad.norm()) == 0)]
    tg = {k: float(m.g.w(k).grad.norm()) if m.g.w(k).grad is not None else 0.0 for k in ("h.0.attn.c_attn.weight", "h.11.mlp.c_fc.weight")}
    out["g3"] = {"passes": len(kept), "mem_zero_grad": mz, "pgain_grad": float(m.pgain.grad.norm()) if m.pgain.grad is not None else 0.0, "trunk_grads": tg,
                 "S_grad_by_pass": [float(k_.grad.norm()) if k_.grad is not None else 0.0 for k_ in kept]}
    out["g3_pass"] = len(kept) == 8 and not mz and out["g3"]["pgain_grad"] > 0 and min(tg.values()) > 0 and min(out["g3"]["S_grad_by_pass"][:2]) > 0; ok &= out["g3_pass"]
    # (4)
    bench = os.path.join(ROOT, "bench", "data", "mult_4x4_test.txt"); g4 = {}
    tiny = ["--seed", "1", "--steps", "6", "--sched_scale", "0.002", "--log_every", "2", "--bs", "4", "--test_n", "4", "--ext_n", "4", "--probe_n_fit", "32",
            "--eval_bs", "4", "--ckpt_every", "1000", "--bench", bench]
    for nm, extra in (("fade_index", ["--think_k", "4n", "--fade", "12000,36000", "--pass_index", "sin_gain"]), ("k2", ["--think_k", "2"])):
        dn, do = os.path.join(SP, "new_" + nm), os.path.join(SP, "old_" + nm); [shutil.rmtree(x_, ignore_errors=True) for x_ in (dn, do) if not os.environ.get("GATE179_REUSE")]
        rn, en = run(extra + tiny, dn); ro, eo = run(extra + tiny, do, old)
        if rn or ro:
            g4[nm] = {"rc_new": rn, "rc_old": ro, "err": (en if rn else eo)[-500:]}; continue
        Ln = [json.loads(x) for x in open(os.path.join(dn, "log.jsonl"))]; Lo = [json.loads(x) for x in open(os.path.join(do, "log.jsonl"))]
        loss_eq = [r1.get("loss") for r1 in Ln] == [r2.get("loss") for r2 in Lo]
        fn = flat(json.load(open(os.path.join(dn, "eval.json")))); fo = flat(json.load(open(os.path.join(do, "eval.json"))))
        skip = lambda k: k.endswith("_sec") or k == "blind_decoder"   # noqa: E731
        diff = sorted(k for k in set(fn) | set(fo) if not skip(k) and fn.get(k) != fo.get(k))
        g4[nm] = {"rc_new": rn, "rc_old": ro, "log_losses_equal": loss_eq, "eval_fields_differing": diff[:20], "n_fields": len(fn)}
        for d_ in (dn, do):
            for f_ in ("model.pt", "ckpt_last.pt", "ckpt_prev.pt"):
                if os.path.exists(os.path.join(d_, f_)):
                    os.remove(os.path.join(d_, f_))
    out["g4"] = g4; out["g4_pass"] = all(v.get("log_losses_equal") and not v.get("eval_fields_differing") for v in g4.values()) and len(g4) == 2; ok &= out["g4_pass"]
    # (5)
    g5 = {}
    for nm, extra in (("blind", ["--blind_decoder"]), ("seeing", [])):
        d_ = os.path.join(SP, "cell_" + nm); os.environ.get("GATE179_REUSE") or shutil.rmtree(d_, ignore_errors=True)
        rc, err = run(["--think_k", "4n", "--fade", "24000,48000", "--pass_index", "sin_gain", "--fast_add", "--p0_evals", "12000,24000", "--stop_at", "48",
                       "--seed", "1", "--steps", "96", "--sched_scale", "0.002", "--log_every", "12", "--bs", "4", "--test_n", "8", "--ext_n", "4", "--probe_n_fit", "32",
                       "--eval_bs", "4", "--ckpt_every", "1000", "--bench", bench] + extra, d_)
        e_ = json.load(open(os.path.join(d_, "eval.json"))) if rc == 0 else {}; s_ = e_.get("s179", {})
        r5 = {"rc": rc, "err": err[-400:] if rc else "", "steps": sorted(s_.get("acc_p0_by_step", {})), "blind": s_.get("blind_decoder"),
              "fields": sorted(next(iter(s_.get("acc_p0_by_step", {}).values()), {})), "example": s_.get("acc_p0_by_step", {}).get("24000")}
        if nm == "blind" and rc == 0:                                       # halt173's loader on the blind run (the full CPU package took > 1 h: load + decode only)
            from boxes.halt173 import load_model
            kd_, mh, ah = load_model(d_, "cpu"); A_, B_ = gen_operands(np.random.default_rng(0), 2, 4), gen_operands(np.random.default_rng(1), 2, 4)
            with torch.no_grad():
                gh = mh.greedy_ans(mh.mem.zero_state(2, "cpu"), A_, B_, 8)
            r5["halt173_load"] = {"kind": kd_, "blind": bool(getattr(mh, "blind", False)), "prefix": mh.dec_pre(A_, B_).tolist(), "greedy_shape": list(gh.shape)}
            r5["halt173_dir177_rc"] = 0 if (kd_ == "inject" and r5["halt173_load"]["blind"] and r5["halt173_load"]["prefix"] == [[EQ], [EQ]]) else 1
        for f_ in ("model.pt", "ckpt_last.pt", "ckpt_prev.pt"):
            if os.path.exists(os.path.join(d_, f_)):
                os.remove(os.path.join(d_, f_))
        g5[nm] = r5
    out["g5"] = g5
    out["g5_pass"] = all(v["rc"] == 0 and v["steps"] == ["12000", "24000"] and {"acc", "acc_by_column", "len_err", "digit_err"} <= set(v["fields"]) for v in g5.values()) \
        and g5["blind"]["blind"] is True and g5["seeing"]["blind"] is False and g5["blind"].get("halt173_dir177_rc") == 0; ok &= out["g5_pass"]
    out["all_pass"] = bool(ok); os.makedirs(GD, exist_ok=True)
    json.dump(out, open(os.path.join(GD, "gate179.json"), "w", newline="\n"), indent=1); print("GATE179", json.dumps(out), flush=True); sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
