"""§240 LANG-HOLD build gates (LH_v1 §2-3, §6, 62f16228; Note 240-1 Audit conditions; for Audit's G-D). -> scratch/gate_s240/<part>.json ("pass"). 4060 (cuda:0) / CPU.
Reference kept: boxes/_inject_aw_s238.py (the §238 module, 2514c6ca..e62e50a1). INIT: boxes/runs_fsq/df_plaw4_s1/model.pt (and _s2).
  flagsoff  --hold_inc 0, no --hold_log: _inject_aw_s238.py vs the new inject_aw.py at the §238 DF args (LAT + init, CD + init), det, fp32, micro 8, sched 0.02, stop_at 30:
            every listed ckpt field torch.equal, logs equal.
  holdlog   Note 240-1 (3): --hold_log (the CTRL arms' logging) does not change training: new module with vs without --hold_log at the §240 LAT args (init df_plaw4_s1),
            det, fp32, stop_at 30: ckpt model / opt / rngs / stream equal; logs equal after removing the §240 keys; gn_plain_mean100 == gn_preclip_mean100 (rel <= 1e-4:
            no penalty, the same plain gradient).
  detached  Note 240-1 (1): fp64 CPU, perturbed model, gamma 1: a d 2 and a d 4 batch rolled out WITH grad (S_t kept as graph tensors, retain_grad); hold_step(inc_on,
            far on the first item, T_far 16) -> every S_t.grad is None (L_inc never reaches the rollout: zero on the writes of passes <= d + 1, and every other pass);
            L_inc's grads reach the memory write params (qkv, gate, ln_w) and trunk blocks 6-11 and 0-5 (the penalty's own lower forward).
  start     hold_step with inc_on False leaves every param grad None (no backward: L_inc off before hold_start); inc_on True -> nonzero.
  smoke     FH (--hold_inc 1.6) and CTRL (--hold_log) from df_plaw4_s1, sched_scale 0.02 (penalty from step 10), micro 8, stop_at 50, log 10 -> rc 9; the step-0 norm-guard
            record; bins: FH inc_ratio / far_ratio set, CTRL None; gn_plain_mean100 present; the norm-guard trajectory of both (descriptive).
  ftime     4060 step time, --time_steps 40 (steps 1-40; penalty on from step 10 at sched 0.02): plain (§238 path), CTRL (--hold_log), FH -> the FH / plain ratio.
  mem       --mem_check at the FH shape (penalty + far at 40 inside the peak) and the CTRL shape on the 4060 (micro 8 reported; the 3090 decides at K7).
  args      --args_only lh_fh_s1 / lh_fh_s2 / lh_ctrl_s1 / lh_ctrl_s2 vs s238 frozen df_plaw4_s1 / s2: diff == {init_model, init_model_sha, init_model_sha256, steps,
            eval_parts} + the §240 keys {hold_inc, hold_start, hold_far_n, hold_far_lo, hold_far_hi, hold_log} (new keys; FH hold_inc 1.6, CTRL hold_log true).
    python -B scratch/gate_s240.py args detached start flagsoff holdlog smoke ftime mem"""
import json
import math
import os
import sys

ROOT = "D:/Latent"; sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/scratch")
import gate_s228 as G8  # noqa: E402
import gate_s234 as G4  # noqa: E402
import gate_s234_p4 as P4  # noqa: E402
import gate_s237 as G7  # noqa: E402
import gate_s238 as G38  # noqa: E402

OUT = ROOT + "/scratch/gate_s240"; TMP = ROOT + "/scratch/_tmp/s240"; DEV = "cuda:0"
LH_EVAL = "acc,ws,opclass,swap_prod,swap_kd,lp5h,cue_only"; DF = ["--mul_s_p", "2"]
INIT = {1: "boxes/runs_fsq/df_plaw4_s1/model.pt", 2: "boxes/runs_fsq/df_plaw4_s2/model.pt"}


def initf(s):
    return ["--init_model", INIT[s], "--init_model_sha", G7.sha(INIT[s])[:16]]


def lat(s=1):
    fl = P4.LAT4 + DF + initf(s); i = fl.index("--steps"); fl[i + 1] = "8000"; return fl


FH, CTRL = ["--hold_inc", "1.6"], ["--hold_log"]
STRIP = ("gn_plain_mean100", "hold")


def flagsoff():
    o = {}
    for tag, fl in (("lat", G38.LATD), ("cd", G38.CDD)):
        com = fl + ["--fp32", "--micro", "8", "--sched_scale", "0.02", "--ckpt_every", "30", "--log_every", "10", "--stop_at", "30"]
        da, db = TMP + "/fo_s238mod_" + tag, TMP + "/fo_newmod_" + tag
        ra = G4.det("boxes/_inject_aw_s238.py", com, da); rb = G4.det("boxes/inject_aw.py", com, db); r = {"rc": [ra, rb]}
        if os.path.exists(da + "/ckpt_last.pt") and os.path.exists(db + "/ckpt_last.pt"):
            r.update(G8.ck_equal(da + "/ckpt_last.pt", db + "/ckpt_last.pt", keys=("model", "opt", "rng_d", "np_rng", "torch_rng", "cuda_rng", "stream_rng", "stream_stats", "expo",
                                                                                     "binacc", "vbin", "vbin_d", "gnbin", "gn_log", "wg_hist", "dseq", "iseq")))
            strip = lambda p: [{k: v for k, v in json.loads(l_).items() if k != "sec"} for l_ in open(p)]  # noqa: E731
            r["log_equal"] = strip(da + "/log.jsonl") == strip(db + "/log.jsonl")
        r["pass"] = r["rc"] == [9, 9] and r.get("n_mismatch") == 0 and bool(r.get("log_equal")); o[tag] = r
    o["pass"] = all(v["pass"] for v in o.values()); return o


def holdlog():
    com = lat(1) + ["--fp32", "--micro", "8", "--sched_scale", "0.02", "--ckpt_every", "30", "--log_every", "10", "--stop_at", "30"]
    da, db = TMP + "/hl_off", TMP + "/hl_on"; ra = G4.det("boxes/inject_aw.py", com, da); rb = G4.det("boxes/inject_aw.py", com + CTRL, db); o = {"rc": [ra, rb]}
    if os.path.exists(da + "/ckpt_last.pt") and os.path.exists(db + "/ckpt_last.pt"):
        o.update(G8.ck_equal(da + "/ckpt_last.pt", db + "/ckpt_last.pt", keys=("model", "opt", "rng_d", "np_rng", "torch_rng", "cuda_rng", "stream_rng", "stream_stats", "gn_log")))
        la = [json.loads(l_) for l_ in open(da + "/log.jsonl")]; lb = [json.loads(l_) for l_ in open(db + "/log.jsonl")]
        lb_core = [{k: v for k, v in r.items() if k not in STRIP and k != "sec"} for r in lb if not ("norm_guard" in r and r.get("step") == 0)]
        o["log_equal_minus_s240_keys"] = [{k: v for k, v in r.items() if k != "sec"} for r in la] == lb_core
        pairs = [(r["gn_plain_mean100"], r["gn_preclip_mean100"]) for r in lb if "gn_plain_mean100" in r]
        o["gn_plain_vs_preclip_rel_max"] = max(abs(p - q) / max(abs(q), 1e-12) for p, q in pairs) if pairs else None
        o["has_step0_norm_guard"] = any("norm_guard" in r and r.get("step") == 0 for r in lb)
    o["pass"] = (o["rc"] == [9, 9] and o.get("n_mismatch") == 0 and bool(o.get("log_equal_minus_s240_keys")) and o.get("gn_plain_vs_preclip_rel_max") is not None
                 and o["gn_plain_vs_preclip_rel_max"] <= 1e-4 and o.get("has_step0_norm_guard")); return o


def _rollout_keep(m, its, K):
    import boxes.inject_aw as AW
    import torch
    xs, msk = AW.cue_inputs(m, its, "cpu"); S = m.mem.zero_state(len(its), "cpu").double(); S.requires_grad_(True); Ss = [S]
    hls = [m.lower(x) for x in xs]
    for t in range(1, K + 1):
        S = m.write(S, xs[min(t, len(xs)) - 1], msk, hls[min(t, len(xs)) - 1]); S.retain_grad(); Ss.append(S)
    return Ss, xs, msk, torch


def detached():
    import contextlib
    with G8.F64():
        import torch
        import boxes.inject_aw as AW
        import boxes.inject_ow as IO
        import boxes.ow_words as OW
        torch.set_num_threads(2); m, g = G4.aw_model(torch.float64); s = OW.Stream(24900, frames=OW.FRAME_TRAIN, mul_s_p=2, **IO.OWKW); o = {}
        for d in (2, 4):
            its = [AW.tok_aw(s.draw(d), "value", "step", "answer") for _ in range(3)]; m.zero_grad(set_to_none=True)
            Ss, xs, msk, _ = _rollout_keep(m, its, 8)
            st = AW.hold_step(m, [(Ss, xs, msk, [d] * 3)], its[:1], 16, 8, 3, True, 1.6, contextlib.nullcontext, "cpu")
            sg = {t: (None if Ss[t].grad is None else float(Ss[t].grad.abs().max())) for t in range(1, 9)}
            gr = {n: (float(p.grad.norm()) if p.grad is not None else None) for n, p in m.named_parameters()}
            nz = lambda pref: sum(1 for n, v in gr.items() if n.startswith(pref) and v is not None and v > 0 and math.isfinite(v))  # noqa: E731
            o["d%d" % d] = {"S_grad": sg, "all_S_grad_none_or_zero": all(v is None or v == 0 for v in sg.values()),
                            "mem_qkv_gate_lnw_nonzero": [n for n in ("mem.qkv.weight", "mem.gate.weight", "mem.ln_w.weight") if gr.get(n) and gr[n] > 0],
                            "trunk_blocks_upper_nonzero": nz("g.p.h_6") + nz("g.p.h_11"), "trunk_blocks_lower_nonzero": nz("g.p.h_0"), "inc": st["inc"], "far": st["far"]}
    o["pass"] = all(o[k]["all_S_grad_none_or_zero"] and len(o[k]["mem_qkv_gate_lnw_nonzero"]) == 3 and o[k]["trunk_blocks_upper_nonzero"] > 0 and o[k]["trunk_blocks_lower_nonzero"] > 0
                    and o[k]["inc"][1] == 3 and o[k]["far"][1] == 1 for k in ("d2", "d4"))
    return o


def start():
    import contextlib
    with G8.F64():
        import torch
        import boxes.inject_aw as AW
        import boxes.inject_ow as IO
        import boxes.ow_words as OW
        torch.set_num_threads(2); m, g = G4.aw_model(torch.float64); s = OW.Stream(24901, frames=OW.FRAME_TRAIN, mul_s_p=2, **IO.OWKW); o = {}
        its = [AW.tok_aw(s.draw(3), "value", "step", "answer") for _ in range(2)]
        for on in (False, True):
            m.zero_grad(set_to_none=True); Ss, xs, msk, _ = _rollout_keep(m, its, 8)
            AW.hold_step(m, [([x.detach() for x in Ss], xs, msk, [3, 3])], its[:1], 16, 8, 2, on, 1.6, contextlib.nullcontext, "cpu")
            o["inc_on_%s" % on] = sum(1 for p in m.parameters() if p.grad is not None and float(p.grad.abs().max()) > 0)
    o["pass"] = o["inc_on_False"] == 0 and o["inc_on_True"] > 0; return o


def _run(tag, fl, extra):
    d = G8.fresh(TMP + "/" + tag); rc = G8.sh([sys.executable, "-B", "-m", "boxes.inject_aw"] + fl + extra + ["--device", DEV, "--out", d], d + "/stdout.log"); return d, rc


def smoke():
    o = {}
    for tag, ex in (("fh", FH), ("ctrl", CTRL)):
        d, rc = _run("smoke_" + tag, lat(1), ex + ["--sched_scale", "0.02", "--micro", "8", "--stop_at", "50", "--ckpt_every", "0", "--log_every", "10"])
        lg = [json.loads(l_) for l_ in open(d + "/log.jsonl")] if os.path.exists(d + "/log.jsonl") else []
        z = [r for r in lg if r.get("step") == 0 and "norm_guard" in r]; bins = [r for r in lg if "by_d" in r]; hb = bins[-1].get("hold", {}) if bins else {}
        o[tag] = {"rc": rc, "step0": z[0] if z else None, "last_bin_hold": hb, "gn_plain": [r.get("gn_plain_mean100") for r in lg if "gn_plain_mean100" in r][-2:]}
    f, c = o["fh"], o["ctrl"]
    o["pass"] = (f["rc"] == 9 and c["rc"] == 9 and f["step0"] is not None and c["step0"] is not None and f["last_bin_hold"].get("inc_ratio") is not None
                 and f["last_bin_hold"].get("far_ratio") is not None and c["last_bin_hold"].get("inc_ratio") is None and len(f["gn_plain"]) > 0 and len(c["gn_plain"]) > 0)
    return o


def ftime():
    o = {}
    for tag, ex in (("plain", []), ("ctrl", CTRL), ("fh", FH)):
        d, rc = _run("ftime_" + tag, lat(1), ex + ["--sched_scale", "0.02", "--micro", "8", "--time_steps", "40", "--log_every", "10"])
        o[tag] = {"rc": rc, "sec_per_step": json.load(open(d + "/timing.json"))["sec_per_step"] if os.path.exists(d + "/timing.json") else None}
    o["fh_over_plain"] = o["fh"]["sec_per_step"] / o["plain"]["sec_per_step"] if o["fh"]["sec_per_step"] and o["plain"]["sec_per_step"] else None
    o["ctrl_over_plain"] = o["ctrl"]["sec_per_step"] / o["plain"]["sec_per_step"] if o["ctrl"]["sec_per_step"] and o["plain"]["sec_per_step"] else None
    o["note"] = "4060, micro 8, sched 0.02: steps 1-40, the penalty on from step 10 (31 of 40 steps); the 3090 G-TIME line runs --hold_start 1"
    o["pass"] = all(o[k]["rc"] == 0 for k in ("plain", "ctrl", "fh")) and o["fh_over_plain"] is not None; return o


def mem():
    o = {}
    for tag, ex in (("fh", FH), ("ctrl", CTRL)):
        d, rc = _run("mem_" + tag, lat(1), ex + ["--auto_micro", "--mem_budget", "6.0", "--time_steps", "5"])
        o[tag] = {"rc": rc, **({"mem_check": json.load(open(d + "/mem_check.json"))} if os.path.exists(d + "/mem_check.json") else {})}
    o["pass"] = all(o[k]["rc"] == 0 and "mem_check" in o[k] for k in ("fh", "ctrl")); return o


def args_part():
    import s228_args_check as C
    o = {}; der = {"init_model", "init_model_sha", "init_model_sha256", "steps", "eval_parts", "hold_inc", "hold_start", "hold_far_n", "hold_far_lo", "hold_far_hi", "hold_log"}
    for tag, s, ex, ref in (("lh_fh_s1", 1, FH, "df_plaw4_s1"), ("lh_fh_s2", 2, FH, "df_plaw4_s2"), ("lh_ctrl_s1", 1, CTRL, "df_plaw4_s1"), ("lh_ctrl_s2", 2, CTRL, "df_plaw4_s2")):
        fl = lat(s) + ex + (["--seed", "2"] if s == 2 else []) + ["--auto_micro", "--ckpt_every", "1000", "--eval_parts", LH_EVAL]
        d = G8.fresh(TMP + "/args_" + tag); rc = G8.sh([sys.executable, "-B", "-m", "boxes.inject_aw"] + fl + ["--device", "cpu", "--args_only", "--out", d], d + "/stdout.log")
        new = json.load(open(d + "/args.json")); dif = C.diff(json.load(open(ROOT + "/scratch/s238_args_frozen_%s.json" % ref)), new)
        o[tag] = {"rc": rc, "diff": dif, "args_file": d + "/args.json", "pass": rc == 0 and set(dif) <= der and {"init_model", "steps", "eval_parts"} <= set(dif)
                  and new["steps"] == 8000 and (new["hold_inc"] == 1.6 if ex == FH else (new["hold_inc"] == 0 and new["hold_log"] is True))}
    o["pass"] = all(v["pass"] for v in o.values()); return o


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True); os.makedirs(TMP, exist_ok=True)
    fns = {"flagsoff": flagsoff, "holdlog": holdlog, "detached": detached, "start": start, "smoke": smoke, "ftime": ftime, "mem": mem, "args": args_part}
    for part in [x for x in sys.argv[1:] if x in fns]:
        r = fns[part](); json.dump(r, open(os.path.join(OUT, part + ".json"), "w", newline="\n"), indent=1, default=str); print(part, "pass", r["pass"], flush=True)
