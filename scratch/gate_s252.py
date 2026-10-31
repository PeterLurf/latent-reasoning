"""§252 UNSCAFFOLDED build gates (NS_v1 §3; Note 252-1) -> scratch/gate_s252/<part>.json ("pass"). The 4060 (cuda:0); det wrap + fp32 + micro 8 for the bitwise parts.
References: boxes/_inject_aw_s251ac.py (the trainer before E252 / the GEN-CUE head), boxes/_inject_aw_eval_s251cc.py (the eval module before §252).
Flags = rcb_plaw4_s1's recipe (gate_s251.RCB); the pilot's init (rcb_plaw4_s1/model.pt) where a part says so.
  offbit     trainer defaults: _inject_aw_s251ac.py vs the new module, det, 100 steps: ckpt (model / opt / rngs / stream) equal, logs equal.
  perm_tiny  --perm_p 1e-12 (the split-WS path armed: pass_aw + the no_grad readout, no item ever permuted) vs defaults, det, 30 steps: ckpt (model / opt / rngs / stream)
             equal (the readout never feeds back), the log records equal except the new ws_readout_* keys.
  acts       --perm_p 0.5 vs defaults, det, stop 3 (seed 1 opens with d >= 2 steps): rng_d / stream_rng / stream_stats equal (the item stream untouched), model differs.
  perm10k    10,000 REG training-law items (OW.Stream(25170, the run's EXCL), d 2-4 cycling): perm_render.train_perm (solve == answer asserted inside), >= 1 inversion,
             S_0 / G_t / D_k slots unchanged, the op-sentence slots hold a permutation of S_1..S_d; no text equals a TEST-PERM text (TEST2 d 1-4, 25150 + d).
  gencue     tok_aw(cue generic) on TEST2 d 1-4 (4,000): every pass input identical, == [EOT] + enc(' ' + text + GEN_CUE) + SFX, the cue 6 tokens (+ SFX), the text rows
             == the named items' think rows, the decoder == the named decoder; the PLAN-FREE assert fires on a generic-flagged batch whose pass inputs differ (negative test)
             and stays silent on the real generic batch (positive).
  evaloff    the eval module before / after §252 on a rcb_plaw4_s1 copy (named cue), parts acc,c7: every eval_<part>.json equal except the run-meta timing keys.
  evalgp     the new eval module on a GP-args copy of rcb_plaw4_s1 (cue generic), parts acc,ws,c7,transfer,test_perm,lp5h,swap_kd,cue_only,cue_wrong: rc 0, every part
             but cue_only / cue_wrong written (dropped); read_mask on GP items runs (its asserts) and passes() under it.
  smoke      P-GP / P-NP flags from the pilot init, the plain path, log every step, stop 40: rc 9, every loss finite, the stop bin carries ws_readout_perm / _chain with
             d >= 2 entries and n > 0.
  gradflow   gate_s251's instrumented probe on the new module, P-GP flags, det, stop 8: the answer-CE grad nonzero into the write (mem.qkv / gate / ln_w) and read
             (mem.wo) params on every d >= 2 call after step 1.
  args       --args_only for P-NP / P-GP vs rcb_plaw4_s1/args.json (C.diff): NP {perm_p, init_model(_sha)} + module keys; GP + cue.
    python -B scratch/gate_s252.py offbit perm_tiny acts perm10k gencue evaloff evalgp smoke gradflow args"""
import json
import math
import os
import shutil
import subprocess
import sys

import numpy as np

ROOT = "D:/Latent"; sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/scratch"); os.chdir(ROOT)
import gate_s228 as G8  # noqa: E402
import gate_s251 as G1  # noqa: E402
import s228_args_check as C  # noqa: E402

OUT = ROOT + "/scratch/gate_s252"; TMP = ROOT + "/scratch/_tmp/s252"; DEV = "cuda:0"
RCB = list(G1.RCB); KEYS = G1.KEYS; COM = ["--fp32", "--micro", "8"]
INIT = ["--init_model", "boxes/runs_fsq/rcb_plaw4_s1/model.pt", "--init_model_sha", "67775e2492044a8a"]


def pilot_flags():
    fl = list(RCB); i = fl.index("--init_model"); fl[i + 1] = INIT[1]; j = fl.index("--init_model_sha"); fl[j + 1] = INIT[3]; return fl


def det(mod, fl, d):
    G8.fresh(d); return G8.sh([sys.executable, "-B", "-m", "scratch.s222_det_wrap", mod] + fl + ["--device", DEV, "--out", d], d + "/stdout.log")


recs = lambda p: [json.loads(l_) for l_ in open(p)] if os.path.exists(p) else []  # noqa: E731
strip = lambda p, drop=(): [{k: v for k, v in r.items() if k != "sec" and k not in drop} for r in recs(p)]  # noqa: E731


def offbit():
    fl = RCB + COM + ["--ckpt_every", "100", "--log_every", "10", "--stop_at", "100"]
    da, db = TMP + "/off_ref", TMP + "/off_new"; o = {"rc": [det("boxes/_inject_aw_s251ac.py", fl, da), det("boxes/inject_aw.py", fl, db)]}
    if os.path.exists(da + "/ckpt_last.pt") and os.path.exists(db + "/ckpt_last.pt"):
        o.update(G8.ck_equal(da + "/ckpt_last.pt", db + "/ckpt_last.pt", keys=KEYS)); o["log_equal"] = strip(da + "/log.jsonl") == strip(db + "/log.jsonl")
    o["pass"] = o["rc"] == [9, 9] and o.get("n_mismatch") == 0 and bool(o.get("log_equal")); return o


def perm_tiny():
    fl = RCB + COM + ["--ckpt_every", "30", "--log_every", "10", "--stop_at", "30"]
    da, db = TMP + "/tiny_off", TMP + "/tiny_on"; o = {"rc": [det("boxes/inject_aw.py", fl, da), det("boxes/inject_aw.py", fl + ["--perm_p", "1e-12"], db)]}
    if os.path.exists(da + "/ckpt_last.pt") and os.path.exists(db + "/ckpt_last.pt"):
        o.update(G8.ck_equal(da + "/ckpt_last.pt", db + "/ckpt_last.pt", keys=KEYS))
        nk = ("ws_readout_perm", "ws_readout_perm_n", "ws_readout_chain", "ws_readout_chain_n")
        la = strip(da + "/log.jsonl"); lb = strip(db + "/log.jsonl", nk); o["log_equal_minus_new_keys"] = la == lb
        o["new_keys_present"] = any("ws_readout_chain" in r for r in recs(db + "/log.jsonl"))
        o["args_diff"] = sorted(k for k, v in C.diff(json.load(open(da + "/args.json")), json.load(open(db + "/args.json"))).items() if k != "pair_stats")
    o["pass"] = o["rc"] == [9, 9] and o.get("n_mismatch") == 0 and bool(o.get("log_equal_minus_new_keys")) and bool(o.get("new_keys_present")); return o


def acts():
    fl = RCB + COM + ["--ckpt_every", "3", "--log_every", "1", "--stop_at", "3"]
    da, db = TMP + "/acts_off", TMP + "/acts_on"; o = {"rc": [det("boxes/inject_aw.py", fl, da), det("boxes/inject_aw.py", fl + ["--perm_p", "0.5"], db)]}
    if os.path.exists(da + "/ckpt_last.pt") and os.path.exists(db + "/ckpt_last.pt"):
        o["stream"] = G8.ck_equal(da + "/ckpt_last.pt", db + "/ckpt_last.pt", keys=("rng_d", "stream_rng", "stream_stats")); o["model"] = G8.ck_equal(da + "/ckpt_last.pt", db + "/ckpt_last.pt", keys=("model",))
    o["pass"] = o["rc"] == [9, 9] and o.get("stream", {}).get("n_mismatch") == 0 and (o.get("model", {}).get("n_mismatch") or 0) > 0; return o


def perm10k():
    import boxes.ow_words as OW
    import boxes.perm_render as PR
    from boxes import inject_ow as IO
    sets = IO.build_sets(mul_s_p=0); tp, _ = PR.test_perm_sets({d: sets["test2"][d] for d in (1, 2, 3, 4)}, (1, 2, 3, 4)); tpt = {it["text"] for v in tp.values() for it in v}
    s = OW.Stream(25170, sets["excl"], frames=OW.FRAME_TRAIN, mul_s_p=0, **IO.OWKW); n = 0; bad = {"inv": 0, "slots": 0, "tp": 0}; invs = []
    for i in range(10000):
        d = 2 + i % 3; it = s.draw(d); pit = PR.train_perm(it, 1, i, 0); n += 1
        ok_slots = all((x == y) if not (x.startswith("S") and x != "S0") else y.startswith("S") and y != "S0" for x, y in zip(it["order"], pit["order"]))
        bad["slots"] += not ok_slots or sorted(x for x in pit["order"] if x.startswith("S") and x != "S0") != sorted("S%d" % t for t in range(1, d + 1))
        o_ = [int(x[1:]) for x in pit["order"] if x.startswith("S") and x != "S0"]; inv = sum(int(o_[a_] > o_[b_]) for a_ in range(d) for b_ in range(a_ + 1, d))
        bad["inv"] += inv < 1; invs.append(inv); bad["tp"] += pit["text"] in tpt
    o = {"n": n, "bad": bad, "mean_inversions": float(np.mean(invs)), "solve_asserted_inside": True}
    o["pass"] = n == 10000 and not any(bad.values()); return o


def gencue():
    import torch
    from boxes import inject_aw as AW, inject_aw_eval as E, inject_ow as IO
    a = json.load(open("boxes/runs_fsq/rcb_plaw4_s1/args.json")); sets = IO.build_sets(mul_s_p=E.law(a)); o = {"n": 0, "bad": 0}
    gt = IO.ENC.encode(AW.GEN_CUE); o["cue_tokens"] = len(gt)
    for d in (1, 2, 3, 4):
        for it in sets["test2"][d]:
            g_ = AW.tok_aw(it, a["aux_target"], "generic", "answer"); n_ = AW.tok_aw(it, a["aux_target"], "step", "answer"); o["n"] += 1
            q0 = [AW.EOT] + IO.ENC.encode(" " + it["text"] + AW.GEN_CUE) + AW.SFX
            o["bad"] += not (all(q == q0 for q in g_["think_cues"]) and len(g_["think_cues"]) == d and g_["think"] == n_["think"] and g_["dec"] == n_["dec"] and g_.get("cue") == "generic"
                             and q0[len(n_["think"]):] == gt + AW.SFX)
    _, m, g = E.load(ROOT + "/boxes/runs_fsq/rcb_plaw4_s1", DEV); its = [AW.tok_aw(it, a["aux_target"], "generic", "answer") for it in sets["test2"][3][:20]]
    with torch.no_grad():
        E.passes(m, its, DEV, 20, {3}); o["planfree_positive_silent"] = True
        bad_its = [dict(x) for x in its]; q1 = list(bad_its[0]["think_cues"][1]); j_ = len(q1) - len(AW.SFX) - 2; q1[j_] = IO.ENC.encode(" How")[0] if q1[j_] != IO.ENC.encode(" How")[0] else IO.ENC.encode(" Why")[0]
        bad_its[0]["think_cues"] = [bad_its[0]["think_cues"][0], q1, bad_its[0]["think_cues"][2]]   # pass 2's input: one cue token changed, the same length (padding unchanged)
        try:
            E.passes(m, bad_its, DEV, 20, {3}); o["planfree_negative_fires"] = False
        except AssertionError as e:
            o["planfree_negative_fires"] = "PLAN-FREE" in str(e); o["planfree_negative_msg"] = str(e)[:120]
    o["pass"] = o["n"] == 4000 and o["bad"] == 0 and o["cue_tokens"] == 6 and o["planfree_positive_silent"] and o["planfree_negative_fires"] is True; return o


def run_copy(d, cue=None):
    G8.fresh(d); a = json.load(open("boxes/runs_fsq/rcb_plaw4_s1/args.json"))
    if cue:
        a["cue"] = cue
    json.dump(a, open(d + "/args.json", "w", newline="\n"), indent=1); os.link("boxes/runs_fsq/rcb_plaw4_s1/model.pt", d + "/model.pt")
    shutil.copy("boxes/runs_fsq/rcb_plaw4_s1/train_summary.json", d + "/train_summary.json")


def eval_run(mod, d, parts):
    return G8.sh([sys.executable, "-B", "-m", mod, "--run", d, "--parts", parts, "--eval_bs", "100", "--device", DEV], d + "/eval_stdout.log")


def evaloff():
    da, db = TMP + "/ev_old", TMP + "/ev_new"; run_copy(da); run_copy(db); parts = "acc,c7"
    o = {"rc": [eval_run("boxes._inject_aw_eval_s251cc", da, parts), eval_run("boxes.inject_aw_eval", db, parts)]}; eq = {}
    for p_ in parts.split(","):
        fa, fb = da + "/eval_%s.json" % p_, db + "/eval_%s.json" % p_
        if os.path.exists(fa) and os.path.exists(fb):
            ja, jb = json.load(open(fa)), json.load(open(fb)); dr = lambda j: {k: v for k, v in j.items() if k not in ("sec", "eval_sec", "time", "wall_sec")}  # noqa: E731
            eq[p_] = dr(ja) == dr(jb); eq[p_ + "_keys_differ"] = sorted(k for k in set(ja) | set(jb) if ja.get(k) != jb.get(k))
    o["equal"] = eq; o["pass"] = o["rc"] == [0, 0] and all(eq.get(p_) for p_ in parts.split(",")); return o


def evalgp():
    import torch
    from boxes import inject_aw_eval as E, inject_ow as IO
    d = TMP + "/ev_gp"; run_copy(d, "generic"); parts = "acc,ws,c7,transfer,test_perm,lp5h,swap_kd,cue_only,cue_wrong"
    o = {"rc": eval_run("boxes.inject_aw_eval", d, parts)}; o["written"] = {p_: os.path.exists(d + "/eval_%s.json" % p_) for p_ in parts.split(",")}
    a = json.load(open(d + "/args.json")); _, m, g = E.load(d, DEV); T = E.tokd(IO.build_sets(mul_s_p=E.law(a)), a, (2, 4))
    with torch.no_grad():
        its = T[4][:50]
        with E.read_mask(m, its) as st:
            E.passes(m, its, DEV, 50, {4}); o["read_mask_stats"] = dict(st)
    o["pass"] = (o["rc"] == 0 and all(o["written"][p_] for p_ in ("acc", "ws", "c7", "transfer", "test_perm", "lp5h", "swap_kd"))
                 and not o["written"]["cue_only"] and not o["written"]["cue_wrong"] and o["read_mask_stats"]["calls_pass"] > 0); return o


def smoke():
    o = {}
    for arm, ex in (("gp", ["--perm_p", "0.5", "--cue", "generic"]), ("np", ["--perm_p", "0.5"])):
        d = TMP + "/smoke_" + arm; fl = pilot_flags() + ["--micro", "8"] + ex + ["--ckpt_every", "0", "--log_every", "1", "--stop_at", "40"]
        G8.fresh(d); rc = G8.sh([sys.executable, "-B", "-m", "boxes.inject_aw"] + fl + ["--device", DEV, "--out", d], d + "/stdout.log")
        rs = recs(d + "/log.jsonl"); ls = [r["loss"] for r in rs if isinstance(r.get("step"), int) and r.get("loss") is not None]
        bins = [r for r in rs if "ws_readout_perm" in r]; b = bins[-1] if bins else {}
        o[arm] = {"rc": rc, "n_loss": len(ls), "finite": bool(ls) and all(math.isfinite(x) for x in ls), "perm_n": b.get("ws_readout_perm_n"), "chain_n": b.get("ws_readout_chain_n")}
        o[arm]["ok"] = (rc == 9 and o[arm]["finite"] and len(ls) >= 40 and bool(b) and any(int(k) >= 2 for k in (b.get("ws_readout_perm") or {}))
                        and all(n_ > 0 for dv in (b.get("ws_readout_perm_n") or {}).values() for n_ in dv.values()))
    o["pass"] = all(o[k]["ok"] for k in ("gp", "np")); return o


def gradflow():
    src = open(ROOT + "/boxes/inject_aw.py", encoding="utf-8").read(); anchor = "            loss = la + lam_now[0] * aux\n"; assert src.count(anchor) == 1
    open(ROOT + "/boxes/_inject_aw_gradprobe_s252.py", "w", encoding="utf-8", newline="\n").write(src.replace(anchor, anchor + G1.PROBE.lstrip("\n")))
    d = TMP + "/gradflow"; G8.fresh(d); lp = d + "/gradprobe.jsonl"; G8.ENV["GRADPROBE_OUT"] = lp
    fl = pilot_flags() + COM + ["--perm_p", "0.5", "--cue", "generic", "--ckpt_every", "0", "--log_every", "1", "--stop_at", "8"]
    rc = G8.sh([sys.executable, "-B", "-m", "scratch.s222_det_wrap", "boxes/_inject_aw_gradprobe_s252.py"] + fl + ["--device", DEV, "--out", d], d + "/stdout.log")
    G8.ENV.pop("GRADPROBE_OUT", None); rs = recs(lp); per = 8
    rows = [dict(r, step=1 + i // per) for i, r in enumerate(rs)]; el = [r for r in rows if r["d"] >= 2 and r["step"] >= 2]
    req = ("qkv.weight", "gate.weight", "ln_w.weight", "wo.weight")
    o = {"rc": rc, "n_calls": len(rs), "steps_d": sorted({(r["step"], r["d"]) for r in rows}), "n_eligible": len(el),
         "eligible_la_mem_first": el[0]["la_grad_mem"] if el else None, "all_nonzero": bool(el) and all(all((r["la_grad_mem"].get(k) or 0) > 0 for k in req) for r in el)}
    o["pass"] = rc == 9 and o["all_nonzero"]; return o


def args():
    ref = json.load(open("boxes/runs_fsq/rcb_plaw4_s1/args.json")); o = {}
    for arm, ex in (("np", ["--perm_p", "0.5"]), ("gp", ["--perm_p", "0.5", "--cue", "generic"])):
        d = G8.fresh(TMP + "/args_" + arm); rc = G8.sh([sys.executable, "-B", "-m", "boxes.inject_aw"] + pilot_flags() + ex + ["--micro", "8", "--stop_at", "2000", "--ckpt_every", "1000",
                                                                                             "--log_every", "100", "--device", "cpu", "--args_only", "--out", d], d + "/stdout.log")
        new = json.load(open(d + "/args.json")); dif = {k: v for k, v in C.diff(ref, new).items() if k != "pair_stats"}; o[arm] = {"rc": rc, "diff": dif, "keys": sorted(dif)}
    o["pass"] = all(o[k]["rc"] == 0 for k in ("np", "gp")) and "perm_p" in o["np"]["keys"] and "cue" not in o["np"]["keys"] and {"perm_p", "cue"} <= set(o["gp"]["keys"]); return o


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True); os.makedirs(TMP, exist_ok=True)
    fns = {k: globals()[k] for k in ("offbit", "perm_tiny", "acts", "perm10k", "gencue", "evaloff", "evalgp", "smoke", "gradflow", "args")}
    for part in [x for x in sys.argv[1:] if x in fns]:
        r = fns[part](); json.dump(r, open(os.path.join(OUT, part + ".json"), "w", newline="\n"), indent=1, default=str); print(part, "pass", r["pass"], flush=True)
