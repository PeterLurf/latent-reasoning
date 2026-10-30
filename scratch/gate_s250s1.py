"""§250 FRAME-DIV Stage 1 build gates (FD_v1 §4 + Note 250-1) -> scratch/gate_s250s1/<part>.json ("pass"). The 4060 (cuda:0) / CPU.
  args      --args_only fd_div_s1 / fd_div_s2 / fd_ctrl_s1 / fd_ctrl_s2: DIV vs CTRL per seed == {train_frames} exactly; writes scratch/s250_args_frozen_<run>.json.
  stream    FD_v1 §4 disclosure: the DIV (0-5, 8-11) and CTRL (FRAME_TRAIN) training streams (SEED_TRAIN, build_sets(2) excl, DF law), 20,000 draws (d cycling 1..4):
            the share of draws with identical programs (ops, src_j, values, spine names) - descriptive, reported (the frames differ by design).
  trainbit  --train_frames 0,1,2,3,4,5 (explicit) vs the default, det wrap, fp32, micro 8, stop_at 30, at the Stage-1 CTRL flags (from lh_fh_s1): ckpt / logs equal.
  hold0     --hold_start 0: a 20-step smoke at the DIV flags (micro 8, log 10): rc 9, the bin-0 hold record has inc_ratio and far_ratio set (penalty from step 1),
            gn_plain_mean100 logged, the step-0 norm-guard record present.
  f67       the frames 6 / 7 eval-only sets (built in part_frames): 1,000 per d 1..4, every op in that frame, 0 sigs in TEST2-DF / TRANSFER-DF / excl.
  evalbit   the §250-Stage-0 eval module (boxes/_inject_aw_eval_s250s0.py) vs the new one on parts ws, opclass, cue_only (lh_fh_s1): equal bar sec / module_sha256.
  smoke     scratch/s250_frames.py on df_plaw4_s1 (NOT a Stage-1 model): frames 6..15 x d 1..4 fields present.
    python -B scratch/gate_s250s1.py args stream trainbit hold0 f67 evalbit smoke"""
import json
import os
import shutil
import subprocess
import sys

ROOT = "D:/Latent"; sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/scratch"); os.chdir(ROOT)
import gate_s228 as G8  # noqa: E402
import gate_s237 as G7  # noqa: E402

OUT = ROOT + "/scratch/gate_s250s1"; TMP = ROOT + "/scratch/_tmp/s250s1"; RUNS = ROOT + "/boxes/runs_fsq"; DEV = "cuda:0"
EVAL = "acc,ws,opclass,swap_prod,swap_kd,lp5h,cue_only,lp5,cue_wrong,frames"; DIVF = "0,1,2,3,4,5,8,9,10,11"
INIT = {1: "boxes/runs_fsq/lh_fh_s1/model.pt", 2: "boxes/runs_fsq/lh_fh_s2/model.pt"}


def base(s, div):
    return ["--arm", "inject", "--think_k", "8", "--aux_step", "1", "--aux_fade", "48000,48000", "--aux_reweight", "--fast_add", "--cache_lower", "--d_max", "4",
            "--steps", "8000", "--cue", "step", "--write_src", "answer", "--init_trunk", "boxes/runs_fsq/s233_askname_s1/model.pt", "--init_sha", "c00f7ec0bd7977fe",
            "--write_mix_start", "0", "--write_mix_end", "0", "--mul_s_p", "2", "--init_model", INIT[s], "--init_model_sha", G7.sha(INIT[s])[:16],
            "--hold_inc", "1.6", "--hold_start", "0"] + (["--seed", "2"] if s == 2 else []) + (["--train_frames", DIVF] if div else [])


def full(s, div, run):
    return base(s, div) + ["--auto_micro", "--ckpt_every", "1000", "--eval_parts", EVAL, "--args_ref", "scratch/s250_args_frozen_%s.json" % run]


RUNSPEC = (("fd_div_s1", 1, True), ("fd_div_s2", 2, True), ("fd_ctrl_s1", 1, False), ("fd_ctrl_s2", 2, False))


def args_part():
    import s228_args_check as C
    o = {}; new = {}
    for run, s, div in RUNSPEC:
        fl = full(s, div, run); fl = fl[:fl.index("--args_ref")]
        d = G8.fresh(TMP + "/args_" + run); rc = G8.sh([sys.executable, "-B", "-m", "boxes.inject_aw"] + fl + ["--device", "cpu", "--args_only", "--out", d], d + "/stdout.log")
        new[run] = json.load(open(d + "/args.json")); o[run] = {"rc": rc, "train_frames": new[run].get("train_frames"), "hold_start": new[run].get("hold_start")}
    for s in (1, 2):
        o["div_vs_ctrl_s%d" % s] = sorted(C.diff(new["fd_ctrl_s%d" % s], new["fd_div_s%d" % s]))
    o["pass"] = all(o[r]["rc"] == 0 for r in new) and all(o["div_vs_ctrl_s%d" % s] == ["train_frames"] for s in (1, 2))
    if o["pass"]:
        for r, a in new.items():
            json.dump(a, open(ROOT + "/scratch/s250_args_frozen_%s.json" % r, "w", newline="\n"), indent=1)
    return o


def stream():
    import boxes.inject_ow as IO
    import boxes.ow_words as OW
    s = IO.build_sets(mul_s_p=2); fr = tuple(int(x) for x in DIVF.split(","))
    sd = OW.Stream(OW.SEED_TRAIN, s["excl"], frames=fr, mul_s_p=2, **IO.OWKW); sc = OW.Stream(OW.SEED_TRAIN, s["excl"], frames=OW.FRAME_TRAIN, mul_s_p=2, **IO.OWKW)
    same = 0; same_first = None; fr_div = set()
    for i in range(20000):
        a, b = sd.draw(1 + i % 4), sc.draw(1 + i % 4); fr_div.update(a["frames"])
        eq = a["ops"] == b["ops"] and a["src_j"] == b["src_j"] and a["values"] == b["values"] and a["names"]["spine"] == b["names"]["spine"]
        same += eq
        if not eq and same_first is None:
            same_first = i
    return {"draws": 20000, "identical_programs": same, "share": same / 20000, "first_divergence": same_first, "div_frames_seen": sorted(fr_div), "pass": True,
            "note": "descriptive (FD_v1 §4 disclosure): programs match until the frame draws' rng use differs"}


def det(mod, fl, d):
    G8.fresh(d); return G8.sh([sys.executable, "-B", "-m", "scratch.s222_det_wrap", mod] + fl + ["--device", DEV, "--out", d], d + "/stdout.log")


def trainbit():
    com = base(1, False) + ["--fp32", "--micro", "8", "--sched_scale", "0.02", "--ckpt_every", "30", "--log_every", "10", "--stop_at", "30"]
    da, db = TMP + "/tb_default", TMP + "/tb_explicit"; ra = det("boxes/inject_aw.py", com, da); rb = det("boxes/inject_aw.py", com + ["--train_frames", "0,1,2,3,4,5"], db); o = {"rc": [ra, rb]}
    if os.path.exists(da + "/ckpt_last.pt") and os.path.exists(db + "/ckpt_last.pt"):
        o.update(G8.ck_equal(da + "/ckpt_last.pt", db + "/ckpt_last.pt", keys=("model", "opt", "rng_d", "np_rng", "torch_rng", "cuda_rng", "stream_rng", "stream_stats", "gn_log")))
        st = lambda p: [{k: v for k, v in json.loads(l_).items() if k != "sec"} for l_ in open(p)]  # noqa: E731
        o["log_equal"] = st(da + "/log.jsonl") == st(db + "/log.jsonl")
    o["pass"] = o["rc"] == [9, 9] and o.get("n_mismatch") == 0 and bool(o.get("log_equal")); return o


def hold0():
    d = G8.fresh(TMP + "/hold0"); rc = G8.sh([sys.executable, "-B", "-m", "boxes.inject_aw"] + base(1, True) + ["--micro", "8", "--stop_at", "20", "--ckpt_every", "0",
                                             "--log_every", "10", "--device", DEV, "--out", d], d + "/stdout.log")
    lg = [json.loads(l_) for l_ in open(d + "/log.jsonl")] if os.path.exists(d + "/log.jsonl") else []
    bins = [r for r in lg if "by_d" in r]; hb = bins[-1].get("hold", {}) if bins else {}
    o = {"rc": rc, "step0": any(r.get("step") == 0 and "norm_guard" in r for r in lg), "gn_plain": [r.get("gn_plain_mean100") for r in lg if "gn_plain_mean100" in r],
         "bin_hold_keys": sorted(hb), "inc_ratio_set": hb.get("inc_ratio") is not None, "far_ratio_set": hb.get("far_ratio") is not None}
    o["pass"] = rc == 9 and o["step0"] and o["inc_ratio_set"] and o["far_ratio_set"] and len(o["gn_plain"]) > 0; return o


def f67():
    import boxes.inject_ow as IO
    import boxes.ow_words as OW
    s = IO.build_sets(mul_s_p=2); t2 = {it["sig"] for d in s["test2"] for it in s["test2"][d]}; tr = {it["sig"] for d in s["transfer"] for it in s["transfer"][d]}; o = {}
    for f in (6, 7):
        sf = OW.Stream(IO.SEED_FRAMESET + f, frames=(f,), mul_s_p=2, **IO.OWKW); fs = {d: [sf.draw(d) for _ in range(IO.TEST_N)] for d in IO.FRAMESET_DS}; sig = [it["sig"] for d in fs for it in fs[d]]
        o[str(f)] = {"n": len(sig), "frames_ok": all(set(it["frames"]) == {f} for d in fs for it in fs[d]), "in_test2": sum(x in t2 for x in sig), "in_transfer": sum(x in tr for x in sig),
                     "in_excl": sum(x in s["excl"] for x in sig), "sha": OW.set_sha([it for d in IO.FRAMESET_DS for it in fs[d]])[:12]}
    o["pass"] = all(o[f]["n"] == 4000 and o[f]["frames_ok"] and o[f]["in_test2"] == 0 and o[f]["in_transfer"] == 0 and o[f]["in_excl"] == 0 for f in ("6", "7")); return o


def tmp_run(src, name):
    d = os.path.join(TMP, name); shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
    for f in ("args.json", "train_summary.json"):
        shutil.copy(os.path.join(RUNS, src, f), d)
    os.link(os.path.join(RUNS, src, "model.pt"), os.path.join(d, "model.pt")); return d


def evalbit():
    parts = "ws,opclass,cue_only"; res = {}
    for mod, name in (("boxes._inject_aw_eval_s250s0", "eb_old"), ("boxes.inject_aw_eval", "eb_new")):
        d = tmp_run("lh_fh_s1", name)
        res[name] = (d, subprocess.run([sys.executable, "-B", "-m", mod, "--run", d, "--parts", parts, "--eval_bs", "250", "--device", DEV]).returncode)
    eq = {}
    for p_ in parts.split(","):
        x = [json.load(open(os.path.join(res[n][0], "eval_%s.json" % p_))) for n in ("eb_old", "eb_new")]
        for y in x:
            y.pop("sec", None); y.pop("module_sha256", None)
        eq[p_] = x[0] == x[1]
    for n in res:
        os.remove(os.path.join(res[n][0], "model.pt"))
    return {"rc": {k: v[1] for k, v in res.items()}, "equal": eq, "pass": all(v[1] == 0 for v in res.values()) and all(eq.values())}


def smoke():
    d = tmp_run("df_plaw4_s1", "smoke")
    rc = subprocess.run([sys.executable, "-B", "scratch/s250_frames.py", "--run", "df_plaw4_s1", "--src_dir", d, "--out", d, "--eval_bs", "250", "--device", DEV]).returncode
    os.remove(os.path.join(d, "model.pt")); o = {"rc": rc}; f = os.path.join(d, "eval_frames.json")
    if rc == 0 and os.path.exists(f):
        e = json.load(open(f)); o["frames"] = sorted((k for k in e if k.isdigit()), key=int)
        o["fields_ok"] = all(all(set(e[x][str(dd)]) >= {"n", "acc", "ws", "wrap_any_share"} and e[x][str(dd)]["n"] == 1000 for dd in range(1, 5)) for x in o["frames"])
        o["sec"] = e.get("sec"); o["frameset_shas_67"] = {k: e["frameset_shas"][k][:12] for k in ("6", "7")}
    o["pass"] = rc == 0 and o.get("frames") == [str(x) for x in range(6, 16)] and o.get("fields_ok"); return o


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True); os.makedirs(TMP, exist_ok=True)
    fns = {"args": args_part, "stream": stream, "trainbit": trainbit, "hold0": hold0, "f67": f67, "evalbit": evalbit, "smoke": smoke}
    for part in [x for x in sys.argv[1:] if x in fns]:
        r = fns[part](); json.dump(r, open(os.path.join(OUT, part + ".json"), "w", newline="\n"), indent=1, default=str); print(part, "pass", r["pass"], flush=True)
