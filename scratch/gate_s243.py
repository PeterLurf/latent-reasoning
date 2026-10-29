"""§243 DEPTH-CURRIC build gates (DC_v1 §2-3, §7; Audit G-D additions) -> scratch/gate_s243/<part>.json ("pass"). The 3070 (cuda:1; the 4060 runs the pilot) / CPU.
Reference kept: boxes/_inject_aw_s240.py (the §240 trainer), boxes/_inject_aw_eval_s242.py (the §242 eval module). INIT: boxes/runs_fsq/lh_fh_s1/model.pt (and _s2).
  args       --args_only dc_s1 / dc_s2 / dc_ctrl_s1 / dc_ctrl_s2 (CPU): DC vs CTRL per seed differs ONLY in {d_max} + keys derived from it (listed); each vs the §240 frozen
             lh_fh_s<n>: {think_k, init_model, init_model_sha, init_model_sha256, eval_parts} + derived. Writes scratch/s243_args_frozen_<run>.json.
  trainbit   flags-off: the §240 trainer vs the new one at the DC shape (d_max 6, think_k 10, init lh_fh_s1), det wrap, fp32, micro 32, sched 0.02, stop_at 30: (a) plain (no hold)
             and (b) --hold_inc 1.6 (penalty from step 10): ckpt fields torch.equal; logs equal after removing hold.norm_guard_by_d (b: present in every bin, d keys 1..6 seen).
  mem        --auto_micro --time_steps 5 (mem_budget 7.0; the micro search) at the DC shape (d 6 items, K 10, penalty + the far pass at 40 inside the peak) and the CTRL shape, on the 3070; the
             3090 decides at K7 (budget 21) - the per-chunk peak is reported for the micro-4 projection.
  long       run_long smoke (Audit G-D 4): part_long on the 250 LONGEST TEST2-DF d 12 items, K' 8,10,16,64,128, eval_bs 250, lh_fh_s1 -> rc / peak GiB / sec (no acc kept).
  evalbit    the §242 eval module vs the new one on parts ws, opclass, cue_only (lh_fh_s1): equal bar sec / module_sha256; + a ws_long smoke on a lh_fh_s1 copy: fields
             {sig, pred_by_pass 1..10, v (N x 10)} for d 1..10, n 1000 (no values kept), eval_bs 100 (8 GB).
    python -B scratch/gate_s243.py args trainbit mem long evalbit"""
import json
import math
import os
import shutil
import subprocess
import sys

ROOT = "D:/Latent"; sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/scratch"); os.chdir(ROOT)
import gate_s228 as G8  # noqa: E402
import gate_s237 as G7  # noqa: E402

OUT = ROOT + "/scratch/gate_s243"; TMP = ROOT + "/scratch/_tmp/s243"; DEV = "cuda:1"; RUNS = ROOT + "/boxes/runs_fsq"
DC_EVAL = "acc,ws,opclass,swap_prod,swap_kd,lp5h,cue_only,lp5,cue_wrong,ws_long"
INIT = {1: "boxes/runs_fsq/lh_fh_s1/model.pt", 2: "boxes/runs_fsq/lh_fh_s2/model.pt"}


def base(s, d_max):
    return ["--arm", "inject", "--think_k", "10", "--aux_step", "1", "--aux_fade", "48000,48000", "--aux_reweight", "--fast_add", "--cache_lower", "--d_max", str(d_max),
            "--steps", "8000", "--cue", "step", "--write_src", "answer", "--init_trunk", "boxes/runs_fsq/s233_askname_s1/model.pt", "--init_sha", "c00f7ec0bd7977fe",
            "--write_mix_start", "0", "--write_mix_end", "0", "--mul_s_p", "2", "--init_model", INIT[s], "--init_model_sha", G7.sha(INIT[s])[:16], "--hold_inc", "1.6"] \
        + (["--seed", "2"] if s == 2 else [])


def full(s, d_max, run):
    return base(s, d_max) + ["--auto_micro", "--ckpt_every", "1000", "--eval_parts", DC_EVAL, "--args_ref", "scratch/s243_args_frozen_%s.json" % run]


def args_part():
    import s228_args_check as C
    o = {}; new = {}
    for run, s, dm in (("dc_s1", 1, 6), ("dc_s2", 2, 6), ("dc_ctrl_s1", 1, 4), ("dc_ctrl_s2", 2, 4)):
        fl = full(s, dm, run); fl = fl[:fl.index("--args_ref")]                # the frozen file is made here (as §240: args_ref "" in the frozen json)
        d = G8.fresh(TMP + "/args_" + run); rc = G8.sh([sys.executable, "-B", "-m", "boxes.inject_aw"] + fl + ["--device", "cpu", "--args_only", "--out", d], d + "/stdout.log")
        new[run] = json.load(open(d + "/args.json")); o[run] = {"rc": rc}
        fr = json.load(open(ROOT + "/scratch/s240_args_frozen_lh_fh_s%d.json" % s)); o[run]["diff_vs_lh_fh"] = C.diff(fr, new[run])
    for s in (1, 2):
        dif = C.diff(new["dc_ctrl_s%d" % s], new["dc_s%d" % s]); o["dc_vs_ctrl_s%d" % s] = {"diff": dif, "keys": sorted(dif)}
    allowed_dc = {"d_max"}; derived = {"excl_n", "pair_stats", "s226_test_sha"}            # derived = keys computed from d_max by the trainer (reported, not flags)
    o["pass"] = (all(o[r]["rc"] == 0 for r in new) and all(set(o["dc_vs_ctrl_s%d" % s]["keys"]) - derived == allowed_dc for s in (1, 2))
                 and all(set(o[r]["diff_vs_lh_fh"]) - derived - {"d_max"} <= {"think_k", "init_model", "init_model_sha", "init_model_sha256", "eval_parts", "args_ref", "args_only"}
                         for r in new))
    if o["pass"]:
        for r, a in new.items():
            json.dump(a, open(ROOT + "/scratch/s243_args_frozen_%s.json" % r, "w", newline="\n"), indent=1)
    return o


def det(mod, fl, d):
    G8.fresh(d); return G8.sh([sys.executable, "-B", "-m", "scratch.s222_det_wrap", mod] + fl + ["--device", DEV, "--out", d], d + "/stdout.log")


KEYS = ("model", "opt", "rng_d", "np_rng", "torch_rng", "cuda_rng", "stream_rng", "stream_stats", "expo", "binacc", "vbin", "vbin_d", "gnbin", "gn_log", "dseq", "iseq")


def strip_ng(p):
    out = []
    for l_ in open(p):
        r = {k: v for k, v in json.loads(l_).items() if k != "sec"}
        if isinstance(r.get("hold"), dict):
            r["hold"] = {k: v for k, v in r["hold"].items() if k != "norm_guard_by_d"}
        out.append(r)
    return out


def trainbit():
    o = {}
    for tag, ex in (("plain", ["--hold_inc", "0"]), ("hold", [])):
        com = base(1, 6) + ex + ["--fp32", "--micro", "32", "--sched_scale", "0.02", "--ckpt_every", "30", "--log_every", "10", "--stop_at", "30"]
        da, db = TMP + "/tb_s240mod_" + tag, TMP + "/tb_newmod_" + tag
        ra = det("boxes/_inject_aw_s240.py", com, da); rb = det("boxes/inject_aw.py", com, db); r = {"rc": [ra, rb]}
        if os.path.exists(da + "/ckpt_last.pt") and os.path.exists(db + "/ckpt_last.pt"):
            r.update(G8.ck_equal(da + "/ckpt_last.pt", db + "/ckpt_last.pt", keys=KEYS + (("hbin", "gn_plain_log", "rng_far") if tag == "hold" else ())))
            r["log_equal_minus_ng_by_d"] = strip_ng(da + "/log.jsonl") == strip_ng(db + "/log.jsonl")
            lb = [json.loads(l_) for l_ in open(db + "/log.jsonl")]; bins = [x for x in lb if "by_d" in x]
            ngd = [x.get("hold", {}).get("norm_guard_by_d") for x in bins]
            r["ng_by_d_bins"] = len([x for x in ngd if x]); r["ng_by_d_keys"] = sorted({k for x in ngd if x for k in x}); r["n_bins"] = len(bins)
        ok_ng = (tag == "plain" and r.get("ng_by_d_bins") == 0) or (tag == "hold" and r.get("ng_by_d_bins") == r.get("n_bins", -1) and r.get("n_bins", 0) > 0)
        r["pass"] = r["rc"] == [9, 9] and r.get("n_mismatch") == 0 and bool(r.get("log_equal_minus_ng_by_d")) and ok_ng; o[tag] = r
    o["pass"] = all(v["pass"] for v in o.values()); return o


def mem():
    o = {}
    for tag, dm in (("dc", 6), ("ctrl", 4)):
        d = G8.fresh(TMP + "/mem_" + tag)
        rc = G8.sh([sys.executable, "-B", "-m", "boxes.inject_aw"] + base(1, dm) + ["--auto_micro", "--mem_budget", "7.0", "--time_steps", "5", "--device", DEV, "--out", d], d + "/stdout.log")
        o[tag] = {"rc": rc, **({"mem_check": json.load(open(d + "/mem_check.json"))} if os.path.exists(d + "/mem_check.json") else {})}
    o["pass"] = all(o[k]["rc"] == 0 and "mem_check" in o[k] and math.isfinite(o[k]["mem_check"].get("peak_gib_train", math.inf)) for k in ("dc", "ctrl")); return o


LONG_CODE = r'''
import sys, json, time, torch
sys.path.insert(0, ROOT_)
from boxes import inject_aw_eval as E, inject_ow as IO
dev = DEV_; a, m, g = E.load(RUN_, dev); s = IO.build_sets(mul_s_p=E.law(a))
T12 = sorted(s["test2"][12], key=lambda it: -len(E.seq_std(E.tokd({"test2": {12: [it]}}, a, [12])[12][0], 1)))[:250]
torch.cuda.reset_peak_memory_stats(dev); t1 = time.time()
r = E.part_long(m, g, {"test2": {12: T12}}, a, dev, 250, [8, 10, 16, 64, 128], [12])
L = [len(E.seq_std(x, 1)) for x in E.tokd({"test2": {12: T12}}, a, [12])[12]]
print("LONGJSON " + json.dumps({"n": len(T12), "tok_max": max(L), "tok_min": min(L), "peak_gib": torch.cuda.max_memory_allocated(dev) / 2 ** 30, "sec": round(time.time() - t1),
                                "nonfinite": r["by_d"]["12"]["nonfinite"], "keys": sorted(r["by_d"]["12"])}))
'''


def long_part():
    code = LONG_CODE.replace("ROOT_", repr(ROOT)).replace("DEV_", repr(DEV)).replace("RUN_", repr(RUNS + "/lh_fh_s1"))
    p = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True); ln = [x for x in p.stdout.splitlines() if x.startswith("LONGJSON ")]
    o = json.loads(ln[-1][9:]) if ln else {"stderr": p.stderr[-2000:]}; o["rc"] = p.returncode
    o["pass"] = p.returncode == 0 and o.get("n") == 250 and o.get("peak_gib", 99) < 7.5; return o


def tmp_run(src, name):
    d = os.path.join(TMP, name); shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
    for f in ("args.json", "train_summary.json"):
        shutil.copy(os.path.join(RUNS, src, f), d)
    os.link(os.path.join(RUNS, src, "model.pt"), os.path.join(d, "model.pt")); return d


def evalbit():
    parts = "ws,opclass,cue_only"; res = {}
    for mod, name in (("boxes._inject_aw_eval_s242", "eb_old"), ("boxes.inject_aw_eval", "eb_new")):
        d = tmp_run("lh_fh_s1", name)
        res[name] = (d, subprocess.run([sys.executable, "-B", "-m", mod, "--run", d, "--parts", parts, "--eval_bs", "250", "--device", DEV]).returncode)
    eq = {}
    for p_ in parts.split(","):
        x = [json.load(open(os.path.join(res[n][0], "eval_%s.json" % p_))) for n in ("eb_old", "eb_new")]
        for y in x:
            y.pop("sec", None); y.pop("module_sha256", None)
        eq[p_] = x[0] == x[1]
    d = res["eb_new"][0]; rc = subprocess.run([sys.executable, "-B", "-m", "boxes.inject_aw_eval", "--run", d, "--parts", "ws_long", "--eval_bs", "100", "--device", DEV]).returncode   # bs 100 on the 8 GB 3070 (bs 250 OOMs at d 10; the host 3090 has 24 GB)
    w = {}
    if rc == 0:
        e = json.load(open(os.path.join(d, "eval_ws_long.json"))); it = e["items"]
        w = {"ds": sorted(it, key=int), "n": {k: len(v["sig"]) for k, v in it.items()}, "passes": sorted(it["10"]["pred_by_pass"], key=int),
             "v_shape_ok": all(len(v["v"]) == len(v["sig"]) and all(len(r) == 10 for r in v["v"]) for v in it.values()), "sec": e.get("sec"), "eval_gpu": e.get("eval_gpu")}
    for n in res:
        os.remove(os.path.join(res[n][0], "model.pt"))
    o = {"rc": {k: v[1] for k, v in res.items()}, "equal": eq, "ws_long_rc": rc, "ws_long": w}
    o["pass"] = (all(v[1] == 0 for v in res.values()) and all(eq.values()) and rc == 0 and w.get("ds") == [str(x) for x in range(1, 11)]
                 and all(x == 1000 for x in w["n"].values()) and w.get("passes") == [str(x) for x in range(1, 11)] and w.get("v_shape_ok")); return o


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True); os.makedirs(TMP, exist_ok=True)
    fns = {"args": args_part, "trainbit": trainbit, "mem": mem, "long": long_part, "evalbit": evalbit}
    for part in [x for x in sys.argv[1:] if x in fns]:
        r = fns[part](); json.dump(r, open(os.path.join(OUT, part + ".json"), "w", newline="\n"), indent=1, default=str); print(part, "pass", r["pass"], flush=True)
