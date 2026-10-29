"""§244 ANSWER-ONLY build gates (AO_v1 §3, §7; Note 244-1) -> scratch/gate_s244/<part>.json ("pass"). The 4060 (cuda:0), det wrap, fp32 / CPU.
References: boxes/_inject_aw_p2.py (the P-2 module, sha 9d59733d = aw_plaw2_s1's module_sha256), boxes/_inject_aw_s243.py (HEAD before E244).
  p2bit   the P-2 module vs the current one at aw_plaw2_s1's flags (gen_q_b347), 100 steps (real schedule), micro 8: ckpt model / opt / rngs / stream equal, logs equal
          (bar sec). The host replay of the logged step-100 loss (3.3319097250699996, RTX 3090) is the b366 queue's first line (aw_plaw2_s1_rep_gtime).
  lam     (a) lam_at (its own source, ast-extracted from boxes/inject_aw.py) under --aux_fade 0,0: 1.0 at step 0, 0.0 at every step 1..18000 (training starts at step 1);
          (b) the AO flags, det, 100 steps, --log_every 1: aux_lambda == 0.0 in every step record 1..100 and aux_ce finite;
          (c) gradflow: a gate-only copy (boxes/_inject_aw_noaux_s244.py: loss = la, the aux term removed) vs the AO module at the AO flags after 1 and after 100
          updates: ckpt model + opt torch.equal -> the aux term's gradient is exactly zero on every parameter at every update.
  keep    --keep_final_ckpt (20-step full run, no eval parts): ckpt_final.pt holds step 20, no ckpt_last.pt; without the flag neither file remains.
  args    --args_only ao_s1 / ao_s2 / aux_s2: ao_s2 vs aux_s2 = {aux_fade}; aux_s2 vs aw_plaw2_s1/args.json = {seed} + keys the later modules added (listed);
          writes scratch/s244_args_frozen_<run>.json.
    python -B scratch/gate_s244.py p2bit lam keep args"""
import ast
import json
import math
import os
import sys

ROOT = "D:/Latent"; sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/scratch"); os.chdir(ROOT)
import gate_s228 as G8  # noqa: E402

OUT = ROOT + "/scratch/gate_s244"; TMP = ROOT + "/scratch/_tmp/s244"; DEV = "cuda:0"; RUNS = ROOT + "/boxes/runs_fsq"
P2 = ("--arm inject --think_k 8 --aux_step 1 --aux_fade 48000,48000 --aux_reweight --fast_add --cache_lower --d_max 2 --steps 12000 --cue step --write_src answer "
      "--init_trunk boxes/runs_fsq/s233_askname_s1/model.pt --init_sha c00f7ec0bd7977fe --write_mix_start 0 --write_mix_end 0").split()
P2_EVAL = "acc,lp5,mean_s,cue_only,cue_wrong,swap_k,swap_1,probe,ws"


def ao(fl):
    fl = list(fl); fl[fl.index("--aux_fade") + 1] = "0,0"; return fl


def det(mod, fl, d):
    G8.fresh(d); return G8.sh([sys.executable, "-B", "-m", "scratch.s222_det_wrap", mod] + fl + ["--device", DEV, "--out", d], d + "/stdout.log")


KEYS = ("model", "opt", "rng_d", "np_rng", "torch_rng", "cuda_rng", "stream_rng", "stream_stats")
strip = lambda p: [{k: v for k, v in json.loads(l_).items() if k != "sec"} for l_ in open(p)]  # noqa: E731


def p2bit():
    com = P2 + ["--fp32", "--micro", "8", "--ckpt_every", "100", "--log_every", "10", "--stop_at", "100"]
    da, db = TMP + "/p2_old", TMP + "/p2_new"; ra = det("boxes/_inject_aw_p2.py", com, da); rb = det("boxes/inject_aw.py", com, db); o = {"rc": [ra, rb]}
    if os.path.exists(da + "/ckpt_last.pt") and os.path.exists(db + "/ckpt_last.pt"):
        o.update(G8.ck_equal(da + "/ckpt_last.pt", db + "/ckpt_last.pt", keys=KEYS)); o["log_equal"] = strip(da + "/log.jsonl") == strip(db + "/log.jsonl")
        o["step100_loss_local"] = [r.get("loss") for r in strip(db + "/log.jsonl") if r.get("step") == 100]
    o["host_logged_step100_loss"] = 3.3319097250699996
    o["pass"] = o["rc"] == [9, 9] and o.get("n_mismatch") == 0 and bool(o.get("log_equal")); return o


def lam_part():
    src = open(ROOT + "/boxes/inject_aw.py", encoding="utf-8").read(); fn = None
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef) and node.name == "lam_at":
            fn = ast.get_source_segment(src, node)
    import textwrap
    ns = {"a": type("A", (), {"aux_step": 1.0})(), "f0_": 0.0, "f1_": 0.0, "sc": 1.0, "max": max}; exec(textwrap.dedent(fn), ns)
    vals = {s: ns["lam_at"](s) for s in range(0, 18001)}; o = {"lam_at_src": fn, "lam_0": vals[0], "lam_ge1_all_zero": all(v == 0.0 for s, v in vals.items() if s >= 1)}
    com = ao(P2) + ["--fp32", "--micro", "8", "--log_every", "1"]
    runs = {}
    for tag, mod, stop in (("ao_1", "boxes/inject_aw.py", 1), ("noaux_1", "boxes/_inject_aw_noaux_s244.py", 1),
                           ("ao_100", "boxes/inject_aw.py", 100), ("noaux_100", "boxes/_inject_aw_noaux_s244.py", 100)):
        d = TMP + "/lam_" + tag; runs[tag] = (d, det(mod, com + ["--ckpt_every", str(stop), "--stop_at", str(stop)], d))
    lg = [json.loads(l_) for l_ in open(runs["ao_100"][0] + "/log.jsonl")]; st = [r for r in lg if "aux_lambda" in r]
    o["ao_records"] = len(st); o["ao_steps"] = [st[0]["step"], st[-1]["step"]] if st else None
    o["aux_lambda_all_zero"] = bool(st) and all(r["aux_lambda"] == 0.0 for r in st); o["aux_ce_finite"] = bool(st) and all(r.get("aux_ce") is not None and math.isfinite(r["aux_ce"]) for r in st)
    for n in ("1", "100"):
        a_, b_ = runs["ao_" + n][0] + "/ckpt_last.pt", runs["noaux_" + n][0] + "/ckpt_last.pt"
        o["noaux_vs_ao_" + n] = G8.ck_equal(a_, b_, keys=("model", "opt")) if os.path.exists(a_) and os.path.exists(b_) else {"missing": True}
    o["rc"] = {k: v[1] for k, v in runs.items()}
    o["pass"] = (o["lam_0"] == 1.0 and o["lam_ge1_all_zero"] and all(v == 9 for v in o["rc"].values()) and o["ao_records"] == 100 and o["ao_steps"] == [1, 100]
                 and o["aux_lambda_all_zero"] and o["aux_ce_finite"] and all(o["noaux_vs_ao_" + n].get("n_mismatch") == 0 for n in ("1", "100")))
    return o


def keep():
    import torch
    o = {}
    for tag, ex in (("on", ["--keep_final_ckpt"]), ("off", [])):
        d = G8.fresh(TMP + "/keep_" + tag)
        rc = G8.sh([sys.executable, "-B", "-m", "boxes.inject_aw"] + [x if x != "12000" else "20" for x in ao(P2)] + ex
                   + ["--micro", "8", "--ckpt_every", "10", "--log_every", "10", "--eval_parts=", "--device", DEV, "--out", d], d + "/stdout.log")
        f = d + "/ckpt_final.pt"
        o[tag] = {"rc": rc, "ckpt_final": os.path.exists(f), "ckpt_last": os.path.exists(d + "/ckpt_last.pt"),
                  "final_step": torch.load(f, map_location="cpu", weights_only=False).get("step") if os.path.exists(f) else None}
        for x in ("ckpt_final.pt", "model.pt"):
            if os.path.exists(os.path.join(d, x)):
                os.remove(os.path.join(d, x))
    o["pass"] = (o["on"]["rc"] == 0 and o["on"]["ckpt_final"] and not o["on"]["ckpt_last"] and o["on"]["final_step"] == 20
                 and o["off"]["rc"] == 0 and not o["off"]["ckpt_final"] and not o["off"]["ckpt_last"]); return o


def args_part():
    import s228_args_check as C
    ref = json.load(open(RUNS + "/aw_plaw2_s1/args.json")); new = {}; o = {}
    for run, fl in (("ao_s1", ao(P2)), ("ao_s2", ao(P2) + ["--seed", "2"]), ("aux_s2", P2 + ["--seed", "2"])):
        d = G8.fresh(TMP + "/args_" + run)
        rc = G8.sh([sys.executable, "-B", "-m", "boxes.inject_aw"] + fl + ["--auto_micro", "--ckpt_every", "1000", "--eval_parts", P2_EVAL, "--keep_final_ckpt",
                                                                         "--device", "cpu", "--args_only", "--out", d], d + "/stdout.log")
        new[run] = json.load(open(d + "/args.json")); o[run] = {"rc": rc, "diff_vs_aw_plaw2_s1": C.diff(ref, new[run])}
    o["ao_s2_vs_aux_s2"] = C.diff(new["aux_s2"], new["ao_s2"]); o["ao_s1_vs_ao_s2"] = C.diff(new["ao_s1"], new["ao_s2"])
    dx = o["aux_s2"]["diff_vs_aw_plaw2_s1"]
    o["aux_s2_vs_aw_plaw2_s1_changed"] = {k: v for k, v in dx.items() if k in ref}; o["aux_s2_vs_aw_plaw2_s1_new_keys"] = {k: v[1] for k, v in dx.items() if k not in ref}
    shas = {k for k in o["aux_s2_vs_aw_plaw2_s1_changed"] if k.endswith("sha256")}
    o["pass"] = (all(o[r]["rc"] == 0 for r in new) and set(o["ao_s2_vs_aux_s2"]) == {"aux_fade"} and set(o["ao_s1_vs_ao_s2"]) == {"seed"}
                 and set(o["aux_s2_vs_aw_plaw2_s1_changed"]) - shas == {"seed"})
    if o["pass"]:
        for r, a in new.items():
            json.dump(a, open(ROOT + "/scratch/s244_args_frozen_%s.json" % r, "w", newline="\n"), indent=1)
    return o


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True); os.makedirs(TMP, exist_ok=True)
    fns = {"p2bit": p2bit, "lam": lam_part, "keep": keep, "args": args_part}
    for part in [x for x in sys.argv[1:] if x in fns]:
        r = fns[part](); json.dump(r, open(os.path.join(OUT, part + ".json"), "w", newline="\n"), indent=1, default=str); print(part, "pass", r["pass"], flush=True)
