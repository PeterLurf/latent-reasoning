"""§237 REG-BOOT build gates (THEORY §237 69fc9e4e + Note 237-1 Audit conditions; for Audit's G-D). -> scratch/gate_s237/<part>.json ("pass"). 4060 (cuda:0) / CPU.
Reference kept: boxes/_inject_aw_s236.py (the §236 module, 86f38688). INITS: boxes/runs_fsq/aw_plaw2_s1/model.pt (LAT-AW) and aw_pcd2_s1/model.pt (CD-AN).
  strict    the §237 build path (build_aw at the rcb args + --init_model, strict): the loaded params' key set == the file's, every tensor torch.equal to the file;
            for LAT-AW and CD-AN; a wrong --init_model_sha asserts (main, --args_only).
  step0     Note 237-1 (2): the loaded LAT-AW model (fp32, eval) reproduces aw_plaw2_s1's TEST2 d 2 answers at K' 8 on the first 200 TEST2 d 2 items (default law, the
            P-2 items), one batch of 200: acc and the per-item differences vs aw_plaw2_s1/eval_acc.json (3090, eval_bs 250); CD-AN likewise vs aw_pcd2_s1's (K' 0).
            Pass: <= 2 item differences each (fp32 batch / card noise; §236 bs checks: 0-6 / 12,000).
  dose      Note 237-1 (2): skipping the calibration leaves the dose / lambda paths unchanged: rcb (init) vs rc (no init) LAT-AW, 20 steps (stop_at 20, log_every 1, micro 8,
            REG args): the per-step aux_lambda / gamma (== 1.0) / fade sequences equal; rc writes s228_init.json (calibrated), rcb does not; rcb's s_add at step 0 == the file's.
  gnlog     Note 237-1 (2): the rcb smoke's step-20 bin carries gn_preclip_by_d with a key per d seen (and the rc smoke's bin does not: default logs unchanged).
  flagsoff  _inject_aw_s236.py vs the new inject_aw.py at the §236 REG args (rc_plaw4_s1 / rc_pcd4_s1), det wrap, fp32, micro 8, sched 0.02, stop_at 30: every
            listed ckpt field torch.equal, logs (minus sec) equal.
  args      --args_only rcb_plaw4_s1 / rcb_plaw4_s2 / rcb_pcd4_s1 vs scratch/s236_args_frozen_rc_plaw4_s1.json / _rc_pcd4_s1.json: diff == {init_model, init_model_sha,
            init_model_sha256} (+ seed for s2); init_model_sha256 == the init file's full sha256.
    python -B scratch/gate_s237.py strict args step0 dose gnlog flagsoff"""
import argparse
import hashlib
import json
import os
import sys

ROOT = "D:/Latent"; sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/scratch")
import gate_s228 as G8  # noqa: E402
import gate_s234 as G4  # noqa: E402
import gate_s234_p4 as P4  # noqa: E402

OUT = ROOT + "/scratch/gate_s237"; TMP = ROOT + "/scratch/_tmp/s237"; DEV = "cuda:0"
P4_EVAL = P4.P4_EVAL; REG = ["--mul_s_p", "0"]; LATR, CDR = P4.LAT4 + REG, P4.CD4 + REG
INIT_L, INIT_C = "boxes/runs_fsq/aw_plaw2_s1/model.pt", "boxes/runs_fsq/aw_pcd2_s1/model.pt"


def sha(p):
    return hashlib.sha256(open(os.path.join(ROOT, p), "rb").read()).hexdigest()


def initf(arm):
    p = INIT_L if arm == "lat" else INIT_C; return ["--init_model", p, "--init_model_sha", sha(p)[:16]]


def build(arm, dev="cpu"):
    """main()'s build path: seed, build_aw (args parsed by inject_aw's own parser via --args_only-free parse), then the strict init load."""
    import torch
    import boxes.inject_aw as AW
    fl = (LATR if arm == "lat" else CDR) + initf(arm)
    ap_ns = {"arm": fl[fl.index("--arm") + 1], "mem_write": "top", "write_src": "answer" if arm == "lat" else "all", "init_trunk": fl[fl.index("--init_trunk") + 1],
             "init_sha": fl[fl.index("--init_sha") + 1]}
    ns = argparse.Namespace(**ap_ns); torch.manual_seed(1); m, g, params = AW.build_aw(ns, dev)
    sd = torch.load(os.path.join(ROOT, INIT_L if arm == "lat" else INIT_C), map_location=dev, weights_only=True); params.load_state_dict(sd, strict=True)
    return m, g, params, sd


def strict():
    import torch
    o = {}
    for arm in ("lat", "cd"):
        m, g, params, sd = build(arm); ps = params.state_dict()
        o[arm] = {"keys_equal": sorted(ps) == sorted(sd), "n_keys": len(sd), "all_equal": all(torch.equal(ps[k], sd[k]) for k in sd), "class": type(params).__name__}
    d = G8.fresh(TMP + "/wrongsha"); fl = LATR + ["--init_model", INIT_L, "--init_model_sha", "0000000000000000"]
    rc = G8.sh([sys.executable, "-B", "-m", "boxes.inject_aw"] + fl + ["--device", "cpu", "--args_only", "--out", d], d + "/stdout.log")
    o["wrong_sha_rc"] = rc; o["wrong_sha_asserted"] = rc != 0 and "init_model sha" in open(d + "/stdout.log").read()
    o["pass"] = all(o[a]["keys_equal"] and o[a]["all_equal"] for a in ("lat", "cd")) and o["wrong_sha_asserted"]; return o


def step0():
    import numpy as np
    import torch
    import boxes.inject_ow as IO
    import boxes.inject_aw as AW
    import boxes.inject_aw_eval as AE
    torch.backends.cuda.matmul.allow_tf32 = False; S = IO.build_sets(); o = {}
    for arm, run, kk in (("lat", "aw_plaw2_s1", 8), ("cd", "aw_pcd2_s1", 0)):
        m, g, params, _ = build(arm, DEV); params.eval(); ra = json.load(open(ROOT + "/boxes/runs_fsq/%s/args.json" % run))
        its = [AW.tok_aw(it, ra.get("aux_target", "value"), ra.get("cue", "none"), ra.get("write_src", "all")) for it in S["test2"][2][:200]]
        P = AE.answers_by_kprime(m, g, its, AE.KMAX if m is not None else 0, DEV, 200); pred = P[:, kk] if m is not None else P[:, 0]
        ref = json.load(open(ROOT + "/boxes/runs_fsq/%s/eval_acc.json" % run))["items"]["2"]; rp = np.array([(x[kk] if isinstance(x, list) else x) for x in ref["pred"][:200]])
        y = np.array([it["ans"] for it in its]); same = ref["sig"][:200] == [it["sig"] for it in its]
        o[arm] = {"run": run, "k": kk, "same_items": same, "acc_now": float((pred == y).mean()), "acc_ref": float((rp == y).mean()), "n_differ": int((pred != rp).sum()),
                  "fade": float(m.mem.fade) if m is not None else None, "gamma": float(m.mem.gamma) if m is not None else None}
        del m, g, params; torch.cuda.empty_cache()
    o["pass"] = all(o[a]["same_items"] and o[a]["n_differ"] <= 2 for a in ("lat", "cd")); return o


def _smoke(tag, fl):
    d = G8.fresh(TMP + "/smoke_" + tag); rc = G8.sh([sys.executable, "-B", "-m", "boxes.inject_aw"] + fl + ["--micro", "8", "--stop_at", "20", "--ckpt_every", "0", "--log_every", "1",
                                                                                                      "--device", DEV, "--out", d], d + "/stdout.log")
    lg = [json.loads(l_) for l_ in open(d + "/log.jsonl")] if os.path.exists(d + "/log.jsonl") else []
    return d, rc, lg


def dose():
    import torch
    o = {}; runs = {}
    for tag, fl in (("rc", LATR), ("rcb", LATR + initf("lat"))):
        d, rc, lg = _smoke(tag, fl); rows = [r for r in lg if "gamma" in r]
        runs[tag] = {"rc": rc, "seq": [(r["step"], r["aux_lambda"], r["gamma"], r["fade"]) for r in rows], "s228_init": os.path.exists(d + "/s228_init.json"), "dir": d}
    o["rc"] = {k: v for k, v in runs["rc"].items() if k != "seq"}; o["rcb"] = {k: v for k, v in runs["rcb"].items() if k != "seq"}
    o["dose_lambda_paths_equal"] = runs["rc"]["seq"] == runs["rcb"]["seq"] and len(runs["rc"]["seq"]) == 20; o["gamma_all_1"] = all(x[2] == 1.0 for x in runs["rcb"]["seq"])
    m, g, params, sd = build("lat"); o["s_add_loaded_equals_file"] = bool(torch.equal(m.mem.s_add.detach(), sd["mem.s_add"])) if "mem.s_add" in sd else None
    o["pass"] = (runs["rc"]["rc"] == 9 and runs["rcb"]["rc"] == 9 and o["dose_lambda_paths_equal"] and o["gamma_all_1"] and runs["rc"]["s228_init"] and not runs["rcb"]["s228_init"]
                 and o["s_add_loaded_equals_file"] is True)
    return o


def gnlog():
    br_b = [json.loads(l_) for l_ in open(TMP + "/smoke_rcb/log.jsonl") if '"by_d"' in l_][-1]; br_r = [json.loads(l_) for l_ in open(TMP + "/smoke_rc/log.jsonl") if '"by_d"' in l_][-1]
    g_ = br_b.get("gn_preclip_by_d", {}); o = {"rcb_gn_preclip_by_d": g_, "rcb_by_d_keys": sorted(br_b["by_d"]), "rc_has_key": "gn_preclip_by_d" in br_r}
    o["pass"] = sorted(g_) == sorted(br_b["by_d"]) and len(g_) > 0 and not o["rc_has_key"]; return o


def flagsoff():
    o = {}
    for tag, fl in (("lat_reg", LATR), ("cd_reg", CDR)):
        com = fl + ["--fp32", "--micro", "8", "--sched_scale", "0.02", "--ckpt_every", "30", "--log_every", "10", "--stop_at", "30"]
        da, db = TMP + "/fo_s236mod_" + tag, TMP + "/fo_newmod_" + tag
        ra = G4.det("boxes/_inject_aw_s236.py", com, da); rb = G4.det("boxes/inject_aw.py", com, db); r = {"rc": [ra, rb]}
        if os.path.exists(da + "/ckpt_last.pt") and os.path.exists(db + "/ckpt_last.pt"):
            r.update(G8.ck_equal(da + "/ckpt_last.pt", db + "/ckpt_last.pt", keys=("model", "opt", "rng_d", "np_rng", "torch_rng", "cuda_rng", "stream_rng", "stream_stats", "expo",
                                                                                     "binacc", "vbin", "vbin_d", "gn_log", "wg_hist", "dseq", "iseq")))
            strip = lambda p: [{k: v for k, v in json.loads(l_).items() if k != "sec"} for l_ in open(p)]  # noqa: E731
            r["log_equal"] = strip(da + "/log.jsonl") == strip(db + "/log.jsonl")
        r["pass"] = r["rc"] == [9, 9] and r.get("n_mismatch") == 0 and bool(r.get("log_equal")); o[tag] = r
    o["pass"] = all(v["pass"] for v in o.values()); return o


def args_part():
    import s228_args_check as C
    o = {}; der = {"init_model", "init_model_sha", "init_model_sha256"}
    for tag, fl, ref, extra, arm in (("rcb_plaw4_s1", LATR + initf("lat") + ["--auto_micro", "--ckpt_every", "1000", "--eval_parts", P4_EVAL], "scratch/s236_args_frozen_rc_plaw4_s1.json", set(), "lat"),
                                     ("rcb_plaw4_s2", LATR + initf("lat") + ["--seed", "2", "--auto_micro", "--ckpt_every", "1000", "--eval_parts", P4_EVAL], "scratch/s236_args_frozen_rc_plaw4_s2.json", set(), "lat"),
                                     ("rcb_pcd4_s1", CDR + initf("cd") + ["--auto_micro", "--ckpt_every", "1000", "--eval_parts", "acc"], "scratch/s236_args_frozen_rc_pcd4_s1.json", set(), "cd")):
        d = G8.fresh(TMP + "/args_" + tag); rc = G8.sh([sys.executable, "-B", "-m", "boxes.inject_aw"] + fl + ["--device", "cpu", "--args_only", "--out", d], d + "/stdout.log")
        new = json.load(open(d + "/args.json")); dif = C.diff(json.load(open(ROOT + "/" + ref)), new)
        o[tag] = {"rc": rc, "diff": dif, "args_file": d + "/args.json", "init_sha_full_ok": new.get("init_model_sha256") == sha(INIT_L if arm == "lat" else INIT_C),
                  "pass": rc == 0 and set(dif) == der | extra and new.get("init_model_sha256") == sha(INIT_L if arm == "lat" else INIT_C)}
    o["pass"] = all(v["pass"] for v in o.values()); return o


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True); os.makedirs(TMP, exist_ok=True)
    fns = {"strict": strict, "step0": step0, "dose": dose, "gnlog": gnlog, "flagsoff": flagsoff, "args": args_part}
    for part in [x for x in sys.argv[1:] if x in fns]:
        r = fns[part](); json.dump(r, open(os.path.join(OUT, part + ".json"), "w", newline="\n"), indent=1, default=str); print(part, "pass", r["pass"], flush=True)
