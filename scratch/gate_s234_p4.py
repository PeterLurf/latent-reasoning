"""§234 P-4 build gates (Note 234-5; for Audit's G-D): the regenerated boxes/inject_aw.py / inject_aw_eval.py (WS-READOUT over d 1..d_max; new eval parts lp5h,
swap_kd) vs the P-2 versions saved as boxes/_inject_aw_p2.py / _inject_aw_eval_p2.py (git HEAD 6a8fa80e). -> scratch/gate_s234/p4_<part>.json. 4060 (cuda:0).
  eqtrain  the P-2 command (LAT-AW, d <= 2) under the det wrapper, fp32, micro 8, sched_scale 0.02, stop_at 30 (ckpt 30): P-2 module vs new module -> every ckpt
           field torch.equal and logs (minus sec) equal (incl. the step-30 bin's ws_readout over d 1 / 2).
  eqeval   the 20-step P-2 evalsmoke model (scratch/_tmp/s234/evsmoke_lat_aw, copied twice): the P-2 eval module vs the new one, the 9 P-2 parts at eval_bs 250 on
           TEST2 cut to 250 per d -> every eval_<part>.json equal minus sec / module_sha256.
  smoke4   P-4 LAT-AW (d <= 4, 16k) and CD-AN (d <= 4) 50 steps (stop_at 50, micro 8) -> rc 9; LAT-AW's step-50 bin: ws_readout keys d 1..4 x passes 1..8,
           aux_vacc_by_d_pass pools to aux_vacc_by_pass (|diff| <= 1e-12).
  ev4      every part (11) at eval_bs 250 on a 20-step P-4 LAT-AW model (TEST2 cut to 250 per d) and acc on a 20-step P-4 CD-AN (eval_bs 100): files, by_d keys
           (lp5h d 3 / 4 with t_mean 2; swap_kd d 3 / 4: n_pairs 1,000 each, t_mid 1 / 2 (Note 234-5 amendment), 0 sig overlap with TEST2; ws d 1..4; cue_only / cue_wrong / lp5 d 2..4).
  f4       memory / timing at the P-4 shapes (budget 6.0, --auto_micro --time_steps 30): LAT-AW d <= 4 and CD-AN d <= 4.
  args4    --args_only P-4 vs the P-2 frozen args: LAT-AW == {d_max, steps, eval_parts}; CD-AN == {d_max, steps}.
  ast4     every part_* in boxes/inject_aw_eval.py is @torch.no_grad() (11 parts).
    python -B scratch/gate_s234_p4.py ast4 args4 eqtrain eqeval smoke4 ev4 f4"""
import json
import os
import shutil
import sys

ROOT = "D:/Latent"; sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/scratch")
import gate_s228 as G8  # noqa: E402
import gate_s234 as G4  # noqa: E402

OUT = G4.OUT; TMP = G4.TMP + "/p4"; DEV = G4.DEV
P4_EVAL = "acc,lp5,mean_s,cue_only,cue_wrong,swap_k,swap_1,probe,ws,lp5h,swap_kd"
LAT4 = [x if x not in ("2", "12000") else {"2": "4", "12000": "16000"}[x] for x in G4.LAT_AW]
CD4 = [x if x not in ("2", "12000") else {"2": "4", "12000": "16000"}[x] for x in G4.CD_AN]
assert LAT4[LAT4.index("--d_max") + 1] == "4" and LAT4[LAT4.index("--steps") + 1] == "16000" and LAT4.count("4") == 1
assert CD4[CD4.index("--d_max") + 1] == "4" and CD4[CD4.index("--steps") + 1] == "16000"


def eqtrain():
    com = G4.LAT_AW + ["--fp32", "--micro", "8", "--sched_scale", "0.02", "--ckpt_every", "30", "--log_every", "10", "--stop_at", "30"]
    da, db = TMP + "/eq_p2mod", TMP + "/eq_newmod"
    ra = G4.det("boxes/_inject_aw_p2.py", com, da); rb = G4.det("boxes/inject_aw.py", com, db); r = {"rc": [ra, rb]}
    if os.path.exists(da + "/ckpt_last.pt") and os.path.exists(db + "/ckpt_last.pt"):
        r.update(G8.ck_equal(da + "/ckpt_last.pt", db + "/ckpt_last.pt", keys=("model", "opt", "rng_d", "np_rng", "torch_rng", "cuda_rng", "stream_rng", "expo", "binacc",
                                                                                 "vbin", "vbin_d", "gn_log", "wg_hist")))
        strip = lambda p: [{k: v for k, v in json.loads(l_).items() if k != "sec"} for l_ in open(p)]  # noqa: E731
        la, lb = strip(da + "/log.jsonl"), strip(db + "/log.jsonl"); r["log_equal"] = la == lb
        r["ws_readout_ds"] = sorted([x for x in lb if "by_d" in x][-1].get("ws_readout", {}))
    r["pass"] = r["rc"] == [9, 9] and r.get("n_mismatch") == 0 and bool(r.get("log_equal")) and r.get("ws_readout_ds") == ["1", "2"]; return r


def eqeval():
    import torch
    import boxes.inject_ow as IO
    import boxes._inject_aw_eval_p2 as E2
    import boxes.inject_aw_eval as E4
    src = TMP.rsplit("/", 1)[0] + "/evsmoke_lat_aw"; da, db = G8.fresh(TMP + "/eqev_p2"), G8.fresh(TMP + "/eqev_new")
    for d_ in (da, db):
        for f_ in ("args.json", "model.pt", "train_summary.json"):
            shutil.copy(os.path.join(src, f_), os.path.join(d_, f_))
    S = IO.build_sets(); S["test2"] = {k: v[:250] for k, v in S["test2"].items()}; torch.zeros(1, device=DEV)
    parts = list(E2.PARTS); E2.run(da, parts, DEV, 250, sets=S); E4.run(db, parts, DEV, 250, sets=S); res = {}; defs = {}
    for p_ in parts:
        fa, fb = os.path.join(da, "eval_%s.json" % p_), os.path.join(db, "eval_%s.json" % p_)
        if not (os.path.exists(fa) and os.path.exists(fb)):
            res[p_] = "missing"; continue
        ja, jb = (json.load(open(f_)) for f_ in (fa, fb))
        if ja.get("def") != jb.get("def"):
            defs[p_] = [ja.get("def"), jb.get("def")]                       # the description string only (ws: "d 1 / 2" -> "d 1..d_max")
        for j_ in (ja, jb):
            j_.pop("sec", None); j_.pop("module_sha256", None); j_.pop("def", None)
        res[p_] = ja == jb
    return {"parts": res, "def_changed": defs, "pass": len(res) == 9 and all(v is True for v in res.values()) and set(defs) <= {"ws"}}


def smoke4():
    o = {}
    for tag, fl in (("lat_aw4", LAT4), ("cd_an4", CD4)):
        d = G8.fresh(TMP + "/smoke_" + tag); rc = G8.sh([sys.executable, "-B", "-m", "boxes.inject_aw"] + fl + ["--micro", "8", "--stop_at", "50", "--ckpt_every", "0", "--log_every", "10",
                                                                                                            "--device", DEV, "--out", d], d + "/stdout.log")
        o[tag] = {"rc": rc}
        if tag == "lat_aw4" and os.path.exists(d + "/log.jsonl"):
            br = [json.loads(l_) for l_ in open(d + "/log.jsonl") if '"by_d"' in l_][-1]; p = br.get("aux_vacc_by_pass", {}); bd = br.get("aux_vacc_by_d_pass", {}); bn = br.get("aux_vacc_by_d_pass_n", {})
            pooled = {t: sum(bd[dd][t] * bn[dd][t] for dd in bd if t in bd[dd]) / sum(bn[dd][t] for dd in bn if t in bn[dd]) for t in p}
            ws = br.get("ws_readout", {})
            o[tag].update({"ws_ds": sorted(ws), "ws_passes": {k: sorted(v, key=int) for k, v in ws.items()}, "d_keys": sorted(bd), "exposure_by_d": br.get("exposure_by_d"),
                           "pool_max_abs_diff": max(abs(pooled[t] - p[t]) for t in p) if p else None})
    la = o["lat_aw4"]
    o["pass"] = (la["rc"] == 9 and o["cd_an4"]["rc"] == 9 and la.get("ws_ds") == ["1", "2", "3", "4"]
                 and all(v == [str(t) for t in range(1, 9)] for v in la["ws_passes"].values()) and la.get("pool_max_abs_diff") is not None and la["pool_max_abs_diff"] <= 1e-12)
    return o


def ev4():
    import torch
    import boxes.inject_ow as IO
    import boxes.inject_aw_eval as AE
    o = {}
    for tag, fl, ebs, parts in (("lat_aw4", LAT4, 250, P4_EVAL.split(",")), ("cd_an4", CD4, 100, ["acc"])):
        d = G8.fresh(TMP + "/ev_" + tag); rc = G8.sh([sys.executable, "-B", "-m", "boxes.inject_aw"] + fl + ["--steps", "20", "--micro", "8", "--eval_bs", str(ebs), "--eval_parts", "",
                                                                                                         "--ckpt_every", "0", "--log_every", "10", "--device", DEV, "--out", d], d + "/stdout.log")
        S = IO.build_sets(); S["test2"] = {k: v[:250] for k, v in S["test2"].items()}; torch.zeros(1, device=DEV); torch.cuda.reset_peak_memory_stats(DEV)
        AE.run(d, parts, DEV, ebs, sets=S); pk = torch.cuda.max_memory_allocated(DEV) / 2 ** 30
        fs = sorted(f_ for f_ in os.listdir(d) if f_.startswith("eval")); o[tag] = {"rc": rc, "files": fs, "peak_gib": pk}
        if tag == "lat_aw4":
            J = {p_: json.load(open(os.path.join(d, "eval_%s.json" % p_))) for p_ in parts}; t2 = {it["sig"] for dd in (3, 4) for it in S["test2"][dd]}
            T2full = IO.build_sets()["test2"]; t2f = {it["sig"] for dd in (3, 4) for it in T2full[dd]}
            sk = J["swap_kd"]
            o[tag].update({"lp5h": {k: {"t_mean": v["t_mean"], "ks": sorted(v["intact"])} for k, v in J["lp5h"]["by_d"].items()},
                           "swap_kd": {k: {"n_pairs": v["n_pairs"], "tried": v["tried"], "t_mid": v["t_mid"], "mid_ks": sorted(v["swap_mid"])} for k, v in sk["by_d"].items()},
                           "swap_kd_test2_overlap": sum(len((set(sk["items"][k]["rec_sig"]) | set(sk["items"][k]["don_sig"])) & t2f) for k in sk["items"]),
                           "ws_ds": sorted(J["ws"]["by_d"]), "cue_only_ds": sorted(J["cue_only"]["by_d"]), "cue_wrong_ds": sorted(J["cue_wrong"]["by_d"]),
                           "lp5_ds": sorted(J["lp5"]["by_d"]), "swap_k_n": J["swap_k"]["n_pairs"], "probe_n": J["probe"]["n"], "n_test2_cut": len(t2)})
    la = o["lat_aw4"]
    o["pass"] = (la["rc"] == 0 and o["cd_an4"]["rc"] == 0 and set(la["files"]) == {"eval.json"} | {"eval_%s.json" % p_ for p_ in P4_EVAL.split(",")}
                 and o["cd_an4"]["files"] == ["eval.json", "eval_acc.json"] and sorted(la["lp5h"]) == ["3", "4"] and all(v["t_mean"] == 2 for v in la["lp5h"].values())
                 and sorted(la["swap_kd"]) == ["3", "4"] and all(v["n_pairs"] == 1000 for v in la["swap_kd"].values()) and [la["swap_kd"][k]["t_mid"] for k in ("3", "4")] == [1, 2]
                 and la["swap_kd_test2_overlap"] == 0 and la["ws_ds"] == ["1", "2", "3", "4"] and la["cue_only_ds"] == ["2", "3", "4"] and la["lp5_ds"] == ["2", "3", "4"]
                 and la["swap_k_n"] == 1000 and la["probe_n"] == 3000 and la["peak_gib"] < 7.5)
    return o


def f4():
    o = {}
    for tag, fl in (("lat_aw4", LAT4), ("cd_an4", CD4)):
        d = G8.fresh(TMP + "/f4_" + tag); rc = G8.sh([sys.executable, "-B", "-m", "boxes.inject_aw"] + fl + ["--auto_micro", "--mem_budget", "6.0", "--time_steps", "30", "--log_every", "10",
                                                                                                         "--device", DEV, "--out", d], d + "/stdout.log")
        r = {"rc": rc}
        for f_ in ("mem_check.json", "timing.json"):
            if os.path.exists(os.path.join(d, f_)):
                r[f_[:-5]] = json.load(open(os.path.join(d, f_)))
        o[tag] = r
    o["pass"] = all(v.get("rc") == 0 and "timing" in v for v in o.values()); return o


def args4():
    import s228_args_check as C
    o = {}
    for tag, fl, ref, exp in (("lat_aw4", LAT4 + ["--auto_micro", "--ckpt_every", "1000", "--eval_parts", P4_EVAL], "scratch/s234_args_frozen_aw_plaw2_s1.json", {"d_max", "steps", "eval_parts"}),
                              ("cd_an4", CD4 + ["--auto_micro", "--ckpt_every", "1000", "--eval_parts", "acc"], "scratch/s234_args_frozen_aw_pcd2_s1.json", {"d_max", "steps"})):
        d = G8.fresh(TMP + "/args_" + tag); rc = G8.sh([sys.executable, "-B", "-m", "boxes.inject_aw"] + fl + ["--device", "cpu", "--args_only", "--out", d], d + "/stdout.log")
        new = json.load(open(d + "/args.json")); dif = C.diff(json.load(open(ROOT + "/" + ref)), new)
        o[tag] = {"rc": rc, "diff": dif, "args_file": d + "/args.json", "pass": rc == 0 and set(dif) == exp}
    o["pass"] = all(v["pass"] for v in o.values()); return o


def ast4():
    import ast
    t = ast.parse(open(ROOT + "/boxes/inject_aw_eval.py", encoding="utf-8").read()); r = {}
    for n in t.body:
        if isinstance(n, ast.FunctionDef) and n.name.startswith("part_"):
            r[n.name] = [ast.unparse(d) for d in n.decorator_list]
    return {"parts": r, "pass": len(r) == 11 and all(v == ["torch.no_grad()"] for v in r.values())}


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True); os.makedirs(TMP, exist_ok=True)
    fns = {"eqtrain": eqtrain, "eqeval": eqeval, "smoke4": smoke4, "ev4": ev4, "f4": f4, "args4": args4, "ast4": ast4}
    for part in [x for x in sys.argv[1:] if x in fns]:
        r = fns[part](); json.dump(r, open(os.path.join(OUT, "p4_" + part + ".json"), "w", newline="\n"), indent=1, default=str); print(part, "pass", r["pass"], flush=True)
