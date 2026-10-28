"""§239 LONG-RUN build gates (LR_v1 §6, 1e8b2033; Note 239-1; for Audit's G-D). -> scratch/gate_s239/<part>.json ("pass"). 4060 (cuda:0).
Reference kept: boxes/_inject_aw_eval_s238.py (the §238 eval module, 2514c6ca).
  flagsoff  the default eval path: _inject_aw_eval_s238.run vs the new run, all 13 parts at eval_bs 250 on the 20-step DF model (scratch/_tmp/s238/ev_lat, TEST2-DF cut 250):
            every eval_<part>.json equal minus sec / module_sha256.
  smoke64   Note 239-1: part_long at K' 8,16,32,64 on df_plaw4_s1's model.pt (hard-linked into a scratch dir) with TEST2-DF d 4 := its LONGEST item x eval_bs (100), fp32:
            rc, peak GiB < 7.5, no non-finite S, sec (-> the projected time per model: 4 d x 10 batches).
  cli       python -m boxes.inject_aw_eval --run <copy of the 20-step DF model> --kprime_list 8,16,32,64 --long_ds 2 --eval_bs 100: eval_long.json written with by_d.2
            {acc, s_norm, s_cos, nonfinite} x K' and items.2 {sig, ans, pred {K'}}; s_cos[8] == 1 (within 1e-5); eval.json and every other file untouched (mtimes).
    python -B scratch/gate_s239.py smoke64 cli flagsoff"""
import json
import os
import shutil
import sys

ROOT = "D:/Latent"; sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/scratch")
import gate_s228 as G8  # noqa: E402

OUT = ROOT + "/scratch/gate_s239"; TMP = ROOT + "/scratch/_tmp/s239"; DEV = "cuda:0"; KPL = [8, 16, 32, 64]
DFSRC = ROOT + "/scratch/_tmp/s238/ev_lat"


def flagsoff():
    import torch
    import boxes.inject_ow as IO
    import boxes._inject_aw_eval_s238 as E0
    import boxes.inject_aw_eval as E1
    da, db = G8.fresh(TMP + "/eq_s238"), G8.fresh(TMP + "/eq_new")
    for d_ in (da, db):
        for f_ in ("args.json", "model.pt", "train_summary.json"):
            shutil.copy(os.path.join(DFSRC, f_), os.path.join(d_, f_))
    S = IO.build_sets(mul_s_p=2); S["test2"] = {k: v[:250] for k, v in S["test2"].items()}; torch.zeros(1, device=DEV)
    parts = list(E0.PARTS); E0.run(da, parts, DEV, 250, sets=S); E1.run(db, parts, DEV, 250, sets=S); res = {}
    for p_ in parts:
        ja, jb = (json.load(open(os.path.join(x, "eval_%s.json" % p_))) for x in (da, db))
        for j_ in (ja, jb):
            j_.pop("sec", None); j_.pop("module_sha256", None)
        res[p_] = ja == jb
    return {"parts": res, "n_parts": len(parts), "pass": len(res) == 13 and all(res.values())}


def smoke64():
    import time
    import torch
    import boxes.inject_ow as IO
    import boxes.inject_aw_eval as AE
    d = G8.fresh(TMP + "/smoke64"); src = ROOT + "/boxes/runs_fsq/df_plaw4_s1"; shutil.copy(src + "/args.json", d + "/args.json"); os.link(src + "/model.pt", d + "/model.pt")
    torch.zeros(1, device=DEV); a, m, g = AE.load(d, DEV); S = IO.build_sets(mul_s_p=2); L = lambda it: len(IO.ENC.encode(it["text"]))  # noqa: E731
    big = max(S["test2"][4], key=L); S["test2"] = {4: [big] * 100}; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(DEV); t0 = time.time()
    r = AE.part_long(m, g, S, a, DEV, 100, KPL, [4]); sec = time.time() - t0; pk = torch.cuda.max_memory_allocated(DEV) / 2 ** 30
    os.remove(d + "/model.pt")
    o = {"item_tokens": L(big), "eval_bs": 100, "peak_gib": pk, "sec_one_batch": sec, "projected_min_per_model_d2to5": round(sec * 10 * 4 / 60, 1),
         "nonfinite": r["by_d"]["4"]["nonfinite"], "s_cos_8": r["by_d"]["4"]["s_cos"]["8"], "k_keys": sorted(r["by_d"]["4"]["acc"], key=int)}
    o["pass"] = pk < 7.5 and all(v == 0 for v in o["nonfinite"].values()) and abs(o["s_cos_8"] - 1) < 1e-5 and o["k_keys"] == [str(k) for k in KPL]; return o


def cli():
    d = G8.fresh(TMP + "/cli")
    for f_ in ("args.json", "model.pt", "train_summary.json", "eval.json", "eval_acc.json"):
        if os.path.exists(os.path.join(DFSRC, f_)):
            shutil.copy(os.path.join(DFSRC, f_), os.path.join(d, f_))
    snap = {f_: os.path.getmtime(os.path.join(d, f_)) for f_ in os.listdir(d)}
    rc = G8.sh([sys.executable, "-B", "-m", "boxes.inject_aw_eval", "--run", d, "--kprime_list", "8,16,32,64", "--long_ds", "2", "--eval_bs", "100", "--device", DEV], TMP + "/cli.log")
    fs = sorted(os.listdir(d)); o = {"rc": rc, "files": fs, "others_untouched": {f_: os.path.getmtime(os.path.join(d, f_)) for f_ in snap} == snap}
    if "eval_long.json" in fs:
        J = json.load(open(d + "/eval_long.json")); b = J["by_d"]["2"]
        o.update({"by_d_keys": sorted(J["by_d"]), "fields": sorted(b), "k_keys": sorted(b["acc"], key=int), "item_keys": sorted(J["items"]["2"]), "pred_k": sorted(J["items"]["2"]["pred"], key=int),
                  "n_items": len(J["items"]["2"]["sig"]), "s_cos_8": b["s_cos"]["8"], "meta": {k: J.get(k) for k in ("eval_bs", "eval_dtype", "kprime_list", "ds", "law", "set_law")}})
    o["pass"] = (rc == 0 and set(fs) == set(snap) | {"eval_long.json"} and o["others_untouched"] and o.get("fields") == ["acc", "nonfinite", "s_cos", "s_norm"]
                 and o.get("k_keys") == [str(k) for k in KPL] and o.get("pred_k") == [str(k) for k in KPL] and abs(o.get("s_cos_8", 0) - 1) < 1e-5 and o.get("n_items") == 1000)
    return o


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True); os.makedirs(TMP, exist_ok=True)
    fns = {"flagsoff": flagsoff, "smoke64": smoke64, "cli": cli}
    for part in [x for x in sys.argv[1:] if x in fns]:
        r = fns[part](); json.dump(r, open(os.path.join(OUT, part + ".json"), "w", newline="\n"), indent=1, default=str); print(part, "pass", r["pass"], flush=True)
