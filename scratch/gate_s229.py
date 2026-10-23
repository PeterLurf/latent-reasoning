"""§229 LAT-OW-CUM build gates (THEORY §229, b8ef3778; for Audit's G-D). -> scratch/gate_s229/<part>.json ("pass"). Run dirs under scratch/_tmp/s229.
  flagsoff  --aux_target value (default) is bitwise §228: the committed boxes/inject_ow.py (HEAD, copied to boxes/_inject_ow_head.py for the run, then
            removed) vs the working tree, §228 P-LAT2 flags, det wrapper, fp32, micro 8, sched_scale 0.02, stop_at 30 (ckpt 30): model + opt + streams
            torch.equal, and the logs (minus sec) identical.
  cumtok    2,000 OW TOPO items per d 1..8 (seed 22997): tok_ow(.., 'cumulative'): aux_t decodes to ' ' + R[t] + EOT, the value index is unique and
            decodes to ' v_t', len(aux) == d; mean aux tokens per pass vs the value target (the memory driver).
  gradflow  fp64 CPU, perturbed InjectOW (gate_s228.perturbed), 4 d 2 items with the cumulative target, K 8: the answer CE and the aux CE each give a
            nonzero finite grad on all 148 GPT-2 tensors and all 12 memory tensors; AUX_W on passes 1..2.
  plain     --arm direct (no new flags), 1 step on the 4060 -> rc 9.
  mem       the s229 cell's flags on the 4060 (--auto_micro --mem_budget 6.0 --time_steps 20; 7.0 chose micro 4 at 6.59 GiB and OOMed in training on the 8 GB card): mem_check.json + timing (the 3090 numbers come from b342's
            K7 / G-TIME at admission).
  args      --args_only s229_plat2c_s1 vs scratch/s228_args_frozen_s228_plat2_s1.json: the non-CONTROL diff == {aux_target}.
    python -B scratch/gate_s229.py flagsoff cumtok gradflow plain mem args"""
import json
import os
import shutil
import subprocess
import sys

ROOT = "D:/Latent"; sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/scratch")
OUT = ROOT + "/scratch/gate_s229"; TMP = ROOT + "/scratch/_tmp/s229"; DEV = "cuda:0"
ENV = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
LAT2 = ["--arm", "inject", "--think_k", "8", "--aux_step", "1", "--aux_fade", "48000,48000", "--aux_reweight", "--fast_add", "--cache_lower", "--d_max", "2", "--steps", "8000"]
import gate_s228 as G8  # noqa: E402


def flagsoff():
    head = ROOT + "/boxes/_inject_ow_head.py"
    open(head, "w", newline="\n").write(subprocess.run(["git", "show", "HEAD:boxes/inject_ow.py"], capture_output=True, text=True, cwd=ROOT).stdout)
    o = {}
    try:
        common = LAT2 + ["--fp32", "--micro", "8", "--sched_scale", "0.02", "--ckpt_every", "30", "--log_every", "10", "--device", DEV, "--stop_at", "30"]
        da = G8.fresh(TMP + "/off_head"); db = G8.fresh(TMP + "/off_new")
        ra = G8.sh([sys.executable, "-B", "scratch/s222_det_wrap.py", "boxes/_inject_ow_head.py"] + common + ["--out", da], da + "/stdout.log")
        rb = G8.sh([sys.executable, "-B", "scratch/s222_det_wrap.py", "boxes/inject_ow.py"] + common + ["--out", db], db + "/stdout.log")
        o["rc"] = [ra, rb]
        if os.path.exists(da + "/ckpt_last.pt") and os.path.exists(db + "/ckpt_last.pt"):
            o.update(G8.ck_equal(da + "/ckpt_last.pt", db + "/ckpt_last.pt", keys=("model", "opt", "rng_d", "stream_rng", "expo", "binacc", "vbin", "gn_log")))
            strip = lambda p: [{k: v for k, v in json.loads(l_).items() if k != "sec"} for l_ in open(p)]  # noqa: E731
            o["log_equal"] = strip(da + "/log.jsonl") == strip(db + "/log.jsonl")
    finally:
        os.remove(head)
    o["pass"] = o["rc"] == [9, 9] and o.get("n_mismatch") == 0 and o.get("log_equal"); return o


def cumtok():
    import numpy as np
    import boxes.inject_ow as IO
    import boxes.ow_words as OW
    s = OW.Stream(22997, frames=OW.FRAME_TRAIN, **IO.OWKW); o = {"by_d": {}}; ok = True
    for d in range(1, 9):
        bad = {"text": 0, "vidx": 0, "len": 0}; La, Lv = [], []
        for _ in range(2000):
            it = s.draw(d); c = IO.tok_ow(it, "cumulative"); v = IO.tok_ow(it, "value")
            bad["len"] += len(c["aux"]) != d
            for t in range(1, d + 1):
                q = c["aux"][t - 1]; bad["text"] += IO.ENC.decode(q[:-1]) != " " + it["R"][t] or q[-1] != IO.EOT
                bad["vidx"] += IO.ENC.decode([q[c["vidx"][t - 1]]]) != " %d" % it["values"][t]; La.append(len(q)); Lv.append(len(v["aux"][t - 1]))
        o["by_d"][str(d)] = {**bad, "aux_tokens_mean": float(np.mean(La)), "value_tokens_mean": float(np.mean(Lv))}; ok = ok and not any(bad.values())
    o["pass"] = ok; return o


def gradflow():
    with G8.F64():
        import math
        import torch
        import boxes.inject_ow as IO
        import boxes.nested170 as N7
        import boxes.ow_words as OW
        torch.set_num_threads(2); m, g = G8.perturbed(torch.float64); s = OW.Stream(22997, frames=OW.FRAME_TRAIN, **IO.OWKW)
        its = [IO.tok_ow(s.draw(2), "cumulative") for _ in range(4)]; r = {}
        for term in ("answer_ce", "aux_ce"):
            m.zero_grad(set_to_none=True); la, aux, _, _, _, W = G8.loss_terms(m, its, 8, True); (la if term == "answer_ce" else aux).backward()
            gr = {n: (float(p.grad.norm()) if p.grad is not None else None) for n, p in m.named_parameters()}
            okf = lambda v: v is not None and v > 0 and math.isfinite(v)  # noqa: E731
            gp = {n: v for n, v in gr.items() if n.startswith("g.")}; mp = {n: v for n, v in gr.items() if n.startswith("mem.")}
            r[term] = {"value": float((la if term == "answer_ce" else aux).detach()), "gpt2_nonzero": sum(okf(v) for v in gp.values()), "n_gpt2": len(gp),
                       "mem_nonzero": sum(okf(v) for v in mp.values()), "n_mem": len(mp), "zero_or_none": sorted(n for n, v in gr.items() if not okf(v))[:20]}
        r["aux_weights"] = sorted(set(W)); r["aux_tokens"] = [len(q) for q in its[0]["aux"]]
        r["pass"] = (all(r[t]["gpt2_nonzero"] == r[t]["n_gpt2"] == 148 and r[t]["mem_nonzero"] == r[t]["n_mem"] == 12 for t in ("answer_ce", "aux_ce"))
                     and all(w == N7.AUX_W[t - 1] for t, w in W) and sorted({t for t, _ in W}) == [1, 2])
        return r


def plain():
    d = G8.fresh(TMP + "/plain"); rc = G8.sh([sys.executable, "-B", "-m", "boxes.inject_ow", "--arm", "direct", "--d_max", "2", "--steps", "8000", "--stop_at", "1", "--ckpt_every", "0",
                                             "--log_every", "1", "--device", DEV, "--out", d], d + "/stdout.log")
    return {"rc": rc, "pass": rc == 9}


def mem():
    d = G8.fresh(TMP + "/mem"); rc = G8.sh([sys.executable, "-B", "-m", "boxes.inject_ow"] + LAT2 + ["--aux_target", "cumulative", "--auto_micro", "--mem_budget", "6.0", "--time_steps", "20",
                                                                        "--log_every", "10", "--device", DEV, "--out", d], d + "/stdout.log")
    o = {"rc": rc}
    for f_ in ("mem_check.json", "timing.json"):
        if os.path.exists(os.path.join(d, f_)):
            o[f_[:-5]] = json.load(open(os.path.join(d, f_)))
    o["pass"] = rc == 0 and "timing" in o; return o


def args():
    d = G8.fresh(TMP + "/args"); rc = G8.sh([sys.executable, "-B", "-m", "boxes.inject_ow"] + LAT2 + ["--aux_target", "cumulative", "--auto_micro", "--ckpt_every", "1000", "--eval_parts", "acc,decode",
                                                                         "--device", "cpu", "--args_only", "--out", d], d + "/stdout.log")
    r = subprocess.run([sys.executable, "scratch/s228_args_check.py", "diff", d + "/args.json", "scratch/s228_args_frozen_s228_plat2_s1.json"], capture_output=True, text=True, cwd=ROOT)
    dif = json.loads(r.stdout)["diff"]; o = {"rc": rc, "diff": dif}
    o["pass"] = rc == 0 and set(dif) == {"aux_target"} and dif["aux_target"] == ["cumulative", None]
    if o["pass"]:
        shutil.copy(d + "/args.json", ROOT + "/scratch/s228_args_frozen_s229_plat2c_s1.json")   # the frozen file the §228 filler / args check read
    return o


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True); os.makedirs(TMP, exist_ok=True)
    for part in [x for x in sys.argv[1:] if x in ("flagsoff", "cumtok", "gradflow", "plain", "mem", "args")]:
        r = {"flagsoff": flagsoff, "cumtok": cumtok, "gradflow": gradflow, "plain": plain, "mem": mem, "args": args}[part]()
        json.dump(r, open(os.path.join(OUT, part + ".json"), "w", newline="\n"), indent=1, default=str); print(part, "pass", r["pass"], flush=True)
