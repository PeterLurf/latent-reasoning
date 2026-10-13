"""Log-only gate for the per-step grad-norm logging (Theory / Audit 2026-10-13): nested170 and prosqa168s2 add "gn_preclip" (this step's pre-clip total
norm) and "clip_frac_100" (share of the last 100 steps with norm > 1.0) to every log.jsonl record; nested170 also checkpoints gn_hist / gn_log and
restores them on --resume. 1 CPU thread, pre-change modules = git HEAD copies (argv[1] nested170, argv[2] prosqa168s2):
  g1 nested170, two configs (inject + aux; WINDOW + rms, which fills gn_hist): log records equal after dropping the two new keys, the new keys on
     every training record, model.pt bitwise, eval.json equal (sec fields aside)
  g2 prosqa168s2 stepaux: the same three checks
  g3 nested170 resume (inject + rms; not WINDOW: --resume does not restore the k_window rng): 4 steps with ckpt_every 2 == 2 steps + --resume (model.pt bitwise); the step-2 ckpt holds gn_log (2) / gn_hist; the resumed
     log's clip_frac_100 at step 4 equals the uninterrupted run's
    python scratch/gate_gnlog.py <pre nested170.py> <pre prosqa168s2.py>   -> boxes/runs_fsq/gate_gnlog/gate.json"""
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
os.environ.setdefault("BOXES_NB", "5")
import torch  # noqa: E402

GD = os.path.join(ROOT, "boxes", "runs_fsq", "gate_gnlog"); OUT = {}; SP = os.environ.get("GATE_TMP", os.path.join(ROOT, "scratch", "_gate_gnlog_tmp"))
NEW = ("gn_preclip", "clip_frac_100")
N_BASE = ["--arm", "inject", "--think_k", "2", "--aux_step", "1.0", "--aux_fade", "48000,48000", "--fast_add", "--seed", "1", "--bs", "4", "--per_n", "2",
          "--serial_pool", "40", "--log_every", "1", "--kprimes", "0,1,2,4", "--eval_bs", "8", "--device", "cpu"]
N_CFG = {"inject_aux": [], "window_rms": ["--k_window", "1,2", "--kmatch_impl", "masked", "--add_norm", "rms", "--micro", "2", "--think_k", "10"]}
P_BASE = ["--arm", "stepaux", "--think_k", "2", "--device", "cpu", "--log_every", "1", "--bs", "2", "--train_n", "40", "--test_n", "4", "--extras_n", "2",
          "--probe_n_fit", "8", "--eval_bs", "4", "--sched_scale", "0.0001", "--fast_add", "--no_normcap", "--ckpt_every", "0"]


def run(mod, out, args, keep_ckpt=False, clean=True):
    if clean:
        shutil.rmtree(out, ignore_errors=True)
    code = ("import torch,sys,os,runpy;torch.set_num_threads(1);sys.argv=['x']+sys.argv[1:];" + ("os.remove=lambda p: None;" if keep_ckpt else "") +
            "runpy.run_module('%s',run_name='__main__')" % mod)
    r = subprocess.run([sys.executable, "-c", code] + args + ["--out", out], cwd=ROOT, capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": ROOT, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    assert r.returncode == 0, (mod, args, r.stderr[-2500:])
    return [json.loads(l) for l in open(os.path.join(out, "log.jsonl"))], json.load(open(os.path.join(out, "eval.json"))), torch.load(os.path.join(out, "model.pt"), weights_only=True)


def strip_log(L):
    return [{k: v for k, v in r.items() if k not in NEW + ("sec",)} for r in L]


def strip_eval(e):
    return {k: v for k, v in e.items() if k not in ("eval_sec", "train_sec", "sec")}


def compare(new, pre):
    (ln, en, mn), (lp, ep, mp) = new, pre
    tr = [r for r in ln if "loss" in r]
    return {"log_equal_minus_new": strip_log(ln) == strip_log(lp), "new_keys_on_every_train_record": all(all(k in r for k in NEW) for r in tr) and len(tr) > 0,
            "pre_has_no_new_keys": not any(k in r for r in lp for k in NEW), "model_equal": set(mn) == set(mp) and all(torch.equal(mn[k], mp[k]) for k in mn),
            "eval_equal": strip_eval(en) == strip_eval(ep), "gn_sample": [(r["step"], r.get("gn_preclip"), r.get("clip_frac_100")) for r in tr][:4]}


PART = os.environ.get("GATE_PART", "all")                                    # parallel parts: g1a g1b g2 g3 g4 (one CPU thread each), then GATE_PART=merge
PARTS = ("g1a", "g1b", "g2", "g3", "g4")


def dump():
    os.makedirs(GD, exist_ok=True); json.dump(OUT, open(os.path.join(GD, "gate.json" if PART == "all" else "gate_%s.json" % PART), "w", newline="\n"), indent=1, default=str)


def want(p_):
    return PART in ("all", p_)


def resume_case(sp, a, tag):
    lf, ef, mf = run("boxes.nested170", os.path.join(sp, tag + "_full"), a)
    half = os.path.join(sp, tag + "_half"); run("boxes.nested170", half, a, keep_ckpt=True)
    ck = torch.load(os.path.join(half, "ckpt_last.pt"), weights_only=False); cp = os.path.join(sp, tag + "_ck2.pt"); shutil.copy(os.path.join(half, "ckpt_last.pt"), cp)
    rd = os.path.join(sp, tag + "_res"); shutil.rmtree(rd, ignore_errors=True); os.makedirs(rd)
    for f_ in ("args.json", "s170_init.json"):
        if os.path.exists(os.path.join(half, f_)):
            shutil.copy(os.path.join(half, f_), rd)
    lr, er, mr = run("boxes.nested170", rd, a + ["--resume", cp], clean=False)
    return lf, ef, mf, ck, lr, er, mr


def merge():
    for p_ in PARTS:
        OUT.update(json.load(open(os.path.join(GD, "gate_%s.json" % p_))))
    OUT["g1"] = {**OUT.pop("g1a", {}), **OUT.pop("g1b", {})}; OUT["g1_pass"] = bool(OUT.pop("g1a_pass", False) and OUT.pop("g1b_pass", False))
    OUT["all_pass"] = bool(all(OUT.get("g%d_pass" % i) for i in (1, 2, 3, 4)))
    json.dump(OUT, open(os.path.join(GD, "gate.json"), "w", newline="\n"), indent=1, default=str); print(json.dumps(OUT, default=str)[:2500], flush=True)


def main():
    if PART == "merge":
        return merge()
    sp = SP + "_" + PART; os.makedirs(sp, exist_ok=True); sfx = "" if PART == "all" else "_" + PART
    pn, pp = os.path.join(ROOT, "boxes", "_pre_gn_nested170%s.py" % sfx), os.path.join(ROOT, "boxes", "_pre_gn_prosqa168s2%s.py" % sfx); shutil.copy(sys.argv[1], pn); shutil.copy(sys.argv[2], pp)
    try:
        for p_, name in (("g1a", "inject_aux"), ("g1b", "window_rms")):
            if want(p_):
                a = N_BASE + ["--steps", "3"] + N_CFG[name]
                d = compare(run("boxes.nested170", os.path.join(sp, "n_new_" + name), a), run("boxes._pre_gn_nested170" + sfx, os.path.join(sp, "n_pre_" + name), a))
                OUT[p_] = {name: d}; OUT[p_ + "_pass"] = all(v for k, v in d.items() if k != "gn_sample"); dump(); print(p_, OUT[p_ + "_pass"], flush=True)
        if want("g2"):
            a = P_BASE + ["--steps", "3"]
            OUT["g2"] = compare(run("boxes.prosqa168s2", os.path.join(sp, "p_new"), a), run("boxes._pre_gn_prosqa168s2" + sfx, os.path.join(sp, "p_pre"), a))
            OUT["g2_pass"] = all(v for k, v in OUT["g2"].items() if k != "gn_sample"); dump(); print("g2", OUT["g2_pass"], flush=True)
        if want("g3"):                                                          # inject + rms (fills gn_hist), no k_window
            lf, _, mf, ck, lr, _, mr = resume_case(sp, N_BASE + ["--steps", "4", "--ckpt_every", "2", "--add_norm", "rms"], "r")
            c4f = [r.get("clip_frac_100") for r in lf if r.get("step") == 4 and "loss" in r]; c4r = [r.get("clip_frac_100") for r in lr if r.get("step") == 4 and "loss" in r]
            OUT["g3"] = {"ckpt_step": ck["step"], "ckpt_gn_log_len": len(ck.get("gn_log", [])), "ckpt_gn_hist_len": len(ck.get("gn_hist", [])), "model_equal": all(torch.equal(mf[k], mr[k]) for k in mf),
                         "clip_frac_100_step4_full": c4f, "clip_frac_100_step4_resumed": c4r}
            OUT["g3_pass"] = ck["step"] == 2 and OUT["g3"]["ckpt_gn_log_len"] == 2 and OUT["g3"]["ckpt_gn_hist_len"] == 2 and OUT["g3"]["model_equal"] and c4f == c4r and len(c4f) == 1
            dump(); print("g3", OUT["g3_pass"], flush=True)
        if want("g4"):                                                          # WINDOW resume continues the K draws (rng_kw / kcount checkpointed)
            lf, ef, mf, ck, lr, er, mr = resume_case(sp, N_BASE + ["--steps", "4", "--ckpt_every", "2"] + N_CFG["window_rms"], "w")
            OUT["g4"] = {"ckpt_has_rng_kw": "rng_kw" in ck, "ckpt_kcount": ck.get("kcount"), "model_equal": all(torch.equal(mf[k], mr[k]) for k in mf),
                         "kmatch_counts_equal": ef.get("s170", {}).get("kmatch_counts") == er.get("s170", {}).get("kmatch_counts")}
            OUT["g4_pass"] = OUT["g4"]["ckpt_has_rng_kw"] and OUT["g4"]["model_equal"] and OUT["g4"]["kmatch_counts_equal"]; dump(); print("g4", OUT["g4_pass"], flush=True)
    finally:
        for f_ in (pn, pp):
            os.remove(f_)
        shutil.rmtree(sp, ignore_errors=True)
        if PART == "all":
            OUT["all_pass"] = bool(all(OUT.get("g%d_pass" % i) for i in (1, 2, 3, 4)))
        dump(); print(json.dumps(OUT, default=str)[:1500], flush=True)


if __name__ == "__main__":
    main()
