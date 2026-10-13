"""note 196-1 entry-gated --anneal gates (nested170 AnnealGate + --anneal_gate 'thr:need:max[:bin]'):
  a  flags-off AND fixed-step --anneal (2:1:2,3:2:48000) == the pre-change module (argv[1]) bitwise: 1 CPU thread, --time_steps 4, log + ckpt model + opt
  b  unit, synthetic bin means (bin 1000, thr 0.5, need 2): 0.9, 0.6, 0.45, 0.48 -> advances at step 4000 (reason gate, next stage from 4001);
     0.45, 0.55, 0.45, 0.55, ... never gates (a single bin < 0.5 then >= 0.5 does not count)
  c  unit resume: the gate state()/load() at step 2500 (mid-bin, consec 1) then continued == uninterrupted (every transition and the final state);
     integration: a tiny CPU run (--anneal 2:1:G,3:2:4 --anneal_gate 100:2:3:1 --steps 7) in one go == 3 steps + --resume (ckpt model + opt + gate
     state bitwise; the same transition records)
  d  unit fallback: bins all 0.9 -> advance at step 16000 (reason fallback); a 3-stage gate -> s_full = 32001, end 56000
  e  LR: the logged lr of every training record == lr_at(lr, {}, step, cap) (warm-up 100, cosine over --steps = the cap), in the one-go and resumed runs
    python scratch/gate_annealgate.py <pre nested170.py>   -> boxes/runs_fsq/gate_annealgate/gate.json"""
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
os.environ.setdefault("BOXES_NB", "5")
import torch  # noqa: E402

import boxes.nested170 as N  # noqa: E402
from boxes.fsq120 import lr_at  # noqa: E402

GD = os.path.join(ROOT, "boxes", "runs_fsq", "gate_annealgate"); OUT = {}; SP = os.environ.get("GATE_TMP", os.path.join(ROOT, "scratch", "_gate_ag_tmp"))
BASE = ["--arm", "inject", "--aux_step", "1.0", "--fast_add", "--add_norm", "rms", "--seed", "3", "--bs", "4", "--per_n", "2", "--serial_pool", "40",
        "--log_every", "1", "--device", "cpu"]


def dump():
    os.makedirs(GD, exist_ok=True); json.dump(OUT, open(os.path.join(GD, "gate.json"), "w", newline="\n"), indent=1, default=str)


def run(mod, out, args, clean=True):
    if clean:
        shutil.rmtree(out, ignore_errors=True)
    code = "import torch,sys,runpy;torch.set_num_threads(1);sys.argv=['x']+sys.argv[1:];runpy.run_module('%s',run_name='__main__')" % mod
    r = subprocess.run([sys.executable, "-c", code] + args + ["--out", out], cwd=ROOT, capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": ROOT, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    assert r.returncode == 0, (mod, args, r.stderr[-2500:])
    assert os.path.exists(os.path.join(out, "ckpt_last.pt")), ("no ckpt_last", out, os.listdir(out))
    return [json.loads(l) for l in open(os.path.join(out, "log.jsonl"))], torch.load(os.path.join(out, "ckpt_last.pt"), weights_only=False)


def teq(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and set(a) == set(b) and all(teq(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(teq(x, y) for x, y in zip(a, b))
    return a == b


def nosec(L):
    return [{k: v for k, v in r.items() if k != "sec"} for r in L]


def feed(gate, bins, start=1, stop=None):
    """bins: per-bin loss values (constant within the bin); -> transitions [(step, stage, reason)]."""
    tr = []
    for b_, v in enumerate(bins):
        for st in range(b_ * gate.bin + 1, (b_ + 1) * gate.bin + 1):
            if st < start:
                continue
            if stop is not None and st > stop:
                return tr
            r_ = gate.update(st, v)
            if r_:
                tr.append((st, r_[0], r_[1]))
    return tr


def main():
    os.makedirs(SP, exist_ok=True); pre = os.path.join(ROOT, "boxes", "_pre_ag_nested170.py"); shutil.copy(sys.argv[1], pre)
    try:
        a_ = {}
        for name, ex in (("off", ["--think_k", "2", "--aux_fade", "48000,48000", "--steps", "48000"]),
                         ("fixed_anneal", ["--anneal", "2:1:2,3:2:48000", "--n_max_train", "3", "--think_k", "2", "--aux_fade", "48000,48000", "--steps", "48000"])):
            args = BASE + ex + ["--time_steps", "4", "--ckpt_every", "2"]
            ln, cn = run("boxes.nested170", os.path.join(SP, "new_" + name), args); lp, cp = run("boxes._pre_ag_nested170", os.path.join(SP, "pre_" + name), args)
            a_[name] = {"log_equal": nosec(ln) == nosec(lp), "model_equal": teq(cn["model"], cp["model"]), "opt_equal": teq(cn["opt"], cp["opt"])}
        OUT["a"] = a_; OUT["a_pass"] = all(all(v.values()) for v in a_.values()); dump(); print("a", OUT["a_pass"], flush=True)
        g1 = N.AnnealGate(2, 0.5, 2, 16000, 24000); t1 = feed(g1, [0.9, 0.6, 0.45, 0.48, 0.3])
        g2 = N.AnnealGate(2, 0.5, 2, 16000, 24000); t2 = feed(g2, [0.45, 0.55] * 7 + [0.45])
        OUT["b"] = {"stream_1": t1, "stream_1_s_full": g1.s_full, "stream_alternating": t2}
        OUT["b_pass"] = t1 == [(4000, 1, "gate")] and g1.s_full == 4001 and t2 == []; dump(); print("b", OUT["b_pass"], flush=True)
        bins = [0.9, 0.45, 0.6, 0.45, 0.47, 0.2]
        gu = N.AnnealGate(3, 0.5, 2, 16000, 24000); tu = feed(gu, bins)
        gr = N.AnnealGate(3, 0.5, 2, 16000, 24000); tr_a = feed(gr, bins, stop=2500); st_ = json.loads(json.dumps(gr.state()))
        gr2 = N.AnnealGate(3, 0.5, 2, 16000, 24000); gr2.load(st_); tr_b = feed(gr2, bins, start=2501)
        uc = {"uninterrupted": tu, "resumed": tr_a + tr_b, "state_equal": gu.state() == gr2.state(), "mid_state": st_}
        AG = ["--anneal", "2:1:G,3:2:4", "--anneal_gate", "100:2:3:1", "--n_max_train", "3", "--think_k", "2", "--steps", "7", "--aux_fade", "7,7", "--ckpt_every", "1"]
        lf, cf = run("boxes.nested170", os.path.join(SP, "ag_full"), BASE + AG + ["--time_steps", "7"])
        lh, ch = run("boxes.nested170", os.path.join(SP, "ag_half"), BASE + AG + ["--time_steps", "3"])
        cp_ = os.path.join(SP, "ag_ck.pt"); torch.save(ch, cp_); rd = os.path.join(SP, "ag_res"); shutil.rmtree(rd, ignore_errors=True); os.makedirs(rd)
        for f_ in ("args.json", "s170_init.json"):
            if os.path.exists(os.path.join(SP, "ag_half", f_)):
                shutil.copy(os.path.join(SP, "ag_half", f_), rd)
        lr_, cr = run("boxes.nested170", rd, BASE + AG + ["--time_steps", "4", "--resume", cp_], clean=False)
        trf = [(r["step"], r.get("next_stage"), r.get("reason")) for r in lf if "next_stage" in r]; trr = [(r["step"], r.get("next_stage"), r.get("reason")) for r in lh + lr_ if "next_stage" in r]
        uc["integration"] = {"full_ckpt_step": cf["step"], "resumed_ckpt_step": cr["step"], "model_equal": teq(cf["model"], cr["model"]), "opt_equal": teq(cf["opt"], cr["opt"]),
                             "gate_state_equal": cf.get("anneal_gate") == cr.get("anneal_gate"), "transitions_full": trf, "transitions_split": trr,
                             "full_train_steps": [r["step"] for r in lf if "loss" in r], "gate_state_full": cf.get("anneal_gate")}
        OUT["c"] = uc; OUT["c_pass"] = (tu == tr_a + tr_b and uc["state_equal"] and uc["integration"]["model_equal"] and uc["integration"]["opt_equal"]
                                        and uc["integration"]["gate_state_equal"] and trf == trr and trf == [(2, 1, "gate")] and uc["integration"]["full_train_steps"] == [1, 2, 3, 4, 5, 6])
        dump(); print("c", OUT["c_pass"], flush=True)
        g3 = N.AnnealGate(2, 0.5, 2, 16000, 24000); t3 = feed(g3, [0.9] * 20)
        g4 = N.AnnealGate(3, 0.5, 2, 16000, 24000); t4 = feed(g4, [0.9] * 40)
        OUT["d"] = {"two_stage": t3, "three_stage": t4, "three_stage_s_full": g4.s_full, "three_stage_end": g4.end()}
        OUT["d_pass"] = t3 == [(16000, 1, "fallback")] and t4 == [(16000, 1, "fallback"), (32000, 2, "fallback")] and g4.s_full == 32001 and g4.end() == 56000
        dump(); print("d", OUT["d_pass"], flush=True)
        lrs = [(r["step"], r["lr"], lr_at(1e-4, {}, r["step"], 7)) for r in lf + lr_ if "loss" in r and "lr" in r]
        OUT["e"] = {"n_records": len(lrs), "max_abs_diff": max(abs(x - y) for _, x, y in lrs) if lrs else None, "sample": lrs[:7]}
        OUT["e_pass"] = len(lrs) >= 9 and OUT["e"]["max_abs_diff"] == 0.0; dump(); print("e", OUT["e_pass"], flush=True)
    finally:
        os.remove(pre); shutil.rmtree(SP, ignore_errors=True)
        OUT["all_pass"] = bool(all(OUT.get(k + "_pass") for k in "abcde")); dump(); print(json.dumps(OUT, default=str)[:3000], flush=True)


if __name__ == "__main__":
    main()
