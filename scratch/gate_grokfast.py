"""§196 GROKFAST-S3 gates (Theory 78487d7a; nested170 --grokfast ema, grokfast_ema). Runs: 1 CPU thread, inject + aux + rms (s185_rms_s3's path at
think_k 2), --steps 48000 --time_steps 4 --ckpt_every 2 --log_every 1 (no eval; compared through ckpt_last.pt = the step-4 model / optimizer / ema):
  ga flags-off == the pre-change module (argv[1], git HEAD copy): log (minus sec) and ckpt model + opt state bitwise
  gb --grokfast ema --gf_lamb 0 == flags-off: model + opt bitwise; log equal minus the gf_* fields (which are present)
  gc unit: grokfast_ema vs a literal copy of ironjr/grokfast gradfilter_ema on a toy model over 5 steps: filtered grads and ema bitwise; ema at step 1
     == g * alpha + g * (1 - alpha); with lamb 0 the grads are untouched
  gd resume: 4 steps in one go == 2 steps + --resume (model + opt + gf_ema bitwise); ckpt carries gf_ema
  ge gradflow: on the rms model through kmatch_aux_loss, answer + aux backward, then the filter: every trainable tensor that had a grad keeps a
     nonzero, finite grad, grad == g + lamb * h exactly, and the mem / trunk grads are nonzero
  gf memory: the filter keeps one parameter-sized buffer (ema) + one transient snapshot on logging steps: 2 x the trainable bytes added to
     s185_rms_s3's cloud peak (mem_check.json) < 21 GiB
  gl log fields: with lamb 2 the logged gf_slow_share / gf_update_norm / gf_ans_max_logit / gf_weight_norm are finite on every training record
    python scratch/gate_grokfast.py <pre nested170.py>   -> boxes/runs_fsq/gate_grokfast/gate.json"""
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
os.environ.setdefault("BOXES_NB", "5")
import math  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

torch.set_num_threads(1)
import boxes.deltamem as DM  # noqa: E402
import boxes.nested170 as N  # noqa: E402
from boxes import s5  # noqa: E402

GD = os.path.join(ROOT, "boxes", "runs_fsq", "gate_grokfast"); OUT = {}; SP = os.environ.get("GATE_TMP", os.path.join(ROOT, "scratch", "_gate_grokfast_tmp"))
BASE = ["--arm", "inject", "--think_k", "2", "--aux_step", "1.0", "--aux_fade", "48000,48000", "--fast_add", "--add_norm", "rms", "--seed", "3", "--bs", "4",
        "--per_n", "2", "--serial_pool", "40", "--log_every", "1", "--steps", "48000", "--ckpt_every", "2", "--device", "cpu"]
GF = ("gf_slow_share", "gf_update_norm", "gf_ans_max_logit", "gf_weight_norm")


def dump():
    os.makedirs(GD, exist_ok=True); json.dump(OUT, open(os.path.join(GD, "gate.json"), "w", newline="\n"), indent=1, default=str)


def run(mod, out, args, clean=True):
    if clean:
        shutil.rmtree(out, ignore_errors=True)
    code = "import torch,sys,runpy;torch.set_num_threads(1);sys.argv=['x']+sys.argv[1:];runpy.run_module('%s',run_name='__main__')" % mod
    r = subprocess.run([sys.executable, "-c", code] + args + ["--out", out], cwd=ROOT, capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": ROOT, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    assert r.returncode == 0, (mod, args, r.stderr[-2500:])
    return [json.loads(l) for l in open(os.path.join(out, "log.jsonl"))], torch.load(os.path.join(out, "ckpt_last.pt"), weights_only=False)


def teq(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and set(a) == set(b) and all(teq(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(teq(x, y) for x, y in zip(a, b))
    return a == b


def strip(L, extra=()):
    return [{k: v for k, v in r.items() if k not in ("sec",) + tuple(extra)} for r in L]


def ref_gradfilter_ema(m, grads=None, alpha=0.98, lamb=2.0):                 # ironjr/grokfast grokfast.py gradfilter_ema, verbatim logic
    if grads is None:
        grads = {n: p.grad.data.detach() for n, p in m.named_parameters() if p.requires_grad and p.grad is not None}
    for n, p in m.named_parameters():
        if p.requires_grad and p.grad is not None:
            grads[n] = grads[n] * alpha + p.grad.data.detach() * (1 - alpha)
            p.grad.data = p.grad.data + grads[n] * lamb
    return grads


def main():
    os.makedirs(SP, exist_ok=True); pre = os.path.join(ROOT, "boxes", "_pre_gf_nested170.py"); shutil.copy(sys.argv[1], pre)
    try:
        a4 = BASE + ["--time_steps", "4"]
        l_new, c_new = run("boxes.nested170", os.path.join(SP, "off_new"), a4); l_pre, c_pre = run("boxes._pre_gf_nested170", os.path.join(SP, "off_pre"), a4)
        OUT["ga"] = {"log_equal": strip(l_new) == strip(l_pre), "model_equal": teq(c_new["model"], c_pre["model"]), "opt_equal": teq(c_new["opt"], c_pre["opt"]),
                     "no_gf_keys_off": not any(k in r for r in l_new for k in GF), "ckpt_step": c_new["step"]}
        OUT["ga_pass"] = all(v for k, v in OUT["ga"].items() if k != "ckpt_step") and c_new["step"] == 4; dump(); print("ga", OUT["ga_pass"], flush=True)
        l_l0, c_l0 = run("boxes.nested170", os.path.join(SP, "lam0"), a4 + ["--grokfast", "ema", "--gf_lamb", "0"])
        OUT["gb"] = {"model_equal": teq(c_l0["model"], c_new["model"]), "opt_equal": teq(c_l0["opt"], c_new["opt"]), "log_equal_minus_gf": strip(l_l0, GF) == strip(l_new),
                     "gf_keys_present": all(all(k in r for k in GF) for r in l_l0 if "loss" in r), "ckpt_has_gf_ema": "gf_ema" in c_l0}
        OUT["gb_pass"] = all(OUT["gb"].values()); dump(); print("gb", OUT["gb_pass"], flush=True)
        torch.manual_seed(0); toy = torch.nn.Sequential(torch.nn.Linear(5, 7), torch.nn.Tanh(), torch.nn.Linear(7, 3)); toy2 = torch.nn.Sequential(torch.nn.Linear(5, 7), torch.nn.Tanh(), torch.nn.Linear(7, 3))
        toy2.load_state_dict(toy.state_dict()); ema, refg = {}, None; ok_g, ok_h, first = True, True, None
        for st in range(5):
            x = torch.randn(4, 5, generator=torch.Generator().manual_seed(st)); y = torch.randn(4, 3, generator=torch.Generator().manual_seed(100 + st))
            for mm in (toy, toy2):
                mm.zero_grad(); Fn.mse_loss(mm(x), y).backward()
            if st == 0:
                first = {n: p.grad.detach().clone() for n, p in toy.named_parameters()}
            N.grokfast_ema(list(toy.named_parameters()), ema, 0.98, 2.0); refg = ref_gradfilter_ema(toy2, refg, 0.98, 2.0)
            ok_g &= all(torch.equal(p.grad, q.grad) for p, q in zip(toy.parameters(), toy2.parameters())); ok_h &= all(torch.equal(ema[n], refg[n]) for n in ema)
            if st == 0:
                ok_init = all(torch.equal(ema[n], first[n] * 0.98 + first[n] * (1 - 0.98)) for n in ema)
        toy.zero_grad(); Fn.mse_loss(toy(torch.ones(2, 5)), torch.zeros(2, 3)).backward(); g0 = {n: p.grad.clone() for n, p in toy.named_parameters()}
        N.grokfast_ema(list(toy.named_parameters()), dict(ema), 0.98, 0.0); lam0_untouched = all(torch.equal(p.grad, g0[n]) for n, p in toy.named_parameters())
        OUT["gc"] = {"grads_bitwise_vs_ref_5_steps": bool(ok_g), "ema_bitwise_vs_ref": bool(ok_h), "ema_step1_formula": bool(ok_init), "lamb0_grads_untouched": bool(lam0_untouched)}
        OUT["gc_pass"] = all(OUT["gc"].values()); dump(); print("gc", OUT["gc_pass"], flush=True)
        g4 = a4 + ["--grokfast", "ema"]; l_f, c_f = run("boxes.nested170", os.path.join(SP, "gf_full"), g4)
        _, c_h = run("boxes.nested170", os.path.join(SP, "gf_half"), BASE + ["--time_steps", "2", "--grokfast", "ema"])
        cp = os.path.join(SP, "gf_ck2.pt"); torch.save(c_h, cp); rd = os.path.join(SP, "gf_res"); shutil.rmtree(rd, ignore_errors=True); os.makedirs(rd)
        for f_ in ("args.json", "s170_init.json"):
            if os.path.exists(os.path.join(SP, "gf_half", f_)):
                shutil.copy(os.path.join(SP, "gf_half", f_), rd)
        _, c_r = run("boxes.nested170", rd, BASE + ["--time_steps", "2", "--grokfast", "ema", "--resume", cp], clean=False)
        OUT["gd"] = {"half_ckpt_step": c_h["step"], "half_has_gf_ema": bool(c_h.get("gf_ema")), "resumed_step": c_r["step"], "model_equal": teq(c_r["model"], c_f["model"]),
                     "opt_equal": teq(c_r["opt"], c_f["opt"]), "gf_ema_equal": teq(c_r.get("gf_ema"), c_f.get("gf_ema"))}
        OUT["gd_pass"] = c_h["step"] == 2 and c_r["step"] == 4 and all(v for k, v in OUT["gd"].items() if isinstance(v, bool)); dump(); print("gd", OUT["gd_pass"], flush=True)
        tr = [r for r in l_f if "loss" in r]
        OUT["gl"] = {"n_train_records": len(tr), "sample": [{k: r.get(k) for k in ("step",) + GF} for r in tr]}
        OUT["gl_pass"] = len(tr) == 4 and all(all(r.get(k) is not None and math.isfinite(r[k]) for k in GF) for r in tr); dump(); print("gl", OUT["gl_pass"], flush=True)
        DM.FAST_ADD = True; g = s5.load_gpt2("cpu"); torch.manual_seed(3); m = N.InjectN(g, 6, "top", 6, 6, "rms")
        with torch.no_grad():
            m.mem.wo.weight.normal_(0, 0.02); m.mem.fade.fill_(1.0); m.mem.gamma.fill_(0.5)
        TE = N.test_set(1000); its = [it for n_ in (3, 5) for it in [x for x in TE if x["n"] == n_][:2]]; named = [(n, p) for n, p in m.named_parameters() if p.requires_grad]; ema = {}
        for st in range(2):
            m.zero_grad(set_to_none=True); la, ax, hx, _ = N.kmatch_aux_loss(m, its, [it["n"] for it in its], "cpu", lambda: torch.autocast("cpu", enabled=False)); (la + ax).backward()
            g_raw = {n: p.grad.detach().clone() for n, p in named if p.grad is not None}; h_prev = {n: v.clone() for n, v in ema.items()}
            N.grokfast_ema(named, ema, 0.98, 2.0)
        exact = all(torch.equal(p.grad, g_raw[n] + (h_prev[n] * 0.98 + g_raw[n] * (1 - 0.98)) * 2.0) for n, p in named if n in g_raw)
        mem_g = float(sum(p.grad.abs().sum() for n, p in named if n.startswith("mem.") and p.grad is not None)); trunk_g = float(sum(p.grad.abs().sum() for n, p in named if not n.startswith("mem.") and p.grad is not None))
        OUT["ge"] = {"n_with_grad": len(g_raw), "all_finite": all(bool(torch.isfinite(p.grad).all()) for n, p in named if p.grad is not None), "grad_eq_g_plus_lamb_h": bool(exact),
                     "mem_grad_abs_sum": mem_g, "trunk_grad_abs_sum": trunk_g}
        OUT["ge_pass"] = OUT["ge"]["all_finite"] and exact and mem_g > 0 and trunk_g > 0; dump(); print("ge", OUT["ge_pass"], flush=True)
        nb = sum(p.numel() * p.element_size() for _, p in named); mc = json.load(open("boxes/runs_fsq/s185_rms_s3/mem_check.json"))
        OUT["gf"] = {"trainable_bytes_gib": nb / 2 ** 30, "added_gib_upper": 2 * nb / 2 ** 30, "s185_rms_s3_cloud": {"micro": mc.get("micro"), "peak_gib": mc.get("peak_gib")},
                     "proj_gib": mc.get("peak_gib", 99) + 2 * nb / 2 ** 30, "note": "ema (1 x trainable, fp32) + the logging-step snapshot (1 x, transient); the cell's own --auto_micro check runs before training (filter off)"}
        OUT["gf_pass"] = OUT["gf"]["proj_gib"] < 21.0; dump(); print("gf", OUT["gf_pass"], flush=True)
    finally:
        os.remove(pre); shutil.rmtree(SP, ignore_errors=True)
        OUT["all_pass"] = bool(all(OUT.get(k + "_pass") for k in ("ga", "gb", "gc", "gd", "ge", "gf", "gl"))); dump(); print(json.dumps(OUT, default=str)[:3000], flush=True)


if __name__ == "__main__":
    main()
