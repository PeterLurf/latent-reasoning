"""nested170 --cache_lower gates (Theory / Audit 2026-10-13; use HELD: never compare a cached cell with an uncached comparator):
  a  flags-off == the pre-change module (argv[1], git HEAD copy), 1 CPU thread, --time_steps 4 via ckpt_last: inject + aux, and WINDOW (k_window
     masked) + aux - log (minus sec), model + opt bitwise
  b  fp64 identity (m.double(), Tensor.float -> double), cached vs uncached on the same batch, rms model: losses and EVERY parameter gradient within
     1e-12 relative (max |diff| / max |grad|), per term, on
       kmatch_loss masked (answer CE), kmatch_aux_loss at a mixed-K WINDOW batch (answer CE and aux CE separately), m.think (loss_items' lambda-0
       path), and loss_items' aux loop emulated with checkpoint(m.write, S, x, msk, hl) on and off (--ckpt_passes)
  c  memory (argv[2], default cuda:1): --mem_check at WINDOW args + --ckpt_passes, chunk 4 (bs 64 micro 16, eval_bs 8), cache on vs off: the ratio
    python scratch/gate_cache_lower_n170.py <pre nested170.py> [cuda dev]   -> boxes/runs_fsq/cache_lower_gates/gate_cache_lower_n170.json"""
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
os.environ.setdefault("BOXES_NB", "5")
import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402
from torch.utils.checkpoint import checkpoint  # noqa: E402

torch.set_num_threads(3)
import boxes.deltamem as DM  # noqa: E402
import boxes.nested170 as N  # noqa: E402
from boxes import s5  # noqa: E402

GD = os.path.join(ROOT, "boxes", "runs_fsq", "cache_lower_gates"); OUT = {}; SP = os.environ.get("GATE_TMP", os.path.join(ROOT, "scratch", "_gate_cl170_tmp"))
BASE = ["--arm", "inject", "--aux_step", "1.0", "--aux_fade", "48000,48000", "--fast_add", "--seed", "1", "--bs", "4", "--per_n", "2", "--serial_pool", "40",
        "--log_every", "1", "--device", "cpu", "--steps", "48000", "--ckpt_every", "2", "--time_steps", "4"]
CFG = {"inject_aux": ["--think_k", "2"], "window_aux": ["--think_k", "10", "--k_window", "1,2", "--kmatch_impl", "masked", "--micro", "2"]}


def dump():
    os.makedirs(GD, exist_ok=True); json.dump(OUT, open(os.path.join(GD, "gate_cache_lower_n170.json"), "w", newline="\n"), indent=1, default=str)


def run(mod, out, args):
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


def grads(m):
    return {n: p.grad.detach().clone() for n, p in m.named_parameters() if p.grad is not None}


def cmp(la, lb, ga, gb):
    gd = max(float((ga[k] - gb[k]).abs().max()) for k in ga); gs = max(float(ga[k].abs().max()) for k in ga)
    return {"loss_a": float(la), "loss_rel_diff": abs(float(la) - float(lb)) / max(abs(float(la)), 1e-300), "grad_max_abs_diff": gd, "grad_scale": gs,
            "grad_rel": gd / gs, "same_keys": set(ga) == set(gb), "n_grads": len(ga)}


def main():
    os.makedirs(SP, exist_ok=True); pre = os.path.join(ROOT, "boxes", "_pre_cl_nested170.py"); shutil.copy(sys.argv[1], pre); dev = sys.argv[2] if len(sys.argv) > 2 else "cuda:1"
    try:
        a_ = {}
        for name, ex in CFG.items():
            ln, cn = run("boxes.nested170", os.path.join(SP, "new_" + name), BASE + ex); lp, cp = run("boxes._pre_cl_nested170", os.path.join(SP, "pre_" + name), BASE + ex)
            a_[name] = {"log_equal": [{k: v for k, v in r.items() if k != "sec"} for r in ln] == [{k: v for k, v in r.items() if k != "sec"} for r in lp],
                        "model_equal": teq(cn["model"], cp["model"]), "opt_equal": teq(cn["opt"], cp["opt"])}
        OUT["a"] = a_; OUT["a_pass"] = all(all(v.values()) for v in a_.values()); dump(); print("a", OUT["a_pass"], flush=True)
        _f = torch.Tensor.float; torch.Tensor.float = lambda self, *a, **k: self.double(); DM.FAST_ADD = True
        try:
            g = s5.load_gpt2("cpu"); torch.manual_seed(1); m = N.InjectN(g, 6, "top", 6, 6, "rms").double()
            with torch.no_grad():
                m.mem.wo.weight.normal_(0, 0.02); m.mem.fade.fill_(1.0); m.mem.gamma.fill_(0.5)
            TE = N.test_set(1000); its = [it for n_ in (2, 4, 6) for it in [x for x in TE if x["n"] == n_][:2]]; ks = [it["n"] + 1 + (i % 2) for i, it in enumerate(its)]
            ac = lambda: torch.autocast("cpu", enabled=False)                   # noqa: E731
            b_ = {}

            def two(fn):
                r_ = []
                for c in (False, True):
                    m.zero_grad(set_to_none=True); l_ = fn(c); l_.backward(); r_.append((l_.detach(), grads(m)))
                return cmp(r_[0][0], r_[1][0], r_[0][1], r_[1][1])
            b_["kmatch_loss_masked"] = two(lambda c: N.kmatch_loss(m, its, ks, "cpu", ac, masked=True, cache=c)[0])
            b_["kmatch_aux_answer"] = two(lambda c: N.kmatch_aux_loss(m, its, ks, "cpu", ac, cache=c)[0])
            b_["kmatch_aux_aux"] = two(lambda c: N.kmatch_aux_loss(m, its, ks, "cpu", ac, cache=c)[1])
            x, msk = m.think_inputs(its, "cpu"); tgt = torch.tensor(N.NUM_IDS[[it["t"] for it in its]])

            def think_path(c):
                S = m.think(m.mem.zero_state(len(its), "cpu").double(), x, msk, 4, h_low=m.lower(x) if c else None); return Fn.cross_entropy(m.dec_logits(S, its, "cpu"), tgt)
            b_["think_answer"] = two(think_path)
            for ck in (False, True):
                def aux_loop(c, ck=ck):
                    S = m.mem.zero_state(len(its), "cpu").double(); hl = m.lower(x) if c else None; per = []
                    for t in range(1, 5):
                        S = checkpoint(m.write, S, x, msk, hl, use_reentrant=False) if ck else m.write(S, x, msk, hl)
                        idx = [i for i, it in enumerate(its) if len(N.aux_of(it)) >= t]
                        if idx:
                            lg_a, y_a = m.aux_logits(S[idx], [its[i] for i in idx], t, "cpu"); per.append(Fn.cross_entropy(lg_a.transpose(1, 2), y_a, ignore_index=-100))
                    return Fn.cross_entropy(m.dec_logits(S, its, "cpu"), tgt) + torch.stack(per).mean()
                b_["aux_loop_ckpt_%s" % ck] = two(aux_loop)
        finally:
            torch.Tensor.float = _f
        OUT["b"] = b_; OUT["b_pass"] = all(v["loss_rel_diff"] <= 1e-12 and v["grad_rel"] <= 1e-12 and v["same_keys"] for v in b_.values()); dump(); print("b", OUT["b_pass"], flush=True)
        pk = {}
        for tag, ex in (("off", []), ("cache", ["--cache_lower"])):
            o_ = os.path.join(SP, "mem_" + tag)
            r = subprocess.run([sys.executable, "-m", "boxes.nested170", "--arm", "inject", "--think_k", "12", "--aux_step", "1.0", "--aux_fade", "48000,48000", "--fast_add",
                                "--k_window", "1,4", "--kmatch_impl", "masked", "--ckpt_passes", "--seed", "1", "--steps", "48000", "--bs", "64", "--micro", "16", "--eval_bs", "8",
                                "--mem_check", "--device", dev, "--out", o_] + ex, cwd=ROOT, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": ROOT})
            pk[tag] = json.load(open(os.path.join(o_, "mem_check.json"))) if r.returncode == 0 else {"err": r.stderr[-600:]}
        ratio = pk["cache"]["peak_gib"] / pk["off"]["peak_gib"] if all("peak_gib" in v for v in pk.values()) else None
        OUT["c"] = {"chunk4_ckpt_passes": pk, "ratio_cache_over_off": ratio, "gpu": torch.cuda.get_device_name(dev)}; OUT["c_pass"] = ratio is not None; dump(); print("c", OUT["c_pass"], ratio, flush=True)
    finally:
        os.remove(pre); shutil.rmtree(SP, ignore_errors=True)
        OUT["all_pass"] = bool(all(OUT.get(k + "_pass") for k in "abc")); dump(); print(json.dumps(OUT, default=str)[:3000], flush=True)


if __name__ == "__main__":
    main()
