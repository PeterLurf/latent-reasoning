"""§198 (b) MEM-HIGH gates (nested170 --mem_layer, Theory e69bc32a):
  a  flags-off == the pre-change module (argv[1]) bitwise: 1 CPU thread, inject + aux + rms, --time_steps 4 (log, ckpt model + opt)
  b  the flag: a tiny CPU run with --mem_layer 9 --cache_lower carries mem_layer 9 in args.json; its model rebuilt by halt173.load_model has mem.layer 9;
     the memory read fires entering block 9 (a read_fixed tap sees the residual after blocks 0..8 == gpt_lower(.., 9))
  c  fp64 identity at l_m 9, cached vs uncached (losses and every gradient within 1e-12 relative): kmatch_loss masked, kmatch_aux answer / aux, think
  d  gradflow at l_m 9 (cached): answer CE and aux CE each give nonzero grads on mem.* and on trunk blocks below and above 9
  e  memory (argv[2], default cuda:1): mem_check at s185_rms_s1's args + --cache_lower, chunk 4 (bs 64 micro 16, eval_bs 8), --mem_layer 9 vs 6; the ratio
     times b280's cloud peak (s198_injc_s1 mem_check, micro 1)
    python scratch/gate_memlayer.py <pre nested170.py> [cuda dev]   -> boxes/runs_fsq/gate_memlayer/gate.json"""
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
os.environ.setdefault("BOXES_NB", "5")
import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

torch.set_num_threads(3)
import boxes.deltamem as DM  # noqa: E402
import boxes.halt173 as H  # noqa: E402
import boxes.nested170 as N  # noqa: E402
from boxes import s5  # noqa: E402

GD = os.path.join(ROOT, "boxes", "runs_fsq", "gate_memlayer"); OUT = {}; SP = os.environ.get("GATE_TMP", os.path.join(ROOT, "scratch", "_gate_ml_tmp"))
BASE = ["--arm", "inject", "--think_k", "2", "--aux_step", "1.0", "--aux_fade", "48000,48000", "--fast_add", "--add_norm", "rms", "--seed", "1", "--bs", "4",
        "--per_n", "2", "--serial_pool", "40", "--log_every", "1", "--device", "cpu", "--steps", "48000", "--ckpt_every", "2", "--time_steps", "4"]


def dump():
    os.makedirs(GD, exist_ok=True); json.dump(OUT, open(os.path.join(GD, "gate.json"), "w", newline="\n"), indent=1, default=str)


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


def main():
    os.makedirs(SP, exist_ok=True); pre = os.path.join(ROOT, "boxes", "_pre_ml_nested170.py"); shutil.copy(sys.argv[1], pre); dev = sys.argv[2] if len(sys.argv) > 2 else "cuda:1"
    try:
        ln, cn = run("boxes.nested170", os.path.join(SP, "new"), BASE); lp, cp = run("boxes._pre_ml_nested170", os.path.join(SP, "pre"), BASE)
        OUT["a"] = {"log_equal": [{k: v for k, v in r.items() if k != "sec"} for r in ln] == [{k: v for k, v in r.items() if k != "sec"} for r in lp],
                    "model_equal": teq(cn["model"], cp["model"]), "opt_equal": teq(cn["opt"], cp["opt"])}
        OUT["a_pass"] = all(OUT["a"].values()); dump(); print("a", OUT["a_pass"], flush=True)
        d9 = os.path.join(SP, "ml9"); _, c9 = run("boxes.nested170", d9, BASE + ["--mem_layer", "9", "--cache_lower"])
        a9 = json.load(open(os.path.join(d9, "args.json"))); torch.save(c9["model"], os.path.join(d9, "model.pt"))
        kind, m9, _ = H.load_model(d9, "cpu"); TE = N.test_set(1000); its = [it for n_ in (2, 4, 6) for it in [x for x in TE if x["n"] == n_][:2]]
        x, msk = m9.think_inputs(its, "cpu"); seen = {}; orig = DM.read_fixed

        def rf(mem, h, S, rec=None):
            seen.setdefault("h", h.detach().clone()); return orig(mem, h, S, rec)
        DM.read_fixed = rf
        try:
            with torch.no_grad():
                m9.write(m9.mem.zero_state(len(its), "cpu"), x, msk); hl = DM.gpt_lower(m9.g, m9.g.embed(x), 9)
        finally:
            DM.read_fixed = orig
        OUT["b"] = {"args_mem_layer": a9.get("mem_layer"), "args_cache_lower": a9.get("cache_lower"), "loaded_mem_layer": m9.mem.layer,
                    "read_input_eq_lower9": bool(torch.equal(seen.get("h"), hl)) if "h" in seen else False}
        OUT["b_pass"] = a9.get("mem_layer") == 9 and a9.get("cache_lower") is True and m9.mem.layer == 9 and OUT["b"]["read_input_eq_lower9"]; dump(); print("b", OUT["b_pass"], flush=True)
        _f = torch.Tensor.float; torch.Tensor.float = lambda self, *a, **k: self.double(); DM.FAST_ADD = True
        try:
            g = s5.load_gpt2("cpu"); torch.manual_seed(1); m = N.InjectN(g, 9, "top", 6, 6, "rms").double()
            with torch.no_grad():
                m.mem.wo.weight.normal_(0, 0.02); m.mem.fade.fill_(1.0); m.mem.gamma.fill_(0.5)
            ks = [it["n"] + 1 + (i % 2) for i, it in enumerate(its)]; ac = lambda: torch.autocast("cpu", enabled=False)   # noqa: E731
            x, msk = m.think_inputs(its, "cpu"); tgt = torch.tensor(N.NUM_IDS[[it["t"] for it in its]]); c_ = {}

            def two(fn):
                r_ = []
                for c in (False, True):
                    m.zero_grad(set_to_none=True); l_ = fn(c); l_.backward(); r_.append((l_.detach(), grads(m)))
                ga, gb = r_[0][1], r_[1][1]; gd = max(float((ga[k] - gb[k]).abs().max()) for k in ga); gs = max(float(ga[k].abs().max()) for k in ga)
                return {"loss_rel_diff": abs(float(r_[0][0]) - float(r_[1][0])) / abs(float(r_[0][0])), "grad_rel": gd / gs, "same_keys": set(ga) == set(gb), "grads": ga}
            for nm, fn in (("kmatch_loss_masked", lambda c: N.kmatch_loss(m, its, ks, "cpu", ac, masked=True, cache=c)[0]),
                           ("kmatch_aux_answer", lambda c: N.kmatch_aux_loss(m, its, ks, "cpu", ac, cache=c)[0]),
                           ("kmatch_aux_aux", lambda c: N.kmatch_aux_loss(m, its, ks, "cpu", ac, cache=c)[1]),
                           ("think_answer", lambda c: Fn.cross_entropy(m.dec_logits(m.think(m.mem.zero_state(len(its), "cpu").double(), x, msk, 4, h_low=m.lower(x) if c else None), its, "cpu"), tgt))):
                c_[nm] = two(fn)
            dd = {}
            for nm in ("kmatch_aux_answer", "kmatch_aux_aux"):
                gg = c_[nm]["grads"]
                dd[nm] = {"mem": float(sum(v.abs().sum() for k, v in gg.items() if k.startswith("mem."))), "trunk_below_9": float(sum(v.abs().sum() for k, v in gg.items() if any(f"h_{i}_" in k for i in range(9)) and not k.startswith("mem."))),
                          "trunk_9_up": float(sum(v.abs().sum() for k, v in gg.items() if any(f"h_{i}_" in k for i in range(9, 12)) and not k.startswith("mem."))),
                          "trunk_any": float(sum(v.abs().sum() for k, v in gg.items() if not k.startswith("mem.")))}
        finally:
            torch.Tensor.float = _f
        OUT["c"] = {k: {kk: vv for kk, vv in v.items() if kk != "grads"} for k, v in c_.items()}
        OUT["c_pass"] = all(v["loss_rel_diff"] <= 1e-12 and v["grad_rel"] <= 1e-12 and v["same_keys"] for v in c_.values())
        OUT["d"] = dd; OUT["d_pass"] = all(v["mem"] > 0 and v["trunk_below_9"] > 0 and v["trunk_9_up"] > 0 for v in dd.values()); dump(); print("c", OUT["c_pass"], "d", OUT["d_pass"], flush=True)
        pk = {}
        for l_ in ("6", "9"):
            o_ = os.path.join(SP, "mem_" + l_)
            r = subprocess.run([sys.executable, "-m", "boxes.nested170", "--arm", "inject", "--think_k", "8", "--aux_step", "1.0", "--aux_fade", "48000,48000", "--fast_add",
                                "--add_norm", "rms", "--cache_lower", "--mem_layer", l_, "--seed", "1", "--steps", "48000", "--bs", "64", "--micro", "16", "--eval_bs", "8",
                                "--mem_check", "--device", dev, "--out", o_], cwd=ROOT, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": ROOT})
            pk[l_] = json.load(open(os.path.join(o_, "mem_check.json"))) if r.returncode == 0 else {"err": r.stderr[-600:]}
        ratio = pk["9"]["peak_gib"] / pk["6"]["peak_gib"] if all("peak_gib" in v for v in pk.values()) else None
        b280 = os.path.join("boxes", "runs_fsq", "s198_injc_s1", "mem_check.json"); cloud = json.load(open(b280)) if os.path.exists(b280) else {"micro": 1, "peak_gib": 15.372199207544327, "src": "b280 host log 17:39"}
        OUT["e"] = {"chunk4": pk, "ratio_l9_over_l6": ratio, "cloud_l6_cached": cloud, "proj_l9_gib": cloud["peak_gib"] * ratio if ratio else None, "gpu": torch.cuda.get_device_name(dev)}
        OUT["e_pass"] = bool(ratio) and OUT["e"]["proj_l9_gib"] < 21.0; dump(); print("e", OUT["e_pass"], flush=True)
    finally:
        os.remove(pre); shutil.rmtree(SP, ignore_errors=True)
        OUT["all_pass"] = bool(all(OUT.get(k + "_pass") for k in "abcde")); dump(); print(json.dumps(OUT, default=str)[:3000], flush=True)


if __name__ == "__main__":
    main()
