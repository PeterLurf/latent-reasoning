"""§198 (c) READ-SLOTS gates (nested170 --arm readslots / ReadSlotsN, Theory e69bc32a):
  a  flags-off == the pre-change module (argv[1]) bitwise: 1 CPU thread, inject + aux + rms, --time_steps 4 (log, ckpt model + opt)
  b  a 1-step CPU run with --arm readslots --add_norm rms --aux_step 1.0: rc 0; args carry arm readslots / add_norm rms / n_slots 8
  c  the input KV is encoded once and reused (in-process, rms + aux through kmatch_aux_loss at K = 4): n_encode == 1, n_slot_pass == 4, the decoder used
     the cached encoding (no second encode)
  d  K' = 0 (Theory (i)): think with K = 0 runs no slot pass; the decoder reads the zero memory (its logits == the decoder on an explicit zero state)
  e  masking: a short item's decoder logits and its 2-pass state are the same alone and batched with a longer item (fp64, max rel <= 1e-10)
  f  gradflow: answer CE and aux CE each give nonzero grads on the slots, mem.* and the trunk (the encoder via the cached KV)
  g  the s170 keys (Theory (ii)): a 2-step GPU run (argv[2]) with the eval: s170.acc_by_n_kprime has K' 0 / 1 / 8 / 16 / 24 / 32 and n+2, readslots.L_mean,
     compute_by_n present
  h  memory: mem_check at the cell's args (s185_rms_s1's + --arm readslots), chunk 4 (bs 64 micro 16, eval_bs 8), vs --arm inject --cache_lower; the ratio
     times b280's cloud peak
    python scratch/gate_readslots.py <pre nested170.py> [cuda dev]   -> boxes/runs_fsq/gate_readslots/gate.json"""
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
import boxes.nested170 as N  # noqa: E402
from boxes import s5  # noqa: E402

GD = os.path.join(ROOT, "boxes", "runs_fsq", "gate_readslots"); OUT = {}; SP = os.environ.get("GATE_TMP", os.path.join(ROOT, "scratch", "_gate_rs_tmp"))
BASE = ["--think_k", "2", "--aux_step", "1.0", "--aux_fade", "48000,48000", "--fast_add", "--add_norm", "rms", "--seed", "1", "--bs", "4", "--per_n", "2",
        "--serial_pool", "40", "--log_every", "1", "--device", "cpu", "--steps", "48000", "--ckpt_every", "2"]
CELL = ["--think_k", "8", "--aux_step", "1.0", "--aux_fade", "48000,48000", "--fast_add", "--seed", "1", "--kprimes", "0,1,4,8,12,16,24,32", "--path_indep", "--add_norm", "rms"]


def dump():
    os.makedirs(GD, exist_ok=True); json.dump(OUT, open(os.path.join(GD, "gate.json"), "w", newline="\n"), indent=1, default=str)


def run(mod, out, args, need_ckpt=True, threads=1, env_dev=None):
    shutil.rmtree(out, ignore_errors=True)
    code = "import torch,sys,runpy;torch.set_num_threads(%d);sys.argv=['x']+sys.argv[1:];runpy.run_module('%s',run_name='__main__')" % (threads, mod)
    r = subprocess.run([sys.executable, "-c", code] + args + ["--out", out], cwd=ROOT, capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": ROOT, "OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads)})
    assert r.returncode == 0, (mod, args, r.stderr[-2500:])
    L = [json.loads(l) for l in open(os.path.join(out, "log.jsonl"))]
    return L, (torch.load(os.path.join(out, "ckpt_last.pt"), weights_only=False) if need_ckpt else None)


def teq(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and set(a) == set(b) and all(teq(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(teq(x, y) for x, y in zip(a, b))
    return a == b


def main():
    os.makedirs(SP, exist_ok=True); pre = os.path.join(ROOT, "boxes", "_pre_rs_nested170.py"); shutil.copy(sys.argv[1], pre); dev = sys.argv[2] if len(sys.argv) > 2 else "cuda:1"
    try:
        a4 = ["--arm", "inject"] + BASE + ["--time_steps", "4"]
        ln, cn = run("boxes.nested170", os.path.join(SP, "new"), a4); lp, cp = run("boxes._pre_rs_nested170", os.path.join(SP, "pre"), a4)
        OUT["a"] = {"log_equal": [{k: v for k, v in r.items() if k != "sec"} for r in ln] == [{k: v for k, v in r.items() if k != "sec"} for r in lp],
                    "model_equal": teq(cn["model"], cp["model"]), "opt_equal": teq(cn["opt"], cp["opt"])}
        OUT["a_pass"] = all(OUT["a"].values()); dump(); print("a", OUT["a_pass"], flush=True)
        lb, _ = run("boxes.nested170", os.path.join(SP, "rs1"), ["--arm", "readslots"] + BASE + ["--time_steps", "1"], need_ckpt=False)
        ab = json.load(open(os.path.join(SP, "rs1", "args.json")))
        OUT["b"] = {"rc": 0, "args": {k: ab.get(k) for k in ("arm", "add_norm", "n_slots", "aux_step", "think_k")}, "first_loss": [r.get("loss") for r in lb if "loss" in r][:1]}
        OUT["b_pass"] = ab.get("arm") == "readslots" and ab.get("add_norm") == "rms" and ab.get("n_slots") == 8; dump(); print("b", OUT["b_pass"], flush=True)
        DM.FAST_ADD = True; g = s5.load_gpt2("cpu"); torch.manual_seed(1); m = N.ReadSlotsN(g, 6, "top", 6, 6, "rms")
        with torch.no_grad():
            m.mem.wo.weight.normal_(0, 0.02); m.mem.fade.fill_(1.0); m.mem.gamma.fill_(0.5)
        TE = N.test_set(1000); its = [it for n_ in (2, 4, 6) for it in [x for x in TE if x["n"] == n_][:2]]; ac = lambda: torch.autocast("cpu", enabled=False)   # noqa: E731
        m.n_encode = m.n_slot_pass = 0; la, ax, _, _ = N.kmatch_aux_loss(m, its, [4] * len(its), "cpu", ac)
        OUT["c"] = {"n_encode": m.n_encode, "n_slot_pass": m.n_slot_pass}; OUT["c_pass"] = m.n_encode == 1 and m.n_slot_pass == 4
        with torch.no_grad():
            x, msk = m.think_inputs(its, "cpu"); n0 = m.n_slot_pass; S0 = m.think(m.mem.zero_state(len(its), "cpu"), x, msk, 0)
            l0 = m.dec_logits(S0, its, "cpu"); lz = m.dec_logits(torch.zeros_like(S0), its, "cpu")
        OUT["d"] = {"slot_passes_at_k0": m.n_slot_pass - n0, "state_all_zero": bool((S0 == 0).all()), "logits_eq_zero_state": bool(torch.equal(l0, lz)), "finite": bool(torch.isfinite(l0).all())}
        OUT["d_pass"] = OUT["d"]["slot_passes_at_k0"] == 0 and OUT["d"]["state_all_zero"] and OUT["d"]["logits_eq_zero_state"] and OUT["d"]["finite"]
        _f = torch.Tensor.float; torch.Tensor.float = lambda self, *a, **k: self.double()
        try:
            md = m.double(); short = [x for x in TE if x["n"] == 2][0]; long_ = [x for x in TE if x["n"] == 8][0]
            with torch.no_grad():
                xa, ma = md.think_inputs([short], "cpu"); Sa = md.think(md.mem.zero_state(1, "cpu").double(), xa, ma, 2); la_ = md.dec_logits(Sa, [short], "cpu")
                bb = [short, long_]; xb, mb = md.think_inputs(bb, "cpu"); Sb = md.think(md.mem.zero_state(2, "cpu").double(), xb, mb, 2); lb_ = md.dec_logits(Sb, bb, "cpu")
            rel = lambda u, v: float((u - v).abs().max() / v.abs().max().clamp_min(1e-300))   # noqa: E731
            OUT["e"] = {"state_rel": rel(Sb[0], Sa[0]), "logits_rel": rel(lb_[0], la_[0]), "len_short": int(ma.sum()), "len_long": int(mb[1].sum())}
            OUT["e_pass"] = OUT["e"]["state_rel"] <= 1e-10 and OUT["e"]["logits_rel"] <= 1e-10
        finally:
            torch.Tensor.float = _f
        m = m.float(); gr = {}
        for term in ("answer", "aux"):
            m.zero_grad(set_to_none=True); la, ax, _, _ = N.kmatch_aux_loss(m, its, [4] * len(its), "cpu", ac); (la if term == "answer" else ax).backward()
            gg = {n_: p_.grad for n_, p_ in m.named_parameters() if p_.grad is not None}
            gr[term] = {"slots": float(gg["slots"].abs().sum()) if "slots" in gg else 0.0, "mem": float(sum(v.abs().sum() for k, v in gg.items() if k.startswith("mem."))),
                        "trunk": float(sum(v.abs().sum() for k, v in gg.items() if k.startswith("g.")))}
        OUT["f"] = gr; OUT["f_pass"] = all(v > 0 for t in gr.values() for v in t.values()); dump(); print("c", OUT["c_pass"], "d", OUT["d_pass"], "e", OUT["e_pass"], "f", OUT["f_pass"], flush=True)
        og = os.path.join(SP, "rs_eval"); shutil.rmtree(og, ignore_errors=True)
        r = subprocess.run([sys.executable, "-m", "boxes.nested170", "--arm", "readslots"] + CELL + ["--steps", "2", "--bs", "8", "--per_n", "20", "--serial_pool", "40", "--eval_bs", "64",
                            "--micro", "1", "--device", dev, "--out", og], cwd=ROOT, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": ROOT})
        ev = json.load(open(os.path.join(og, "eval.json"))) if r.returncode == 0 and os.path.exists(os.path.join(og, "eval.json")) else {}
        kp = sorted((ev.get("s170") or {}).get("acc_by_n_kprime", {}).keys())
        OUT["g"] = {"rc": r.returncode, "err": r.stderr[-800:] if r.returncode else "", "kprime_keys": kp, "readslots": ev.get("readslots"), "has_compute_by_n": "compute_by_n" in ev,
                    "step_recover": (ev.get("s170") or {}).get("step_recover_info"), "s175_keys": sorted((ev.get("s175") or {}).keys())}
        OUT["g_pass"] = r.returncode == 0 and all(k in kp for k in ("0", "1", "8", "16", "24", "32", "n+2")) and bool(ev.get("readslots")) and OUT["g"]["has_compute_by_n"]
        dump(); print("g", OUT["g_pass"], flush=True)
        pk = {}
        for tag, ex in (("inject_cached", ["--arm", "inject", "--cache_lower"]), ("readslots", ["--arm", "readslots"])):
            o_ = os.path.join(SP, "mem_" + tag)
            r = subprocess.run([sys.executable, "-m", "boxes.nested170"] + ex + CELL + ["--steps", "48000", "--bs", "64", "--micro", "16", "--eval_bs", "8", "--mem_check", "--device", dev, "--out", o_],
                               cwd=ROOT, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": ROOT})
            pk[tag] = json.load(open(os.path.join(o_, "mem_check.json"))) if r.returncode == 0 else {"err": r.stderr[-600:]}
        ratio = pk["readslots"]["peak_gib"] / pk["inject_cached"]["peak_gib"] if all("peak_gib" in v for v in pk.values()) else None
        OUT["h"] = {"chunk4": pk, "ratio_rs_over_injc": ratio, "proj_gib_from_b280_15.37": 15.372 * ratio if ratio else None, "gpu": torch.cuda.get_device_name(dev)}
        OUT["h_pass"] = bool(ratio) and OUT["h"]["proj_gib_from_b280_15.37"] < 21.0; dump(); print("h", OUT["h_pass"], flush=True)
    finally:
        os.remove(pre); shutil.rmtree(SP, ignore_errors=True)
        OUT["all_pass"] = bool(all(OUT.get(k + "_pass") for k in "abcdefgh")); dump(); print(json.dumps(OUT, default=str)[:3000], flush=True)


if __name__ == "__main__":
    main()
