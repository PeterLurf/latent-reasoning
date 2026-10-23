"""§228 LAT-OW build gates (LAT_v1 §8 / draft §8; for Audit's G-D). -> scratch/gate_s228/<part>.json ("pass"). Run dirs under scratch/_tmp/s228.
  port     port-kernel identity: nested170 --arm inject (§184's args: think_k 8, aux_step 1, aux_fade 48000,48000, aux_reweight, fast_add) vs inject_ow
           --src nested (nested170's items / EXCL / 17001 calibration rendered to token lists), fp32, det wrapper, --sched_scale 0.02 (write mix
           40 -> 120, fade 10), 50 steps, micro 8 (8 GB card): ckpt_last.pt at step 50 - every model tensor and every AdamW state tensor torch.equal; the g_add
           calibration equal. Both uncached and --cache_lower (nested170's --cache_lower).
  cache64  --cache_lower on OW shapes in float64 (CPU): a perturbed InjectOW (wo / gate / s_add random, gamma 0.5, fade 1), OW TOPO items d 1 / 4 / 8,
           K 8: cached vs uncached S_t, aux logits per pass, dec logits, loss and every parameter's gradient; max relative difference <= 1e-10 (Tensor.float -> double, as gate_cache_lower_n170).
  gradflow per term (fp64 CPU, d 3 items, K 8): the answer CE and the aux CE each backward alone: nonzero finite grad on every GPT-2 tensor and every
           memory tensor at the perturbed point; at the true init (wo = 0) the write-path tensors have zero answer-CE grad (expected, reported); the
           AUX_W weights applied per pass (t <= d) checked against nested170.AUX_W.
  plain    the plain path: --arm direct, no new flags, construct + 1 step (stop_at 1) on the 4060 -> rc 9, args.json, 1 log record.
  resume   inject d_max 2 on the 4060, det wrapper, fp32, sched_scale 0.02: run A to stop_at 40 (ckpt 20, 40); run B resumes A's step-20 ckpt to
           stop_at 40: B's ckpt at 40 = A's (model + opt + rng + bins, torch.equal) and the step-40 bin record identical.
  f2       memory / timing on the 4060 at the pilot shapes: inject d_max 2 and 4 (--auto_micro --mem_budget 7.0, --time_steps 30) and direct d_max 2 / 4.
    python -B scratch/gate_s228.py port cache64 gradflow plain resume f2"""
import json
import math
import os
import shutil
import subprocess
import sys

ROOT = "D:/Latent"; sys.path.insert(0, ROOT)
OUT = ROOT + "/scratch/gate_s228"; TMP = ROOT + "/scratch/_tmp/s228"; DEV = "cuda:0"
ENV = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
S184 = ["--arm", "inject", "--think_k", "8", "--aux_step", "1.0", "--aux_fade", "48000,48000", "--aux_reweight", "--fast_add"]
LAT2 = S184 + ["--cache_lower", "--d_max", "2", "--steps", "8000"]


def sh(cmd, log):
    with open(log, "w") as f_:
        r = subprocess.run(cmd, stdout=f_, stderr=subprocess.STDOUT, cwd=ROOT, env=ENV)
    return r.returncode


def fresh(d):
    if os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(d)
    return d


def ck_equal(p1, p2, keys=("model", "opt")):
    import torch
    a, b = torch.load(p1, map_location="cpu", weights_only=False), torch.load(p2, map_location="cpu", weights_only=False); bad = []
    def cmp(x, y, path):
        if isinstance(x, torch.Tensor):
            if not (isinstance(y, torch.Tensor) and x.shape == y.shape and torch.equal(x, y)):
                bad.append(path)
        elif isinstance(x, dict):
            if not isinstance(y, dict) or set(x) != set(y):
                bad.append(path + " (keys)"); return
            for k in x:
                cmp(x[k], y[k], path + "/" + str(k))
        elif isinstance(x, (list, tuple)):
            if not isinstance(y, (list, tuple)) or len(x) != len(y):
                bad.append(path + " (len)"); return
            for i, (u, v) in enumerate(zip(x, y)):
                cmp(u, v, "%s[%d]" % (path, i))
        elif x != y and not (isinstance(x, float) and math.isnan(x) and math.isnan(y)):
            bad.append(path)
    for k in keys:
        cmp(a[k], b[k], k)
    n = sum(1 for _ in a["model"]); return {"step": [a.get("step"), b.get("step")], "n_model_tensors": n, "mismatch": bad[:20], "n_mismatch": len(bad)}


def port():
    o = {}
    for tag, extra in (("uncached", []), ("cached", ["--cache_lower"])):
        common = S184 + extra + ["--fp32", "--micro", "8", "--sched_scale", "0.02", "--time_steps", "50", "--ckpt_every", "50", "--steps", "48000", "--device", DEV]
        dn = fresh(TMP + "/port_n170_" + tag); di = fresh(TMP + "/port_iow_" + tag)
        rn = sh([sys.executable, "-B", "scratch/s222_det_wrap.py", "boxes/nested170.py"] + common + ["--out", dn], dn + "/stdout.log")
        ri = sh([sys.executable, "-B", "scratch/s222_det_wrap.py", "boxes/inject_ow.py", "--src", "nested"] + common + ["--out", di], di + "/stdout.log")
        r = {"rc": [rn, ri]}
        if os.path.exists(dn + "/ckpt_last.pt") and os.path.exists(di + "/ckpt_last.pt"):
            r.update(ck_equal(dn + "/ckpt_last.pt", di + "/ckpt_last.pt"))
            r["g_add_init"] = [json.load(open(dn + "/s170_init.json"))["g_add_init"], json.load(open(di + "/s228_init.json"))["g_add_init"]]
        r["pass"] = rn == 0 and ri == 0 and r.get("n_mismatch") == 0 and r.get("step") == [50, 50] and r["g_add_init"][0] == r["g_add_init"][1]
        o[tag] = r
    o["pass"] = all(v["pass"] for v in o.values() if isinstance(v, dict)); return o


def perturbed(dt, dev="cpu", seed=5):
    import torch
    import boxes.deltamem as DM
    import boxes.inject_ow as IO
    DM.FAST_ADD = True; torch.manual_seed(1); g = IO.load_trunk(dev, {"scratch": ""}); m = IO.InjectOW(g, 6, "top", 6, 6, None).to(dev).to(dt)
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        m.mem.wo.weight.copy_(torch.randn(m.mem.wo.weight.shape, generator=gen, dtype=dt) * 0.02); m.mem.wo.bias.copy_(torch.randn(768, generator=gen, dtype=dt) * 0.02)
        m.mem.gate.weight.copy_(torch.randn(m.mem.gate.weight.shape, generator=gen, dtype=dt) * 0.02); m.mem.s_add.fill_(-1.0); m.mem.gamma.fill_(0.5); m.mem.fade.fill_(1.0)
    return m, g


def ow_items(ds, n, seed=22997):
    import boxes.inject_ow as IO
    import boxes.ow_words as OW
    s = OW.Stream(seed, frames=OW.FRAME_TRAIN, **IO.OWKW); return {d: [IO.tok_ow(s.draw(d)) for _ in range(n)] for d in ds}


def loss_terms(m, items, K, hl_on, reweight=True):
    """inject_ow.loss_items' inject branch (answer CE, weighted aux) with the cache on / off; -> (la, aux, S list, aux logits list, dec logits, weights)."""
    import torch
    import torch.nn.functional as Fn
    import boxes.inject_ow as IO
    dev = "cpu"; tgt = torch.tensor(IO.NUM_IDS[[it["ans"] for it in items]]); x, msk = m.think_inputs(items, dev); per = [[] for _ in items]; Ss, AL, W = [], [], []
    S = m.mem.zero_state(len(items), dev).to(next(m.parameters()).dtype); hl = m.lower(x) if hl_on else None
    for t in range(1, K + 1):
        S = m.write(S, x, msk, hl); Ss.append(S)
        idx = [i for i, it in enumerate(items) if len(it["aux"]) >= t]
        if idx:
            lg_a, y_a = m.aux_logits(S[idx], [items[i] for i in idx], t, dev); AL.append(lg_a)
            ce = Fn.cross_entropy(lg_a.transpose(1, 2), y_a, ignore_index=-100, reduction="none").sum(1) / (y_a != -100).sum(1)
            for r_, i in enumerate(idx):
                per[i].append(ce[r_] * IO.AUX_W[t - 1] if reweight else ce[r_]); W.append((t, IO.AUX_W[t - 1]))
    lg = m.dec_logits(S, items, dev); la = Fn.cross_entropy(lg, tgt); aux = torch.stack([torch.stack(p_).mean() for p_ in per if p_]).mean()
    return la, aux, Ss, AL, lg, W


class F64:
    """gate_cache_lower_n170's trick: Tensor.float -> double inside the block (DeltaMem casts its input with .float())."""
    def __enter__(self):
        import torch
        self.f = torch.Tensor.float; torch.Tensor.float = lambda self_, *a, **k: self_.double()

    def __exit__(self, *e):
        import torch
        torch.Tensor.float = self.f


def cache64():
    with F64():
        return _cache64()


def _cache64():
    import torch
    torch.set_num_threads(2); m, g = perturbed(torch.float64); its = ow_items((1, 4, 8), 2); o = {"by_d": {}}; ok = True
    for d, it_ in its.items():
        res = []
        for hl_on in (False, True):
            m.zero_grad(set_to_none=True); la, aux, Ss, AL, lg, _ = loss_terms(m, it_, 8, hl_on); (la + aux).backward()
            res.append((la.detach(), aux.detach(), [s.detach() for s in Ss], [x.detach() for x in AL], lg.detach(), {n: p.grad.detach().clone() for n, p in m.named_parameters() if p.grad is not None}))
        rel = lambda u, v: float((u - v).abs().max() / v.abs().max().clamp_min(1e-300))  # noqa: E731
        (a0, x0, S0, A0, L0, G0), (a1, x1, S1, A1, L1, G1) = res
        r = {"loss_rel": rel(a1 + x1, a0 + x0), "S_rel_max": max(rel(u, v) for u, v in zip(S1, S0)), "aux_logit_rel_max": max(rel(u, v) for u, v in zip(A1, A0)),
             "dec_logit_rel": rel(L1, L0), "grad_rel_max": max(rel(G1[n], G0[n]) for n in G0 if G0[n].abs().max() > 0), "n_grads": len(G0), "same_grad_keys": set(G0) == set(G1),
             "think_len": [len(i_["think"]) for i_ in it_]}
        r["pass"] = r["same_grad_keys"] and max(r["loss_rel"], r["S_rel_max"], r["aux_logit_rel_max"], r["dec_logit_rel"], r["grad_rel_max"]) <= 1e-10
        o["by_d"][str(d)] = r; ok = ok and r["pass"]
    o["pass"] = ok; return o


def gradflow():
    with F64():
        return _gradflow()


def _gradflow():
    import torch
    import boxes.inject_ow as IO
    import boxes.nested170 as N7
    torch.set_num_threads(2); its = ow_items((3,), 4)[3]; o = {}
    for tag in ("perturbed", "init"):
        if tag == "perturbed":
            m, g = perturbed(torch.float64)
        else:
            import boxes.deltamem as DM
            DM.FAST_ADD = True; torch.manual_seed(1); g = IO.load_trunk("cpu", {"scratch": ""}); m = IO.InjectOW(g, 6, "top", 6, 6, None).double()
            with torch.no_grad():
                m.mem.gamma.fill_(0.5); m.mem.fade.fill_(1.0)
        r = {}
        for term in ("answer_ce", "aux_ce"):
            m.zero_grad(set_to_none=True); la, aux, _, _, _, W = loss_terms(m, its, 8, True); (la if term == "answer_ce" else aux).backward()
            gr = {n: (float(p.grad.norm()) if p.grad is not None else None) for n, p in m.named_parameters()}
            okf = lambda v: v is not None and v > 0 and math.isfinite(v)  # noqa: E731
            gp = {n: v for n, v in gr.items() if n.startswith("g.")}; mp = {n: v for n, v in gr.items() if n.startswith("mem.")}
            r[term] = {"value": float(la if term == "answer_ce" else aux), "n_gpt2": len(gp), "gpt2_nonzero": sum(okf(v) for v in gp.values()), "n_mem": len(mp),
                       "mem_nonzero": sum(okf(v) for v in mp.values()), "zero_or_none": sorted(n for n, v in gr.items() if not okf(v))[:30]}
        r["aux_weights"] = sorted(set(W)); r["aux_weights_ok"] = all(abs(w - N7.AUX_W[t - 1]) == 0 for t, w in W) and sorted({t for t, _ in W}) == [1, 2, 3]
        o[tag] = r
    p = o["perturbed"]
    o["pass"] = (all(p[t]["gpt2_nonzero"] == p[t]["n_gpt2"] == 148 and p[t]["mem_nonzero"] == p[t]["n_mem"] for t in ("answer_ce", "aux_ce")) and p["aux_weights_ok"]
                 and o["init"]["aux_weights_ok"])
    o["note"] = "at the true init wo = 0: the read adds wo's zero map, so the answer CE gives no gradient to the write path (expected; nested170's init)"
    return o


def plain():
    d = fresh(TMP + "/plain"); rc = sh([sys.executable, "-B", "-m", "boxes.inject_ow", "--arm", "direct", "--d_max", "2", "--steps", "8000", "--stop_at", "1", "--ckpt_every", "0",
                                         "--log_every", "1", "--device", DEV, "--out", d], d + "/stdout.log")
    lg = [json.loads(l_) for l_ in open(d + "/log.jsonl")] if os.path.exists(d + "/log.jsonl") else []
    o = {"rc": rc, "n_log": len(lg), "first": lg[0] if lg else None, "args": os.path.exists(d + "/args.json")}
    o["pass"] = rc == 9 and o["args"] and any(r_.get("step") == 1 and r_.get("loss") is not None for r_ in lg); return o


def resume():
    da = fresh(TMP + "/resume_A"); db = fresh(TMP + "/resume_B")
    common = LAT2 + ["--fp32", "--micro", "8", "--sched_scale", "0.02", "--ckpt_every", "20", "--log_every", "10", "--device", DEV, "--stop_at", "40"]
    ra = sh([sys.executable, "-B", "scratch/s222_det_wrap.py", "boxes/inject_ow.py"] + common + ["--out", da], da + "/stdout.log")
    shutil.copy(da + "/ckpt_prev.pt", db + "/ck20.pt"); shutil.copy(da + "/args.json", db + "/args.json")
    rb = sh([sys.executable, "-B", "scratch/s222_det_wrap.py", "boxes/inject_ow.py"] + common + ["--resume", db + "/ck20.pt", "--out", db], db + "/stdout.log")
    o = {"rc": [ra, rb]}
    if os.path.exists(da + "/ckpt_last.pt") and os.path.exists(db + "/ckpt_last.pt"):
        o.update(ck_equal(da + "/ckpt_last.pt", db + "/ckpt_last.pt", keys=("model", "opt", "rng_d", "stream_rng", "expo", "d2_bins", "ent", "gn_log")))
        rd = lambda p: [json.loads(l_) for l_ in open(p)]  # noqa: E731
        ba = [r_ for r_ in rd(da + "/log.jsonl") if "by_d" in r_ and r_["step"] == 40]; bb = [r_ for r_ in rd(db + "/log.jsonl") if "by_d" in r_ and r_["step"] == 40]
        la = [r_ for r_ in rd(da + "/log.jsonl") if r_.get("step") in (30, 40) and "loss" in r_]; lb = [r_ for r_ in rd(db + "/log.jsonl") if r_.get("step") in (30, 40) and "loss" in r_]
        strip = lambda r_: {k: v for k, v in r_.items() if k != "sec"}  # noqa: E731
        o["bin40_equal"] = bool(ba and bb and ba[-1] == bb[-1]); o["log_30_40_equal"] = [strip(x) for x in la] == [strip(x) for x in lb]
    o["pass"] = ra == 9 and rb == 9 and o.get("n_mismatch") == 0 and o.get("bin40_equal") and o.get("log_30_40_equal"); return o


def f2():
    o = {}
    for nm, fl in (("lat_d2", LAT2), ("lat_d4", S184 + ["--cache_lower", "--d_max", "4", "--steps", "16000"]),
                   ("dir_d2", ["--arm", "direct", "--d_max", "2", "--steps", "8000"]), ("dir_d4", ["--arm", "direct", "--d_max", "4", "--steps", "16000"])):
        d = fresh(TMP + "/f2_" + nm); rc = sh([sys.executable, "-B", "-m", "boxes.inject_ow"] + fl + ["--auto_micro", "--mem_budget", "7.0", "--time_steps", "30", "--log_every", "10",
                                                                      "--device", DEV, "--out", d], d + "/stdout.log")
        r = {"rc": rc}
        for f_ in ("mem_check.json", "timing.json"):
            if os.path.exists(os.path.join(d, f_)):
                r[f_[:-5]] = json.load(open(os.path.join(d, f_)))
        o[nm] = r
    o["pass"] = all(v.get("rc") == 0 and "timing" in v for v in o.values() if isinstance(v, dict)); return o


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True); os.makedirs(TMP, exist_ok=True)
    for part in [x for x in sys.argv[1:] if x in ("port", "cache64", "gradflow", "plain", "resume", "f2")]:
        r = {"port": port, "cache64": cache64, "gradflow": gradflow, "plain": plain, "resume": resume, "f2": f2}[part]()
        json.dump(r, open(os.path.join(OUT, part + ".json"), "w", newline="\n"), indent=1, default=str); print(part, "pass", r["pass"], flush=True)
