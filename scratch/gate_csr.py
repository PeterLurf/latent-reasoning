"""§199 CSR gates (Theory 199-A, 0420e8b6; nested170 --csr / read_parts / csr_lams / csr_terms / --keep_ckpt_steps):
  g1  flags-off == the pre-change module (argv[1]) bitwise: 1 CPU thread, inject + aux + rms, --time_steps 4 (log, ckpt model + opt)
  g1b (think_k 4 since run 2: at think_k 2 pre needs K >= 3 and post n <= K - 1 = 1, so the term is 0 and step 3 cannot differ)
      CSR inactive before its start: --csr 0.1,-0.0513,-0.6931,0.05,3 vs flags-off -> log records at steps 1-2 (loss, answer_ce, aux_ce) bitwise equal,
      step 3 differs; csr_* fields from step 3 only; --keep_ckpt_steps 2,4 writes ckpt_model_2.pt / ckpt_model_4.pt ({model, step}) loadable by the probe
  g2  gradflow of the CSR term alone (in-process, rms model, fade 1): nonzero finite grads on mem.wo.weight and the delta-gate rows of mem.gate; LN_w's grad
      EXACTLY 0 while gamma = 0 (open-loop lemma) and nonzero at gamma 0.5
  g1c (added) think_k 4, CSR from step 1: csr_* fields finite every step, the pre median present (run 1's g1b showed NaN at think_k 2: fixed by capping pre at K-1)
  g3  fixtures: a kappa-geometric R sequence (R_{t+1} - R_t = kappa^t v) gives lambda_t = log kappa at eps0 = 0 and the closed form with eps0 = 1e-3 (|err| <= 1e-3,
      kappa 0.5 / 0.9 / 1.1); a constant-increment
      additive state S_add,t = t dS through the REAL rms add_read path (read_parts) gives lambda_add_t = 0 for t >= 2 (|err| <= 1e-3)
  g4  read_parts: R_add + R_del + fade * b_o == read_fixed(mem, h6, S) - h6 (fp32, max rel <= 1e-5) on random nonzero states
  g5  memory (argv[2], default cuda:1): mem_check at s185_rms_s1's args, chunk 4 (bs 64 micro 16, eval_bs 8), with --csr (the check runs the CSR path) vs
      without; the ratio x s185_rms_s1's cloud peak (mem_check.json)
    python scratch/gate_csr.py <pre nested170.py> [cuda dev]   -> boxes/runs_fsq/gate_csr/gate.json"""
import json
import math
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
os.environ.setdefault("BOXES_NB", "5")
import torch  # noqa: E402

torch.set_num_threads(3)
import boxes.deltamem as DM  # noqa: E402
import boxes.nested170 as N  # noqa: E402
from boxes import s5  # noqa: E402

GD = os.path.join(ROOT, "boxes", "runs_fsq", "gate_csr"); OUT = {}; SP = os.environ.get("GATE_TMP", os.path.join(ROOT, "scratch", "_gate_csr_tmp"))
BASE = ["--arm", "inject", "--think_k", "2", "--aux_step", "1.0", "--aux_fade", "48000,48000", "--fast_add", "--add_norm", "rms", "--seed", "1", "--bs", "4",
        "--per_n", "2", "--serial_pool", "40", "--log_every", "1", "--device", "cpu", "--steps", "48000", "--ckpt_every", "2", "--time_steps", "4"]
CSR = "0.1,-0.0513,-0.6931,0.05"
BASE4 = [v_ if not (v_ == "2" and BASE[i_ - 1] == "--think_k") else "4" for i_, v_ in enumerate(BASE)]   # run 2: at think_k 2 the CSR term is structurally 0


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


def main():
    os.makedirs(SP, exist_ok=True); pre = os.path.join(ROOT, "boxes", "_pre_csr_nested170.py"); shutil.copy(sys.argv[1], pre); dev = sys.argv[2] if len(sys.argv) > 2 else "cuda:1"
    try:
        ln, cn = run("boxes.nested170", os.path.join(SP, "new"), BASE); lp, cp = run("boxes._pre_csr_nested170", os.path.join(SP, "pre"), BASE)
        OUT["g1"] = {"log_equal": [{k: v for k, v in r.items() if k != "sec"} for r in ln] == [{k: v for k, v in r.items() if k != "sec"} for r in lp],
                     "model_equal": teq(cn["model"], cp["model"]), "opt_equal": teq(cn["opt"], cp["opt"])}
        OUT["g1_pass"] = all(OUT["g1"].values()); dump(); print("g1", OUT["g1_pass"], flush=True)
        ln, _ = run("boxes.nested170", os.path.join(SP, "off4"), BASE4)       # g1b's flags-off comparator at think_k 4
        dc = os.path.join(SP, "csr"); lc, _ = run("boxes.nested170", dc, BASE4 + ["--csr", CSR + ",3", "--keep_ckpt_steps", "2,4"])
        tr = lambda L: {r["step"]: (r["loss"], r.get("answer_ce"), r.get("aux_ce")) for r in L if "loss" in r}   # noqa: E731
        A, B = tr(ln), tr(lc); csr_steps = [r["step"] for r in lc if "csr_loss" in r]
        kc = {s_: os.path.exists(os.path.join(dc, "ckpt_model_%d.pt" % s_)) for s_ in (2, 4)}
        ck2 = torch.load(os.path.join(dc, "ckpt_model_2.pt"), weights_only=False) if kc[2] else {}
        OUT["g1b"] = {"steps_1_2_equal": A[1] == B[1] and A[2] == B[2], "step3_differs": A[3] != B[3], "csr_field_steps": csr_steps, "keep_ckpt": kc,
                      "ckpt_model_2_keys": sorted(ck2.keys()) if ck2 else None, "csr_sample": [{k: r.get(k) for k in ("step", "csr_loss", "csr_lam_pre_med", "csr_lam_post_med")} for r in lc if "csr_loss" in r]}
        fin_ = lambda L, ks: all(r.get(k) is None or math.isfinite(r[k]) for r in L for k in ks)  # noqa: E731
        OUT["g1b"]["csr_fields_finite"] = fin_(lc, ("loss", "csr_loss", "csr_lam_pre_med", "csr_lam_post_med"))
        OUT["g1b_pass"] = (OUT["g1b"]["steps_1_2_equal"] and OUT["g1b"]["step3_differs"] and csr_steps == [3, 4] and all(kc.values()) and ck2.get("step") == 2 and "model" in ck2
                           and OUT["g1b"]["csr_fields_finite"] and all(r.get("csr_loss") is not None for r in lc if "csr_loss" in r))
        dump(); print("g1b", OUT["g1b_pass"], flush=True)
        # g1c (added after the first run's g1b NaN at think_k 2): think_k 4 so both pre (t 2..min(n-1, 3)) and post ranges exist; CSR on from step 1
        lk, _ = run("boxes.nested170", os.path.join(SP, "csr4"), BASE4 + ["--csr", CSR + ",1"])
        rk = [r for r in lk if "csr_loss" in r]
        OUT["g1c"] = {"think_k": 4, "csr_steps": [r["step"] for r in rk], "fields_finite": fin_(lk, ("loss", "csr_loss", "csr_lam_pre_med", "csr_lam_pre_med_add", "csr_lam_pre_med_del", "csr_lam_post_med")),
                      "pre_present": all(r.get("csr_lam_pre_med") is not None for r in rk), "post_present": all(r.get("csr_lam_post_med") is not None for r in rk),
                      "sample": [{k: r.get(k) for k in ("step", "csr_loss", "csr_lam_pre_med", "csr_lam_post_med")} for r in rk]}
        OUT["g1c_pass"] = bool(rk and OUT["g1c"]["csr_steps"] == [1, 2, 3, 4] and OUT["g1c"]["fields_finite"] and OUT["g1c"]["pre_present"])
        dump(); print("g1c", OUT["g1c_pass"], flush=True)
        DM.FAST_ADD = True; g = s5.load_gpt2("cpu"); torch.manual_seed(1); m = N.InjectN(g, 6, "top", 6, 6, "rms")
        with torch.no_grad():
            m.mem.wo.weight.normal_(0, 0.02); m.mem.fade.fill_(1.0)
        TE = N.test_set(1000); its = [it for n_ in (2, 4, 6) for it in [x for x in TE if x["n"] == n_][:2]]; ns = [it["n"] for it in its]; Kc = 8
        x, msk = m.think_inputs(its, "cpu")

        def csr_alone():
            with torch.no_grad():
                h6 = DM.gpt_lower(g, g.embed(x), m.mem.layer)
            S = m.mem.zero_state(len(its), "cpu"); ra, rd = N.read_parts(m.mem, h6, S); R = [ra + rd]
            for _ in range(Kc):
                S = m.write(S, x, msk); ra, rd = N.read_parts(m.mem, h6, S); R.append(ra + rd)
            return N.csr_terms(N.csr_lams(R, msk), ns, Kc, -0.0513, -0.6931, 0.05)
        gr = {}
        for gm in (0.0, 0.5):
            with torch.no_grad():
                m.mem.gamma.fill_(gm)
            m.zero_grad(set_to_none=True); l_ = csr_alone(); l_.backward()
            gw = m.mem.wo.weight.grad; gg = m.mem.gate.weight.grad; lnw = [p_.grad for p_ in m.mem.ln_w.parameters()]
            gr[str(gm)] = {"csr": float(l_), "wo": float(gw.abs().sum()) if gw is not None else 0.0, "wo_finite": bool(torch.isfinite(gw).all()) if gw is not None else False,
                           "gate_delta_rows": float(gg.abs().sum()) if gg is not None else 0.0,
                           "ln_w": float(sum(g_.abs().sum() for g_ in lnw if g_ is not None)) if any(g_ is not None for g_ in lnw) else 0.0,
                           "ln_w_grad_is_none_or_zero": all(g_ is None or bool((g_ == 0).all()) for g_ in lnw)}
        OUT["g2"] = gr; OUT["g2_pass"] = (gr["0.0"]["wo"] > 0 and gr["0.0"]["wo_finite"] and gr["0.0"]["gate_delta_rows"] > 0 and gr["0.0"]["ln_w_grad_is_none_or_zero"]
                                          and gr["0.5"]["ln_w"] > 0); dump(); print("g2", OUT["g2_pass"], flush=True)
        g3 = {}
        for kap in (0.5, 0.9, 1.1):
            v = torch.randn(2, 5, 768, generator=torch.Generator().manual_seed(0)); R = [torch.zeros_like(v)]
            for t in range(8):
                R.append(R[-1] + kap ** t * v)
            lam = N.csr_lams(R, torch.ones(2, 5, dtype=torch.bool), eps0=0.0); g3["kappa_%s_eps0_0" % kap] = float((lam - math.log(kap)).abs().max())
            # with the contract's floor eps0 = 1e-3: d_t^2 = kappa^2t |v|^2 + eps0^2 |R_t|^2, |R_t| = |v| (1 - kappa^t) / (1 - kappa) (fixture amended after run 1:
            # exact log kappa holds only at eps0 = 0; at kappa 0.5, t 7 the floor moves lambda by ~0.03)
            lam1 = N.csr_lams(R, torch.ones(2, 5, dtype=torch.bool)); vv = v.pow(2).sum((1, 2)).double()
            dd = torch.stack([torch.sqrt(kap ** (2 * t) * vv + 1e-6 * ((1 - kap ** t) / (1 - kap)) ** 2 * vv + 1e-12) for t in range(8)], 1)
            g3["kappa_%s_eps0_1e-3_vs_closed_form" % kap] = float((lam1.double() - (dd[:, 1:].log() - dd[:, :-1].log())).abs().max())
        with torch.no_grad():
            h6 = torch.randn(3, 6, 768, generator=torch.Generator().manual_seed(1)); dS = torch.randn(3, 6, 64, 64, generator=torch.Generator().manual_seed(2)) * 0.1
            Rs = []
            for t in range(9):
                S = m.mem.zero_state(3, "cpu"); S[:, :6] = t * dS; ra, _ = N.read_parts(m.mem, h6, S); Rs.append(ra)
            lam_add = N.csr_lams(Rs, torch.ones(3, 6, dtype=torch.bool))
        g3["const_increment_lambda_add_t2plus_max_abs"] = float(lam_add[:, 1:].abs().max())
        OUT["g3"] = g3; OUT["g3_pass"] = all(v <= 1e-3 for k, v in g3.items()); dump(); print("g3", OUT["g3_pass"], flush=True)
        with torch.no_grad():
            m.mem.gamma.fill_(0.5); torch.nn.init.normal_(m.mem.wo.bias, 0, 0.01); h6 = torch.randn(3, 7, 768, generator=torch.Generator().manual_seed(3))
            S = m.mem.zero_state(3, "cpu"); S.normal_(0, 0.05, generator=torch.Generator().manual_seed(4)); ra, rd = N.read_parts(m.mem, h6, S)
            ref = DM.read_fixed(m.mem, h6, S) - h6; got = ra + rd + m.mem.fade * m.mem.wo.bias
        OUT["g4"] = {"max_rel": float((got - ref).abs().max() / ref.abs().max()), "bitwise": bool(torch.equal(got, ref))}; OUT["g4_pass"] = OUT["g4"]["max_rel"] <= 1e-5
        dump(); print("g4", OUT["g4_pass"], OUT["g4"], flush=True)
        pk = {}
        for tag, ex in (("off", []), ("csr", ["--csr", CSR + ",500"])):
            o_ = os.path.join(SP, "mem_" + tag)
            r = subprocess.run([sys.executable, "-m", "boxes.nested170", "--arm", "inject", "--think_k", "8", "--aux_step", "1.0", "--aux_fade", "48000,48000", "--fast_add",
                                "--add_norm", "rms", "--seed", "1", "--steps", "48000", "--bs", "64", "--micro", "16", "--eval_bs", "8", "--mem_check", "--device", dev, "--out", o_] + ex,
                               cwd=ROOT, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": ROOT})
            pk[tag] = json.load(open(os.path.join(o_, "mem_check.json"))) if r.returncode == 0 else {"err": r.stderr[-600:]}
        ratio = pk["csr"]["peak_gib"] / pk["off"]["peak_gib"] if all("peak_gib" in v for v in pk.values()) else None
        mc = json.load(open("boxes/runs_fsq/s185_rms_s1/mem_check.json"))
        OUT["g5"] = {"chunk4": pk, "ratio_csr_over_off": ratio, "s185_rms_s1_cloud": {"micro": mc.get("micro"), "peak_gib": mc.get("peak_gib")},
                     "proj_gib": mc["peak_gib"] * ratio if ratio else None, "gpu": torch.cuda.get_device_name(dev)}
        OUT["g5_pass"] = bool(ratio) and OUT["g5"]["proj_gib"] < 21.0; dump(); print("g5", OUT["g5_pass"], flush=True)
    finally:
        os.remove(pre); shutil.rmtree(SP, ignore_errors=True)
        OUT["all_pass"] = bool(all(OUT.get(k + "_pass") for k in ("g1", "g1b", "g1c", "g2", "g3", "g4", "g5"))); dump(); print(json.dumps(OUT, default=str)[:3000], flush=True)


if __name__ == "__main__":
    main()
