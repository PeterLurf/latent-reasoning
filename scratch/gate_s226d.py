"""§226 P0d build gates (Note 226-1 + Audit; Theory 10-23): codi_ps --full_ft in TREE (scratch/_tmp/s226d_tree) until committed. -> scratch/gate_s226d/<part>.json
  build    build_full's fp32 weights cast to bf16 == codi_port.build's bf16 base weights (every GPT-2 tensor; LoRA-wrapped Conv1D -> .base), the 3 new wte
           rows included; prj parameters identical; every build_full parameter requires_grad and is fp32.
  gradflow --full_ft, 4 OW-topo d 2 items, fp64 CPU, eval mode: DIRECT's answer CE gives a nonzero finite grad on every GPT-2 parameter (prj is unused
           without latents); M1 (K 2, recode) gives one on every GPT-2 parameter AND prj.
  flags_off HEAD vs TREE bitwise (scratch/gate_s226.py's flags_off with TREE, det wrapper, DEV): DIRECT / M1 / M2 / PS / PS+m2 / OW-DIRECT, 12 steps.
  mem      --mem_check with P0d's flags on the 4060 (cuda:0, the registered kernel) and --auto_micro --mem_budget 7.0: the chosen micro's peak <= 7.0.
  smoke    P0d's flags, 12 steps, eff_bs 64 (auto_micro) on the 4060: rc 9, 12 log rows, finite loss / gn, the wrap counts logged.
  args     --args_only P0d vs scratch/_tmp/s226_p0/P0b-TOPO/args.json: the non-hash, non-META diff == {full_ft, lr}.
    python scratch/gate_s226d.py build gradflow args [flags_off --device cuda:2] [mem smoke]"""
import importlib.util
import json
import math
import os
import subprocess
import sys

TREE = "D:/Latent/scratch/_tmp/s226d_tree"; OUT = "D:/Latent/scratch/gate_s226d"; TMP = "D:/Latent/scratch/_tmp/s226d_gate"; DEV = "cuda:2"
for i, v in enumerate(sys.argv):
    if v == "--tree":
        TREE = sys.argv[i + 1]
    if v == "--device":
        DEV = sys.argv[i + 1]
P0D = ["--task", "ow", "--order_mode", "topo", "--direct", "--full_ft", "--lr", "1e-4", "--cg_thr", "0.45", "--cg_cap", "6000", "--steps", "48000"]
ENV = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}


def tree_mod():
    sys.path.insert(0, TREE); sp = importlib.util.spec_from_file_location("codi_ps_t", os.path.join(TREE, "boxes", "codi_ps.py"))
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m


def build():
    import torch
    PS = tree_mod(); CP = PS.CP
    mf, pf = PS.build_full("cpu", 1); ml, pl = CP.build("cpu", 1); o = {}
    fl = dict(mf.named_parameters()); base = {}
    for n, p_ in ml.named_parameters():
        base[n.replace(".base.", ".")] = p_
    for n, mod in ml.named_modules():                                       # frozen LoRA bases are parameters of .base (requires_grad False) - covered above
        pass
    keys_f = {n for n in fl}; keys_l = {n for n in base if ".lora_" not in n}
    o["keys_equal"] = keys_f == keys_l; o["missing"] = sorted(keys_f ^ keys_l)[:10]
    eq = {n: bool(torch.equal(fl[n].detach().to(torch.bfloat16), base[n].detach())) for n in keys_f & keys_l}
    o["n_tensors"] = len(eq); o["bf16_equal"] = sum(eq.values()); o["unequal"] = [n for n, v in eq.items() if not v][:10]
    o["prj_equal"] = all(torch.equal(a, b) for a, b in zip(pf.state_dict().values(), pl.state_dict().values()))
    o["all_trainable_fp32"] = all(p_.requires_grad and p_.dtype == torch.float32 for p_ in mf.parameters())
    o["n_params"] = sum(p_.numel() for p_ in mf.parameters())
    o["pass"] = o["keys_equal"] and o["bf16_equal"] == o["n_tensors"] > 0 and o["prj_equal"] and o["all_trainable_fp32"]; return o


def gradflow():
    import argparse
    import tiktoken
    import torch
    PS = tree_mod(); OW = PS.OW; enc = tiktoken.get_encoding("gpt2"); PS.VALUE_TOKS.update(PS.value_tokens(enc))
    m, prj = PS.build_full("cpu", 1); m, prj = m.double().eval(), prj.double().eval()
    st = OW.Stream(22939, order_mode="topo"); its = [PS.tok_item(st.draw(2), enc) for _ in range(4)]; b = PS.collate(its, 0, "cpu")
    a = argparse.Namespace(direct=True, recode=False, lam=0.0, bottleneck=False, codi_compat=False, m2_stage=False, k_cap=10)
    loss, parts, ex = PS.forward(m, prj, None, b, a); ex["ce"].backward()
    g = {n: (float(p_.grad.norm()) if p_.grad is not None else None) for n, p_ in list(m.named_parameters()) + [("prj." + n, p_) for n, p_ in prj.named_parameters()]}
    ok = lambda v: v is not None and v > 0 and math.isfinite(v)  # noqa: E731
    gpt = {k: v for k, v in g.items() if not k.startswith("prj.")}
    o = {"direct": {"n_gpt2": len(gpt), "nonzero_gpt2": sum(ok(v) for v in gpt.values()), "zero_or_none": [k for k, v in g.items() if not ok(v)][:20],
                    "note": "prj is unused in DIRECT (no latent steps): no grad by construction"}, "parts": parts}
    m.zero_grad(set_to_none=True); prj.zero_grad(set_to_none=True)               # M1 under --full_ft: latents through prj + the D head
    D = PS.DHead(m).double().eval(); b2 = PS.collate(its, 2, "cpu"); a2 = argparse.Namespace(**{**vars(a), "direct": False, "recode": True, "lam": 1.0})
    loss2, _, ex2 = PS.forward(m, prj, D, b2, a2); loss2.backward()
    g2 = {n: (float(p_.grad.norm()) if p_.grad is not None else None) for n, p_ in list(m.named_parameters()) + [("prj." + n, p_) for n, p_ in prj.named_parameters()]}
    o["m1"] = {"n": len(g2), "nonzero": sum(ok(v) for v in g2.values()), "zero_or_none": [k for k, v in g2.items() if not ok(v)][:20]}
    o["pass"] = o["direct"]["nonzero_gpt2"] == o["direct"]["n_gpt2"] == 148 and o["m1"]["nonzero"] == o["m1"]["n"]; return o


def flags_off():
    r = subprocess.run([sys.executable, "-B", "D:/Latent/scratch/gate_s226.py", "flags_off", "--tree", TREE, "--device", DEV], capture_output=True, text=True, cwd="D:/Latent", env=ENV)
    o = json.load(open("D:/Latent/scratch/gate_s226/flags_off.json")); o["via"] = "scratch/gate_s226.py flags_off --tree %s (incl. OW_DIRECT)" % TREE; o["stderr"] = r.stderr[-300:]
    return o


def run(extra, out):
    os.makedirs(out, exist_ok=True)
    for f_ in os.listdir(out):
        os.remove(os.path.join(out, f_))
    r = subprocess.run([sys.executable, "-B", os.path.join(TREE, "boxes", "codi_ps.py"), "--out", out, "--device", "cuda:0"] + P0D + extra, capture_output=True, text=True, cwd="D:/Latent", env=ENV)
    return r.returncode, r.stderr[-800:]


def mem():
    out = TMP + "/mem"; rc, err = run(["--mem_check", "--auto_micro", "--mem_budget", "7.0"], out)
    mc = json.load(open(out + "/mem_check.json")) if os.path.exists(out + "/mem_check.json") else {}
    chk = [c for c in mc.get("checks", []) if c.get("micro") == mc.get("micro")]
    o = {"rc": rc, "mem_check": mc, "err": err if rc else ""}; o["pass"] = rc == 0 and bool(chk) and chk[-1]["peak_gib"] <= 7.0; return o


def smoke():
    out = TMP + "/smoke"; rc, err = run(["--auto_micro", "--mem_budget", "7.0", "--log_every", "1", "--stop_at", "12"], out)
    lg = [json.loads(l) for l in open(out + "/log.jsonl")] if os.path.exists(out + "/log.jsonl") else []
    fin = all(math.isfinite(x["loss"]) and math.isfinite(x["gn_preclip"]) for x in lg)
    a = json.load(open(out + "/args.json")) if os.path.exists(out + "/args.json") else {}
    o = {"rc": rc, "n": len(lg), "finite": fin, "wrap": all("n_wrap" in x for x in lg) and len(lg) > 0, "micro": a.get("micro"), "full_ft": a.get("full_ft"),
         "sec_per_step": (lg[-1]["sec"] / lg[-1]["step"]) if lg else None, "err": err if rc not in (0, 9) else ""}
    o["pass"] = rc == 9 and o["n"] == 12 and fin and o["wrap"] and o["full_ft"] is True; return o


def args():
    out = TMP + "/args"; rc, err = run(["--args_only", "--stop_at", "12000", "--auto_micro", "--mem_budget", "7.0"], out); assert rc == 0, err
    a = json.load(open(out + "/args.json")); ref = json.load(open("D:/Latent/scratch/_tmp/s226_p0/P0b-TOPO/args.json"))
    meta = {"device", "train_gpu", "train_driver", "torch_version", "transformers_version", "python_version", "cuda_version", "micro", "accumulation", "out",
            "resume_from", "args_only", "args_ref", "resume", "stop_at", "auto_micro", "mem_check", "time_steps", "mem_budget"}
    d = {k for k in set(a) | set(ref) if k not in meta and a.get(k) != ref.get(k)}; nh = sorted(k for k in d if not k.endswith(("_sha", "_sha256")))
    o = {"diff": sorted(d), "nonhash": nh, "values": {k: [ref.get(k), a.get(k)] for k in nh}}; o["pass"] = nh == ["full_ft", "lr"]; return o


def main():
    os.makedirs(OUT, exist_ok=True)
    for part in [x for x in sys.argv[1:] if x in ("build", "gradflow", "flags_off", "mem", "smoke", "args")]:
        r = {"build": build, "gradflow": gradflow, "flags_off": flags_off, "mem": mem, "smoke": smoke, "args": args}[part](); r["tree"] = TREE
        json.dump(r, open(os.path.join(OUT, part + ".json"), "w", newline="\n"), indent=1, default=str); print(part, "pass", r["pass"], flush=True)


if __name__ == "__main__":
    main()
