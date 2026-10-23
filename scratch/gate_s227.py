"""§227 TEXT-SERIAL build gates (registered f69fd658; Theory / Audit 10-23). Code under test: TREE/boxes/codi_ps.py (scratch/_tmp/s227_tree) until
committed. -> scratch/gate_s227/<part>.json ("pass").
  known    the decode known-answer smoke (CPU, no model): 200 OW-topo items per d 1..8 (train frames, seed 22950): text_target's piecewise encode == the
           whole-string encode; dec[vidx] == " v_t" and dec[nidx] == " v_d"; parsing the gold target text (ANS_RE / STEP_RE) gives the gold answer and
           every gold step value - 100% of items.
  gradflow --text_cot --full_ft DIRECT on 4 d 3 items, fp64 CPU, eval mode: the target CE gives a nonzero finite grad on every GPT-2 tensor; step_value_ce
           finite.
  flags_off scratch/gate_s226.py flags_off with TREE (6 existing cells, det wrapper, DEV) -> written to scratch/gate_s227 (S226_GATE_OUT).
  smoke    the §227 cell's flags on the 4060 (registered kernel), --auto_micro --mem_budget 7.0, 12 steps, --decode_every 6: rc 9, 12 log rows with
           step_value_ce / step_value_acc, 2 decode_evals.jsonl records with by_d for d 1..min(d_max + 1, 8), args cg_signal step_value_ce + text_eval_sha;
           the auto_micro probe (deepest d 8 text shape, >= d 3) peak <= 7.0.
  args     --args_only the §227 cell vs P0d's args.json (scratch/_tmp/s226_p0d/P0d-FULLFT): the non-hash, non-META diff.
    python scratch/gate_s227.py known gradflow args [flags_off --device cuda:2] [smoke]"""
import importlib.util
import json
import math
import os
import subprocess
import sys

TREE = "D:/Latent/scratch/_tmp/s227_tree"; OUT = "D:/Latent/scratch/gate_s227"; TMP = "D:/Latent/scratch/_tmp/s227_gate"; DEV = "cuda:2"
for i, v in enumerate(sys.argv):
    if v == "--tree":
        TREE = sys.argv[i + 1]
    if v == "--device":
        DEV = sys.argv[i + 1]
CELL = ["--task", "ow", "--order_mode", "topo", "--direct", "--full_ft", "--lr", "1e-4", "--text_cot", "--cg_thr", "0.45", "--cg_cap", "6000", "--steps", "48000"]
ENV = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}


def tree_mod():
    sys.path.insert(0, TREE); sp = importlib.util.spec_from_file_location("codi_ps_t", os.path.join(TREE, "boxes", "codi_ps.py"))
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m


def known():
    import tiktoken
    PS = tree_mod(); OW = PS.OW; enc = tiktoken.get_encoding("gpt2"); o = {"by_d": {}}; allok = True
    for d in range(1, 9):
        st = OW.Stream(22950, frames=PS.NW.FRAME_TRAIN, order_mode="topo"); bad = {"encode": 0, "vidx": 0, "nidx": 0, "ans_parse": 0, "step_parse": 0}
        for _ in range(200):
            it = st.draw(d); dec, vidx, nidx = PS.text_target(it, enc)
            whole = [PS.EOT] + enc.encode(" " + " ".join(it["R_step"]) + " The answer is %d." % it["answer"]) + [PS.EOS]
            bad["encode"] += dec != whole
            bad["vidx"] += [enc.decode([dec[i]]) for i in vidx] != [" %d" % v for v in it["values"][1:]]
            bad["nidx"] += enc.decode([dec[nidx]]) != " %d" % it["answer"]
            txt = enc.decode(dec[1:-1]); ma = PS.ANS_RE.search(txt)
            bad["ans_parse"] += not (ma and int(ma.group(1)) == it["answer"])
            bad["step_parse"] += [int(v) for _, v in PS.STEP_RE.findall(txt[:ma.start()] if ma else txt)] != it["values"][1:]
        o["by_d"][str(d)] = bad; allok = allok and not any(bad.values())
    o["pass"] = allok; return o


def gradflow():
    import argparse
    import tiktoken
    PS = tree_mod(); OW = PS.OW; enc = tiktoken.get_encoding("gpt2"); PS.VALUE_TOKS.update(PS.value_tokens(enc))
    m, prj = PS.build_full("cpu", 1); m, prj = m.double().eval(), prj.double().eval()
    st = OW.Stream(22959, order_mode="topo"); its = [PS.tok_item(st.draw(3), enc, True) for _ in range(4)]; b = PS.collate(its, 0, "cpu")
    a = argparse.Namespace(direct=True, recode=False, lam=0.0, bottleneck=False, codi_compat=False, m2_stage=False, k_cap=10)
    loss, parts, ex = PS.forward(m, prj, None, b, a); ex["ce"].backward()
    g = {n: (float(p_.grad.norm()) if p_.grad is not None else None) for n, p_ in m.named_parameters()}
    ok = lambda v: v is not None and v > 0 and math.isfinite(v)  # noqa: E731
    o = {"n_gpt2": len(g), "nonzero": sum(ok(v) for v in g.values()), "zero_or_none": [k for k, v in g.items() if not ok(v)][:10],
         "step_value_ce": parts.get("step_value_ce"), "step_value_acc": parts.get("step_value_acc"), "num_ce": parts.get("num_ce")}
    o["pass"] = o["nonzero"] == o["n_gpt2"] == 148 and o["step_value_ce"] is not None and math.isfinite(o["step_value_ce"]); return o


def flags_off():
    r = subprocess.run([sys.executable, "-B", "D:/Latent/scratch/gate_s226.py", "flags_off", "--tree", TREE, "--device", DEV], capture_output=True, text=True, cwd="D:/Latent",
                       env={**ENV, "S226_GATE_OUT": OUT})
    o = json.load(open(os.path.join(OUT, "flags_off.json"))); o["via"] = "scratch/gate_s226.py flags_off --tree %s" % TREE; o["stderr"] = r.stderr[-300:]; return o


def run(extra, out):
    os.makedirs(out, exist_ok=True)
    for f_ in os.listdir(out):
        os.remove(os.path.join(out, f_))
    r = subprocess.run([sys.executable, "-B", os.path.join(TREE, "boxes", "codi_ps.py"), "--out", out, "--device", "cuda:0"] + CELL + extra, capture_output=True, text=True, cwd="D:/Latent", env=ENV)
    return r.returncode, r.stderr[-800:]


def smoke():
    out = TMP + "/smoke"; rc, err = run(["--auto_micro", "--mem_budget", "7.0", "--log_every", "1", "--stop_at", "12", "--decode_every", "6"], out)
    lg = [json.loads(l) for l in open(out + "/log.jsonl")] if os.path.exists(out + "/log.jsonl") else []
    de = [json.loads(l) for l in open(out + "/decode_evals.jsonl")] if os.path.exists(out + "/decode_evals.jsonl") else []
    a = json.load(open(out + "/args.json")) if os.path.exists(out + "/args.json") else {}; mc = json.load(open(out + "/mem_check.json")) if os.path.exists(out + "/mem_check.json") else {}
    chk = [c for c in mc.get("checks", []) if c.get("micro") == mc.get("micro")]
    o = {"rc": rc, "n_log": len(lg), "svce_logged": all("step_value_ce" in x and "step_value_acc" in x for x in lg) and len(lg) > 0,
         "decode_records": [[x["step"], x["d_max"], sorted(x["by_d"]), sorted(next(iter(x["by_d"].values())).keys()) if x["by_d"] else None] for x in de],
         "cg_signal": a.get("cg_signal"), "text_eval_sha": a.get("text_eval_sha"), "micro": a.get("micro"), "mem_checks": mc.get("checks"),
         "sec_per_step": (lg[-1]["sec"] / lg[-1]["step"]) if lg else None, "err": err if rc not in (0, 9) else ""}
    o["pass"] = (rc == 9 and o["n_log"] == 12 and o["svce_logged"] and [x[0] for x in o["decode_records"]] == [6, 12] and all(x[2] == [str(k) for k in range(1, x[1] + 2)] for x in o["decode_records"])
                 and o["cg_signal"] == "step_value_ce" and bool(o["text_eval_sha"]) and bool(chk) and chk[-1]["peak_gib"] <= 7.0)
    return o


def args():
    out = TMP + "/args"; rc, err = run(["--args_only", "--stop_at", "18000", "--auto_micro", "--mem_budget", "7.0"], out); assert rc == 0, err
    a = json.load(open(out + "/args.json")); ref = json.load(open("D:/Latent/scratch/_tmp/s226_p0d/P0d-FULLFT/args.json"))
    meta = {"device", "train_gpu", "train_driver", "torch_version", "transformers_version", "python_version", "cuda_version", "micro", "accumulation", "out",
            "resume_from", "args_only", "args_ref", "resume", "stop_at", "auto_micro", "mem_check", "time_steps", "mem_budget"}
    d = {k for k in set(a) | set(ref) if k not in meta and a.get(k) != ref.get(k)}; nh = sorted(k for k in d if not k.endswith(("_sha", "_sha256")))
    o = {"diff": sorted(d), "nonhash": nh, "values": {k: [ref.get(k), a.get(k)] for k in nh}}
    o["pass"] = set(nh) <= {"text_cot", "decode_every", "cg_signal", "text_eval_n_per_d", "excl_n"} and "text_cot" in nh and a.get("decode_every") == 1000; return o


def main():
    os.makedirs(OUT, exist_ok=True)
    for part in [x for x in sys.argv[1:] if x in ("known", "gradflow", "flags_off", "smoke", "args")]:
        r = {"known": known, "gradflow": gradflow, "flags_off": flags_off, "smoke": smoke, "args": args}[part](); r["tree"] = TREE
        json.dump(r, open(os.path.join(OUT, part + ".json"), "w", newline="\n"), indent=1, default=str); print(part, "pass", r["pass"], flush=True)


if __name__ == "__main__":
    main()
