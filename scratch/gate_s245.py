"""§245 AO-REP build gate (Theory's build request; §244 AO_v1 recipe at 18k, seeds 3 / 4) -> scratch/gate_s245/args.json ("pass"). CPU only.
  module   boxes/inject_aw.py sha256 starts with 461458f5 (the §244 module: G-D b7cb8630; lam / gradflow / p2bit gates there; host replay bitwise fdef89bd).
  lam      lam_at (its own source, ast-extracted) under --aux_fade 0,0: 1.0 at step 0 and 0.0 at every step 1..18000 (training starts at step 1).
  args     --args_only ao_s3 / ao_s4 vs scratch/s244_args_frozen_ao_s1.json: diff == {steps, seed} exactly; writes scratch/s245_args_frozen_<run>.json.
    python -B scratch/gate_s245.py"""
import ast
import hashlib
import json
import os
import sys
import textwrap

ROOT = "D:/Latent"; sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/scratch"); os.chdir(ROOT)
import gate_s228 as G8  # noqa: E402
import gate_s244 as G4  # noqa: E402

OUT = ROOT + "/scratch/gate_s245"; TMP = ROOT + "/scratch/_tmp/s245"


def flags(seed):
    return [x if x != "12000" else "18000" for x in G4.ao(G4.P2)] + ["--seed", str(seed)]


def main():
    import s228_args_check as C
    os.makedirs(OUT, exist_ok=True); os.makedirs(TMP, exist_ok=True); o = {}
    sha = hashlib.sha256(open(ROOT + "/boxes/inject_aw.py", "rb").read()).hexdigest(); o["module_sha256"] = sha; o["module_ok"] = sha.startswith("461458f5")
    src = open(ROOT + "/boxes/inject_aw.py", encoding="utf-8").read()
    fn = [ast.get_source_segment(src, n) for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == "lam_at"][0]
    ns = {"a": type("A", (), {"aux_step": 1.0})(), "f0_": 0.0, "f1_": 0.0, "sc": 1.0, "max": max}; exec(textwrap.dedent(fn), ns)
    o["lam_0"] = ns["lam_at"](0); o["lam_ge1_all_zero"] = all(ns["lam_at"](s) == 0.0 for s in range(1, 18001))
    ref = json.load(open(ROOT + "/scratch/s244_args_frozen_ao_s1.json")); new = {}
    for run, seed in (("ao_s3", 3), ("ao_s4", 4)):
        d = G8.fresh(TMP + "/args_" + run)
        rc = G8.sh([sys.executable, "-B", "-m", "boxes.inject_aw"] + flags(seed) + ["--auto_micro", "--ckpt_every", "1000", "--eval_parts", G4.P2_EVAL, "--keep_final_ckpt",
                                                                                   "--device", "cpu", "--args_only", "--out", d], d + "/stdout.log")
        new[run] = json.load(open(d + "/args.json")); dif = C.diff(ref, new[run]); o[run] = {"rc": rc, "diff_vs_ao_s1": dif}
    o["pass"] = (o["module_ok"] and o["lam_0"] == 1.0 and o["lam_ge1_all_zero"] and all(o[r]["rc"] == 0 and set(o[r]["diff_vs_ao_s1"]) == {"steps", "seed"} for r in new))
    if o["pass"]:
        for r, a in new.items():
            json.dump(a, open(ROOT + "/scratch/s245_args_frozen_%s.json" % r, "w", newline="\n"), indent=1)
    json.dump(o, open(OUT + "/args.json", "w", newline="\n"), indent=1, default=str); print("args pass", o["pass"], flush=True)


if __name__ == "__main__":
    main()
