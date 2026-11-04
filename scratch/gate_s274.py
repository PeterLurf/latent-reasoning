"""§271 N1 host gate G3 GRADFLOW (N1_v1 §10; CUDA bf16 autocast as training) -> <out>/gradflow_*.json and <out>/gates_pass.json only if all pass. Queue line (the runner
appends --device <d> --out <dir>):
  <dir> scratch.gate_s274 --arms loop[,direct_of,direct_si] -- <the LOOP-SI flags without inits> ;; <the DIRECT flags without inits>
  loop       a gate-only copy of boxes/inject_so.py with gate_s251.PROBE after 'loss = la + lam_now[0] * aux' (per-term grads), stop 8: every probed call with d >= 2 has
             nonzero answer-loss grads on DeltaMem qkv / gate / ln_w / wo; the aux grad norm is finite and > 0.
  direct_of  a gate-only copy with a probe after the DIRECT 'loss = la' line: the answer-loss grad on the trunk is finite and > 0 (stop 4).
  direct_si  as direct_of with --so_direct_si 1: the SI aux term's grad on the trunk is finite and > 0 (stop 4).
Inits are not needed (gradient flow is a property of the graph); the cells' own flags otherwise. Exit 0 iff all requested arms pass."""
import json
import math
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, "scratch"))
os.environ.setdefault("LATENT_ROOT", ROOT)   # 11-04: gate_s251 chdirs to LATENT_ROOT (default D:/Latent) at import; b425's host import check died on it
from gate_s251 import PROBE  # noqa: E402

DPROBE = r'''
            if torch.is_grad_enabled() and loss.requires_grad and os.environ.get("GRADPROBE_OUT"):   # §271 GATE-ONLY: direct-arm grads on the trunk
                _pl = [p_ for p_ in g.parameters() if p_.requires_grad]
                _g1 = torch.autograd.grad(la, _pl, retain_graph=True, allow_unused=True)
                _x = (loss - la) if a.so_direct_si else None
                _g2 = torch.autograd.grad(_x, _pl, retain_graph=True, allow_unused=True) if (_x is not None and _x.requires_grad) else None
                _n = lambda gs: math.sqrt(sum(float((x_.float() ** 2).sum()) for x_ in gs if x_ is not None)) if gs is not None else None  # noqa: E731
                open(os.environ["GRADPROBE_OUT"], "a").write(json.dumps({"d": items[0]["d"], "la_trunk": _n(_g1), "si_trunk": _n(_g2)}) + "\n")
'''


def copy(dst, anchor, probe):
    src = open(os.path.join(ROOT, "boxes", "inject_so.py"), encoding="utf-8").read(); assert src.count(anchor) == 1, anchor
    open(os.path.join(ROOT, "boxes", dst), "w", encoding="utf-8", newline="\n").write(src.replace(anchor, anchor + probe.lstrip("\n")))


def run(mod, fl, d, dev, stop):
    shutil.rmtree(d, ignore_errors=True); os.makedirs(d); lp = os.path.join(d, "gradprobe.jsonl"); env = {**os.environ, "GRADPROBE_OUT": lp}
    with open(os.path.join(d, "stdout.log"), "w") as f_:
        rc = subprocess.run([sys.executable, "-B", "-m", mod] + fl + ["--ckpt_every", "0", "--log_every", "1", "--stop_at", str(stop), "--device", dev, "--out", d],
                            cwd=ROOT, env=env, stdout=f_, stderr=subprocess.STDOUT).returncode
    for f in os.listdir(d):
        if f.endswith(".pt"):
            os.remove(os.path.join(d, f))
    return rc, [json.loads(l_) for l_ in open(lp)] if os.path.exists(lp) else []


def main():
    argv = sys.argv[1:]; dev = argv[argv.index("--device") + 1]; out = os.path.join(ROOT, argv[argv.index("--out") + 1])
    for f in ("--device", "--out"):
        k = argv.index(f); del argv[k:k + 2]
    i = argv.index("--"); arms = argv[argv.index("--arms") + 1].split(","); rest = argv[i + 1:]; j = rest.index(";;"); fl_loop, fl_dir = rest[:j], rest[j + 1:]
    os.makedirs(out, exist_ok=True); res = {}
    if "loop" in arms:
        copy("_inject_so_gradprobe_s274.py", "            loss = la + lam_now[0] * aux\n", PROBE)
        rc, rs = run("boxes._inject_so_gradprobe_s274", fl_loop, os.path.join(out, "loop"), dev, 8); el = [r for r in rs if r["d"] >= 2]
        req = ("qkv.weight", "gate.weight", "ln_w.weight", "wo.weight")
        res["loop"] = {"rc": rc, "n_calls": len(rs), "n_eligible": len(el), "all_nonzero": bool(el) and all(all((r["la_grad_mem"].get(k) or 0) > 0 for k in req) for r in el),
                       "aux_ok": bool(rs) and all(math.isfinite(r["aux_grad_norm"]) and r["aux_grad_norm"] > 0 for r in rs)}
        res["loop"]["pass"] = rc == 9 and res["loop"]["all_nonzero"] and res["loop"]["aux_ok"]
    for arm in [x for x in arms if x.startswith("direct")]:
        copy("_inject_so_gradprobe_d_s274.py", "                loss = la + lam_now[0] * si_aux(items, stat, la * 0.0)\n", DPROBE)   # after the (optional) SI line, at the direct body level
        fl = fl_dir + (["--so_direct_si", "1", "--aux_step", "1", "--aux_reweight"] if arm == "direct_si" and "--so_direct_si" not in fl_dir else [])
        rc, rs = run("boxes._inject_so_gradprobe_d_s274", fl, os.path.join(out, arm), dev, 4)
        ok = bool(rs) and all(r["la_trunk"] is not None and math.isfinite(r["la_trunk"]) and r["la_trunk"] > 0 for r in rs)
        if arm == "direct_si":
            ok = ok and all(r["si_trunk"] is not None and math.isfinite(r["si_trunk"]) and r["si_trunk"] > 0 for r in rs if r["d"] >= 1)
        res[arm] = {"rc": rc, "n_calls": len(rs), "pass": rc == 9 and ok}
    for k, v in res.items():
        json.dump(v, open(os.path.join(out, "gradflow_%s.json" % k), "w"), indent=1)
    ok = all(v["pass"] for v in res.values()) and set(res) == set(arms)
    if ok:
        json.dump({"pass": True, "arms": res}, open(os.path.join(out, "gates_pass.json"), "w"), indent=1)
    print("GATE_S274", json.dumps(res), flush=True); sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
