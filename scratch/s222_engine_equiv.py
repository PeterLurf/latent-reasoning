"""§222 Audit G-D condition 1 (10-21): ENGINE EQUIVALENCE. The cert group loop Audit G-D'd at e6ed5638 (old) vs the Note 222-1 Engine (new, the working
tree) on the same short run: untrained model, smoke sizes (pool 3000, held 64, per_D 3), --steps 150 --eval_every 30 (5 curve points, so the extension
rule is live), group 7 (both groups, all 13 arms), same card / fp32. Each impl runs in its own process with Arm.__init__ wrapped to keep every arm; at the
end every arm's trainable state (P, beta, gam) and steps go to <out>/params.pt. PASS iff per-arm curves (and probe0's test read) are identical and every
final parameter tensor is bitwise equal.
    python -B scratch/s222_engine_equiv.py run old|new --device cuda:2      (old = git show e6ed5638:scratch/s222_cert.py)
    python -B scratch/s222_engine_equiv.py compare                          -> scratch/gate_s222/engine_equiv.json
Note 222-3 / Audit (a') 10-21: --det runs the gate under forced math SDP + torch.use_deterministic_algorithms(True) (scratch/s222_det_wrap.apply; GATE ONLY,
the registered runs keep the memory-efficient kernel) into <impl>_det dirs; compare --det -> engine_equiv_det.json (bitwise required).
compare_pair <dirA> <dirB> <out name>: descriptive same-code envelope (e.g. new vs new2 on the default kernel)."""
import importlib.util
import json
import os
import subprocess
import sys

ROOT = "D:/Latent"; os.chdir(ROOT); T = "scratch/_tmp/s222_equiv"
COMMON = ["--mode", "cert", "--model", "untrained", "--pool_n", "3000", "--hold_n", "64", "--per_D", "3", "--steps", "150", "--eval_every", "30", "--group", "7"]


def run(impl, dev, det=False):
    if det:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8"); sys.path.insert(0, "D:/Latent/scratch"); import s222_det_wrap; s222_det_wrap.apply()
    import torch
    od = os.path.join(T, impl + ("_det" if det else "")); os.makedirs(od, exist_ok=True)
    if impl.startswith("old"):
        src = os.path.join(T, "s222_cert_e6ed5638.py"); open(src, "w", encoding="utf-8", newline="\n").write(subprocess.run(["git", "show", "e6ed5638:scratch/s222_cert.py"], capture_output=True, text=True, check=True).stdout)
        json.dump(json.load(open("scratch/gate_s222/m_rule.json")), open(os.path.join(od, "m_rule.json"), "w"))   # the old loop reads <out>/m_rule.json
        argv = COMMON + ["--device", dev, "--out", od]
    else:
        src = "scratch/s222_cert.py"; argv = COMMON + ["--device", dev, "--out", od, "--m_rule", "scratch/gate_s222/m_rule.json", "--groups", "all"]
    spec = importlib.util.spec_from_file_location("cert_" + impl, src); M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
    REG = []; init0 = M.Arm.__init__

    def init(self, *a, **k):
        init0(self, *a, **k); REG.append(self)
    M.Arm.__init__ = init; sys.argv = ["s222_cert"] + argv; M.main()
    torch.save({x.key: {"P": {k: v.detach().cpu() for k, v in x.pr.P.items()}, "beta": x.pr.beta.detach().cpu(), "gam": x.pr.gam.detach().cpu(), "steps": x.pr.steps}
                for x in REG}, os.path.join(od, "params.pt"))
    print("EQUIV_RUN", impl, "det", det, "arms", len(REG), "src_sha", M.sha_file(src)[:12], flush=True)


def compare(da="old", db="new", name="engine_equiv"):
    import torch
    a, b = json.load(open(os.path.join(T, da, "cert_untrained.json"))), json.load(open(os.path.join(T, db, "cert_untrained.json")))
    pa, pb = torch.load(os.path.join(T, da, "params.pt")), torch.load(os.path.join(T, db, "params.pt"))
    res = {"arms_old": sorted(a["arms"]), "arms_new": sorted(b["arms"]), "curve_diff": [], "field_diff": [], "param_diff": [], "steps": {}}
    for k in sorted(set(a["arms"]) | set(b["arms"])):
        x, y = a["arms"].get(k), b["arms"].get(k)
        if x is None or y is None:
            res["curve_diff"].append(k + " missing"); continue
        if [list(v) for v in x["curve"]] != [list(v) for v in y["curve"]]:
            res["curve_diff"].append(k)
        for f in ("LATE", "steps_run", "extended", "test_acc", "test_acc_by_D", "items"):
            if x.get(f) != y.get(f):
                res["field_diff"].append("%s.%s" % (k, f))
        res["steps"][k] = [x["steps_run"], y["steps_run"]]
    for k in sorted(set(pa) | set(pb)):
        if k not in pa or k not in pb:
            res["param_diff"].append(k + " missing"); continue
        ta = {**{"P." + n: v for n, v in pa[k]["P"].items()}, "beta": pa[k]["beta"], "gam": pa[k]["gam"]}; tb = {**{"P." + n: v for n, v in pb[k]["P"].items()}, "beta": pb[k]["beta"], "gam": pb[k]["gam"]}
        for n in sorted(set(ta) | set(tb)):
            if n not in ta or n not in tb or not torch.equal(ta[n], tb[n]):
                res["param_diff"].append("%s.%s" % (k, n))
    res["n_param_tensors_per_arm"] = len(pa[next(iter(pa))]["P"]) + 2 if pa else 0
    res["extended_arms"] = sorted(k for k, v in b["arms"].items() if v["extended"])
    res["PASS"] = not (res["curve_diff"] or res["field_diff"] or res["param_diff"]) and res["arms_old"] == res["arms_new"] and len(res["arms_new"]) == 13
    res["card"] = b["meta"]["card"]; res["def"] = __doc__; res["old_src"] = "git show e6ed5638:scratch/s222_cert.py"
    res["new_sha256"] = b["meta"]["script_sha256"]; res["dirs"] = [da, db]
    md = {}
    for k in sorted(set(pa) & set(pb)):
        d_ = max(float((pa[k]["P"][n] - pb[k]["P"][n]).abs().max()) for n in pa[k]["P"]); md[k] = d_
    res["max_abs_param_diff_by_arm"] = md
    os.makedirs("scratch/gate_s222", exist_ok=True); json.dump(res, open("scratch/gate_s222/%s.json" % name, "w", newline="\n"), indent=1)
    print("ENGINE-EQUIV", "PASS" if res["PASS"] else "FAIL", {k: res[k] for k in ("curve_diff", "field_diff", "param_diff", "extended_arms")}, flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2], sys.argv[sys.argv.index("--device") + 1] if "--device" in sys.argv else "cuda:2", det="--det" in sys.argv)
    elif sys.argv[1] == "compare_pair":
        compare(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        compare(*(("old_det", "new_det", "engine_equiv_det") if "--det" in sys.argv else ()))
