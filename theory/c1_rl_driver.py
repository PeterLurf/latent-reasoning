"""THEORY §69 driver (amendment 69-1): calibration -> pilot -> scored S3 runs, then (separately) Stage 2 on S4/A5.
Runs theory/c1_rl.py as P concurrent single-thread processes, one seed per process; resumable (skips existing outputs).
  calib : seed 990, tau0 1, 8,000 updates, every method but bon x {tab, mlp}; U_m = round(8000 proc(marg)/proc(m)) if
          proc(m) > 1.5 proc(marg) else 8000 (per cell) -> calib.json
  pilot : every method x cell x tau0 {1, 4} x seeds 990-994 at U_m -> choice.json (tau0 per (method, cell): certified
          count, then mean hard accuracy at N 32, then tau0 = 1)
  scored: every method x cell at its chosen tau0 x seeds 200-209 at U_m; plus hem/grpo/rmer/hyb at the full 8,000 (desc.)
  s2    : --methods (Stage-1 passers; marg is always added) on S4 then A5, bilinear cell, M = 2|G|, tau0 4, 16,000 updates
          rescaled by a seed-990 S4 calibration; seeds 200-202 Zhang (registered) and std (descriptive)
    python theory/c1_rl_driver.py --phase all --P 16 [--out theory/results/c1_rl]
    python theory/c1_rl_driver.py --phase s2 --methods hem rmer --P 16
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
METHODS = ["marg", "hem", "grpo", "rmer", "hyb", "bon"]
CELLS = ["tab", "mlp"]
GSIZE = {"S3": 6, "S4": 24, "A5": 60}


def run_all(jobs, out, P):
    """jobs: list of (name, args, weight); heaviest first. Returns {name: result dict for its single seed}."""
    os.makedirs(os.path.join(out, "logs"), exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")

    def one(j):
        name, args, _ = j
        f = os.path.join(out, name + ".json")
        if not os.path.exists(f):
            with open(os.path.join(out, "logs", name + ".log"), "w") as lg:
                subprocess.run([sys.executable, "-u", os.path.join(HERE, "c1_rl.py")] + args + ["--threads", "1", "--out", f],
                               stdout=lg, stderr=subprocess.STDOUT, env=env)
        try:
            return name, next(iter(json.load(open(f)).values()))
        except Exception:
            return name, None

    jobs = sorted(jobs, key=lambda j: -j[2])
    t0 = time.time()
    with ThreadPoolExecutor(P) as ex:
        res = dict(ex.map(one, jobs))
    print("%d jobs, %d missing, %.0f s" % (len(jobs), sum(v is None for v in res.values()), time.time() - t0), flush=True)
    return res


def args_for(group, cell, method, seed, updates, tau0, curriculum="std", M=0):
    a = ["--group", group, "--cell", cell, "--method", method, "--seeds", str(seed), "--updates", str(updates),
         "--tau0", str(tau0), "--curriculum", curriculum]
    return a + (["--M", str(M)] if M else [])


def calibrate(res, cells, methods, key):
    U = {}
    for c in cells:
        pm = res[key(c, "marg")]["proc_sec"]; U[c] = {}
        for m in methods:
            if m == "bon":
                U[c][m] = 8000; continue
            p = res[key(c, m)]["proc_sec"]
            U[c][m] = int(round(8000 * pm / p)) if p > 1.5 * pm else 8000
        U[c]["_proc"] = {m: res[key(c, m)]["proc_sec"] for m in methods if m != "bon"}
    return U


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all", choices=["all", "calib", "pilot", "scored", "s2"])
    ap.add_argument("--P", type=int, default=16); ap.add_argument("--out", default=os.path.join(HERE, "results", "c1_rl"))
    ap.add_argument("--methods", nargs="*", default=[])
    a = ap.parse_args(); out = a.out; os.makedirs(out, exist_ok=True)
    cf, chf = os.path.join(out, "calib.json"), os.path.join(out, "choice.json")

    if a.phase in ("all", "calib") and not os.path.exists(cf):
        jobs = [("calib_%s_%s" % (c, m), args_for("S3", c, m, 990, 8000, 1.0), 1) for c in CELLS for m in METHODS if m != "bon"]
        res = run_all(jobs, out, a.P)
        U = calibrate(res, CELLS, METHODS, lambda c, m: "calib_%s_%s" % (c, m))
        json.dump(U, open(cf, "w", newline="\n"), indent=1); print("CALIB", U, flush=True)

    if a.phase in ("all", "pilot") and not os.path.exists(chf):
        U = json.load(open(cf))
        jobs = [("pilot_%s_%s_t%d_s%d" % (c, m, t, s), args_for("S3", c, m, s, U[c][m], float(t)), 4 if m == "bon" else 1)
                for c in CELLS for m in METHODS for t in (1, 4) for s in range(990, 995)]
        res = run_all(jobs, out, a.P)
        ch = {}
        for c in CELLS:
            ch[c] = {}
            for m in METHODS:
                sc = {}
                for t in (1, 4):
                    rs = [res["pilot_%s_%s_t%d_s%d" % (c, m, t, s)] for s in range(990, 995)]
                    sc[t] = (sum(bool(r and r["certified"]) for r in rs), sum(r["acc"]["32"] for r in rs if r) / 5, -t)
                ch[c][m] = {"tau0": max((1, 4), key=lambda t: sc[t]), "pilot": {str(t): sc[t][:2] for t in (1, 4)}}
        json.dump(ch, open(chf, "w", newline="\n"), indent=1); print("CHOICE", ch, flush=True)

    if a.phase in ("all", "scored"):
        U, ch = json.load(open(cf)), json.load(open(chf))
        jobs = []
        for c in CELLS:
            for m in METHODS:
                t = float(ch[c][m]["tau0"])
                jobs += [("scored_%s_%s_s%d" % (c, m, s), args_for("S3", c, m, s, U[c][m], t), 4 if m == "bon" else 1)
                         for s in range(200, 210)]
                if m in ("hem", "grpo", "rmer", "hyb") and U[c][m] != 8000:
                    jobs += [("full_%s_%s_s%d" % (c, m, s), args_for("S3", c, m, s, 8000, t), 2) for s in range(200, 210)]
        run_all(jobs, out, a.P)
        open(os.path.join(out, "STAGE1_DONE"), "w").write(time.strftime("%Y-%m-%d %H:%M:%S %Z") + "\n")

    if a.phase == "s2":
        ms = ["marg"] + [m for m in a.methods if m != "marg"]
        s2c = os.path.join(out, "s2_calib.json")
        if not os.path.exists(s2c):
            jobs = [("s2calib_%s" % m, args_for("S4", "bil", m, 990, 16000, 4.0, "zhang", 48), 1) for m in ms if m != "bon"]
            res = run_all(jobs, out, a.P)
            U = calibrate(res, ["bil"], [m for m in ms if m != "bon"], lambda c, m: "s2calib_%s" % m)["bil"]
            U = {m: (v * 2 if isinstance(v, int) else v) for m, v in U.items()}       # calibrate() is per 8,000; Stage 2 is 16,000
            json.dump(U, open(s2c, "w", newline="\n"), indent=1); print("S2CALIB", U, flush=True)
        U = json.load(open(s2c))
        jobs = []
        for g in ("S4", "A5"):
            for m in ms:
                for cur in ("zhang", "std"):
                    jobs += [("s2_%s_%s_%s_s%d" % (g, cur, m, s), args_for(g, "bil", m, s, U.get(m, 16000), 4.0, cur, 2 * GSIZE[g]),
                              (4 if m == "bon" else 1) * GSIZE[g]) for s in range(200, 203)]
        run_all(jobs, out, a.P)
        open(os.path.join(out, "STAGE2_DONE"), "w").write(time.strftime("%Y-%m-%d %H:%M:%S %Z") + "\n")


if __name__ == "__main__":
    main()
