"""THEORY §76 (A) driver: the free-table latent automaton (exact marginal, final answers only) on S4 / A5 / S5,
curricula ge3 (primary: lengths >= 3 only) and std, seeds 2051-2060; M = 2|G|, tau 4 -> 1, lr 1e-2, 16,000 updates; hard
tables saved for (B). Descriptive: A5 ge3 M sweep (60, 90, 240) and tau0 = 1, seeds 2051-2055. P concurrent single-thread processes, one seed per process, resumable. Output
theory/results/c1_tab/<group>_<curriculum>_s<seed>.json, marker DONE.
    python theory/c1_tab_driver.py --P 28 [--groups S4 A5 S5] [--curricula ge3 std] [--device cuda:0] [--no_extras]
"""
import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
GSIZE = {"S4": 24, "A5": 60, "S5": 120}
SEEDS = list(range(2051, 2061))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--P", type=int, default=28); ap.add_argument("--groups", nargs="*", default=["S5", "A5", "S4"])
    ap.add_argument("--out", default=os.path.join(HERE, "results", "c1_tab"))
    ap.add_argument("--device", default="cpu"); ap.add_argument("--curricula", nargs="*", default=["ge3", "std"])
    ap.add_argument("--no_extras", action="store_true", help="skip the A5 M-sweep / tau0 = 1 descriptive arms")
    a = ap.parse_args(); os.makedirs(os.path.join(a.out, "logs"), exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")

    def one(job):
        g, cur, s, M, tau0, tag = job
        name = "%s_%s%s_s%d" % (g, cur, tag, s)
        f = os.path.join(a.out, name + ".json")
        if not os.path.exists(f):
            with open(os.path.join(a.out, "logs", name + ".log"), "w") as lg:
                subprocess.run([sys.executable, "-u", os.path.join(HERE, "c1_rl.py"), "--group", g, "--cell", "tab", "--M",
                                str(M), "--method", "marg", "--tau0", str(tau0), "--updates", "16000", "--curriculum", cur,
                                "--seeds", str(s), "--threads", "1", "--save_table", "--device", a.device, "--out", f],
                               stdout=lg, stderr=subprocess.STDOUT, env=env)
        return os.path.exists(f)

    jobs = [(g, cur, s, 2 * GSIZE[g], 4, "") for g in a.groups for cur in a.curricula for s in SEEDS]   # largest first
    # descriptive (Literature 99ec154): M sweep and tau ablation, A5 ge3, seeds 2051-2055
    if not a.no_extras and "A5" in a.groups:
        jobs += [("A5", "ge3", s, M, 4, "_M%d" % M) for M in (240, 90, 60) for s in SEEDS[:5]]
        jobs += [("A5", "ge3", s, 120, 1, "_tau1") for s in SEEDS[:5]]
    t0 = time.time()
    with ThreadPoolExecutor(a.P) as ex:
        ok = list(ex.map(one, jobs))
    print("%d jobs, %d missing, %.0f s" % (len(jobs), ok.count(False), time.time() - t0), flush=True)
    open(os.path.join(a.out, "DONE"), "w").write(time.strftime("%Y-%m-%d %H:%M:%S %Z") + "\n")


if __name__ == "__main__":
    main()
