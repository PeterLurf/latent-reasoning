"""THEORY §70 driver: every arm on development seeds 1300-1302 and confirmation seeds 1400-1409, P concurrent single-thread
processes, one seed per process; resumable. Output theory/results/c1_acq/<arm>_s<seed>.json, marker DONE.
    python theory/c1_acq_driver.py --P 24
"""
import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = {"A_add": "--cell add --M 24 --method mml", "A_mul": "--cell mul --M 24 --method mml",
        "A_opb12": "--cell opb --M 24 --rank 12 --method mml", "A_opb24": "--cell opb --M 24 --rank 24 --method mml",
        "B_temper": "--cell mlp --M 12 --method temper", "B_eta1": "--cell mlp --M 12 --method eta1",
        "B_mml2": "--cell mlp --M 12 --method mml2", "B_mml": "--cell mlp --M 12 --method mml",
        "B_exr": "--cell mlp --M 12 --method exr"}
SEEDS = list(range(1300, 1303)) + list(range(1400, 1410))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--P", type=int, default=24); ap.add_argument("--out", default=os.path.join(HERE, "results", "c1_acq"))
    a = ap.parse_args(); os.makedirs(os.path.join(a.out, "logs"), exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")

    def one(job):
        arm, s = job
        f = os.path.join(a.out, "%s_s%d.json" % (arm, s))
        if not os.path.exists(f):
            with open(os.path.join(a.out, "logs", "%s_s%d.log" % (arm, s)), "w") as lg:
                subprocess.run([sys.executable, "-u", os.path.join(HERE, "c1_acq.py")] + ARMS[arm].split()
                               + ["--seeds", str(s), "--threads", "1", "--out", f], stdout=lg, stderr=subprocess.STDOUT, env=env)
        return os.path.exists(f)

    jobs = [(arm, s) for arm in ARMS for s in SEEDS]
    jobs.sort(key=lambda j: (0 if j[0] in ("B_temper", "B_eta1", "A_opb24") else 1))   # slowest first
    t0 = time.time()
    with ThreadPoolExecutor(a.P) as ex:
        ok = list(ex.map(one, jobs))
    print("%d jobs, %d missing, %.0f s" % (len(jobs), ok.count(False), time.time() - t0), flush=True)
    open(os.path.join(a.out, "DONE"), "w").write(time.strftime("%Y-%m-%d %H:%M:%S %Z") + "\n")


if __name__ == "__main__":
    main()
