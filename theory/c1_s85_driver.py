"""THEORY §85 driver: A5 free table, (i) curriculum mixP with p in {0.001, 0.01, 0.03, 0.1}, (ii) ge3 with --init_oracle
eps in {0.5, 1, 2, 4}; seeds 2121-2125; M 120, exact marginal, lr 1e-2, tau 4 -> 1, 16k updates; resumable. Output
theory/results/c1_s85/<arm>_s<seed>.json.
    python theory/c1_s85_driver.py --device cuda:0 --P 3
"""
import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = list(range(2121, 2126))
ARMS = {**{"mixP%g" % p: ["--curriculum", "mixP", "--mix_p", str(p)] for p in (0.001, 0.01, 0.03, 0.1)},
        **{"orac%g" % e: ["--curriculum", "ge3", "--init_oracle", str(e)] for e in (0.5, 1, 2, 4)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu"); ap.add_argument("--P", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "c1_s85"))
    a = ap.parse_args(); os.makedirs(os.path.join(a.out, "logs"), exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")

    def one(job):
        arm, s = job; f = os.path.join(a.out, "%s_s%d.json" % (arm, s))
        if not os.path.exists(f):
            with open(os.path.join(a.out, "logs", "%s_s%d.log" % (arm, s)), "w") as lg:
                subprocess.run([sys.executable, "-u", os.path.join(HERE, "c1_rl.py"), "--group", "A5", "--cell", "tab", "--M", "120",
                                "--method", "marg", "--tau0", "4", "--updates", "16000", "--seeds", str(s), "--threads", "1",
                                "--save_table", "--device", a.device, "--out", f] + ARMS[arm], stdout=lg, stderr=subprocess.STDOUT, env=env)
        return os.path.exists(f)

    jobs = [(arm, s) for arm in ARMS for s in SEEDS]
    with ThreadPoolExecutor(a.P) as ex:
        ok = list(ex.map(one, jobs))
    print("%d jobs, %d missing" % (len(jobs), ok.count(False)))
    open(os.path.join(a.out, "DONE"), "w").write("done\n")


if __name__ == "__main__":
    main()
