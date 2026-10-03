"""THEORY §96 driver: S5 free-table teachers (exact marginal, final answers only) with saved tables, for amortised
self-labels. Arms: mixP p = 0.05 (the "little enough" dose) and p = 0.1, and std (reference); seeds 2161-2165;
M 240 (= 2|G|), lr 1e-2, tau 4 -> 1, 16k updates (as §76 / §85); resumable. Output theory/results/c1_s96/<arm>_s<seed>.json.
    python theory/c1_s96_driver.py --device cuda:0 --P 2
"""
import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = list(range(2161, 2166))
ARMS = {"mixP0.05": ["--curriculum", "mixP", "--mix_p", "0.05"], "mixP0.1": ["--curriculum", "mixP", "--mix_p", "0.1"],
        "std": ["--curriculum", "std"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu"); ap.add_argument("--P", type=int, default=2)
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    ap.add_argument("--out", default=os.path.join(HERE, "results", "c1_s96"))
    a = ap.parse_args(); os.makedirs(os.path.join(a.out, "logs"), exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")

    def one(job):
        arm, s = job; f = os.path.join(a.out, "%s_s%d.json" % (arm, s))
        if not os.path.exists(f):
            with open(os.path.join(a.out, "logs", "%s_s%d.log" % (arm, s)), "w") as lg:
                subprocess.run([sys.executable, "-u", os.path.join(HERE, "c1_rl.py"), "--group", "S5", "--cell", "tab", "--M", "240",
                                "--method", "marg", "--tau0", "4", "--updates", "16000", "--seeds", str(s), "--threads", "1",
                                "--save_table", "--device", a.device, "--out", f] + ARMS[arm], stdout=lg, stderr=subprocess.STDOUT, env=env)
        return os.path.exists(f)

    jobs = [(arm, s) for arm in a.arms for s in SEEDS]
    with ThreadPoolExecutor(a.P) as ex:
        ok = list(ex.map(one, jobs))
    print("%d jobs, %d missing" % (len(jobs), ok.count(False)))
    open(os.path.join(a.out, "DONE"), "w").write("done\n")


if __name__ == "__main__":
    main()
