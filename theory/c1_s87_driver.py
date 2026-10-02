"""THEORY §87 driver: A5 free table, ge3 basin entry: (i) warm start (short batches p 0.1 for the first 1/4 or 1/8 of updates), (ii) deterministic annealing tau0 16 or 64 over 3/4 of the run;
seeds 2131-2135; M 120, exact marginal, lr 1e-2, tau 4 -> 1, 16k updates; resumable. Output
theory/results/c1_s87/<arm>_s<seed>.json.
    python theory/c1_s87_driver.py --device cuda:0 --P 3
"""
import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = list(range(2131, 2136))
ARMS = {"warm25": ["--curriculum", "warm", "--mix_p", "0.1", "--warm_frac", "0.25"],
        "warm12": ["--curriculum", "warm", "--mix_p", "0.1", "--warm_frac", "0.125"],
        "da16": ["--curriculum", "ge3", "--tau_frac", "0.75", "--tau0_override", "16"],
        "da64": ["--curriculum", "ge3", "--tau_frac", "0.75", "--tau0_override", "64"],
        "daem": ["--curriculum", "ge3", "--daem_beta0", "0.1"]}       # 87-2: canonical DA-EM (tempered path posterior)


def tau(args):                                   # tau0 = 4 unless the arm overrides it (deterministic annealing)
    if "--tau0_override" in args:
        i = args.index("--tau0_override"); return args[:i] + ["--tau0", args[i + 1]] + args[i + 2:]
    return args + ["--tau0", "4"]


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
                                "--method", "marg", "--updates", "16000", "--seeds", str(s), "--threads", "1",
                                "--save_table", "--device", a.device, "--out", f] + tau(ARMS[arm]), stdout=lg, stderr=subprocess.STDOUT, env=env)
        return os.path.exists(f)

    jobs = [(arm, s) for arm in ARMS for s in SEEDS]
    with ThreadPoolExecutor(a.P) as ex:
        ok = list(ex.map(one, jobs))
    print("%d jobs, %d missing" % (len(jobs), ok.count(False)))
    open(os.path.join(a.out, "DONE"), "w").write("done\n")


if __name__ == "__main__":
    main()
