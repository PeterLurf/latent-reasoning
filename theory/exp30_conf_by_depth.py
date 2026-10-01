"""THEORY §62 NC-5: per-depth confidence of distilled students (the signature of calibration pressure). CPU.

On 512 training-distribution long chains (latent.length_gen.long_chain_graphs, widths 2,4,6, seed 2030), one episode at
K' = 12 from START; the mean sigmoid probability on the reachable chain nodes at distance 1..12 (A0..A5, B0..B5) and on
their look-alikes. Reported per model: conf_reach[1..12], conf_decoy[1..12], and the step d6 -> d7 on reachable nodes.
    python theory/exp30_conf_by_depth.py --runs theory/runs/<student> ... --out theory/results/exp30_conf_by_depth.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.set_num_threads(int(os.environ.get("THEORY_THREADS", "3")))
from eval_arch import load_arch  # noqa: E402
from latent.length_gen import long_chain_graphs  # noqa: E402
from rollout import load, rollout  # noqa: E402

T = lambda a: torch.as_tensor(a)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default="theory/results/exp30_conf_by_depth.json")
    a = ap.parse_args()
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    g, br = long_chain_graphs(np.random.default_rng(2030), 512, [2, 4, 6])
    bi = np.arange(len(g))
    reach = np.concatenate([g.nodes[bi, br["A"]], g.nodes[bi, br["B"]]], 1)
    decoy = np.concatenate([g.nodes[bi, br["D"]], g.nodes[bi, br["C"]]], 1)
    Q = T(np.concatenate([reach, decoy], 1)).long()
    for path in a.runs:
        name = os.path.basename(path.rstrip("/\\"))
        if name in res:
            continue
        m, c = (load_arch if path.startswith("theory/") else load)(path, "cpu"); m.eval()
        p = torch.sigmoid(rollout(m, c["K"], T(g.src), T(g.dst), T(g.start), 12, qnodes=Q)["qlogit"].float()).numpy()
        o = {"conf_reach": [float(v) for v in p[:, :12].mean(0)], "conf_decoy": [float(v) for v in p[:, 12:].mean(0)]}
        o["step_d6_d7"] = o["conf_reach"][5] - o["conf_reach"][6]
        res[name] = o
        json.dump(res, open(a.out, "w", newline="\n"), indent=1)
        print("%-14s reach conf d1-6 %.3f d7-12 %.3f | step d6->d7 %+.3f | decoy conf d1-6 %.3f d7-12 %.3f" % (
            name, np.mean(o["conf_reach"][:6]), np.mean(o["conf_reach"][6:]), o["step_d6_d7"],
            np.mean(o["conf_decoy"][:6]), np.mean(o["conf_decoy"][6:])), flush=True)


if __name__ == "__main__":
    main()
