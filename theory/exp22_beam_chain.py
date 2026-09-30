"""THEORY §48 (review §47, test 1): do multi-start beams fix error compounding across checkpointed segments? CPU.

Models: the §45 CLR1 students (single-anchor restarts, trained label-free). Probe: §37's 24-hop chain (seed 7, widths
2-3, 512 graphs). Segment = K latent steps from one written anchor (a single START token, history cut). Beam width B:
  * segment 1 from START; each episode proposes its B most attended edge targets (sweep head, last step) with their
    attention; the next segment's B anchors are the B distinct candidates with the largest attention over all episodes
  * B = 1 is §45's chain; answers are max-combined over every episode of every segment
Reported: bal-acc by distance 1..24 after S = 2..6 segments, for B = 1 and B = 3.
    python theory/exp22_beam_chain.py --out theory/results/exp22_beam_chain.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.set_num_threads(int(os.environ.get("THEORY_THREADS", "2")))
from deep_chain import deep_chain_graphs  # noqa: E402
from eval_arch import load_arch  # noqa: E402
from rollout import rollout  # noqa: E402

T = lambda a: torch.as_tensor(a)


@torch.no_grad()
def chain(m, K, head, g, s, B, S=6):
    n = len(g); bi = np.arange(n)
    Q = T(np.concatenate([s["reach"], s["decoy"]], 1)); L = s["reach"].shape[1]
    anchors = g.start[:, None]
    best = None; out = {}
    for seg in range(1, S + 1):
        cand_t, cand_a = [], []
        for b in range(anchors.shape[1]):
            o = rollout(m, K, T(g.src), T(g.dst), T(anchors[:, b]), K, qnodes=Q)
            best = o["qlogit"].numpy() if best is None else np.maximum(best, o["qlogit"].numpy())
            att = o["attn1"][K - 1][:, head, :o["P"] - 1].numpy()
            top = np.argsort(-att, 1)[:, :B]
            cand_t.append(g.dst[bi[:, None], top]); cand_a.append(np.take_along_axis(att, top, 1))
        ct, ca = np.concatenate(cand_t, 1), np.concatenate(cand_a, 1)
        new = np.zeros((n, B), dtype=anchors.dtype)
        for i in range(n):
            seen, row = set(), []
            for j in np.argsort(-ca[i]):
                if ct[i, j] not in seen:
                    seen.add(ct[i, j]); row.append(ct[i, j])
                if len(row) == B:
                    break
            while len(row) < B:
                row.append(row[0])
            new[i] = row
        anchors = new
        if seg >= 2:
            p = best > 0
            out[f"S{seg}"] = [float(0.5 * (p[:, j].mean() + 1 - p[:, L + j].mean())) for j in range(L)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="theory/results/exp22_beam_chain.json")
    a = ap.parse_args()
    H = [102, 111, 113, 118, 125, 130, 132, 140, 146, 157]
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    g, s = deep_chain_graphs(np.random.default_rng(7), 512, [2, 3], 4)
    for sd in H:
        name = f"pclr1_s{sd}"
        if name in res:
            continue
        path = f"theory/runs/{name}"
        m, c = load_arch(path, "cpu")
        head = json.load(open(os.path.join(path, "arch.json")))["clr"]["head"]
        res[name] = {f"B{B}": chain(m, c["K"], head, g, s, B) for B in (1, 3)}
        json.dump(res, open(a.out, "w", newline="\n"), indent=1)
        f = lambda B, S, lo, hi: np.mean(res[name][f"B{B}"][f"S{S}"][lo:hi])
        print(name, "S4 d13-18 B1 %.3f B3 %.3f | S4 d19-24 B1 %.3f B3 %.3f | S6 d19-24 B1 %.3f B3 %.3f" % (
            f(1, 4, 12, 18), f(3, 4, 12, 18), f(1, 4, 18, 24), f(3, 4, 18, 24), f(1, 6, 18, 24), f(3, 6, 18, 24)), flush=True)


if __name__ == "__main__":
    main()
