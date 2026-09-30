"""THEORY §50 (C5 v3): a halting policy learned from final-answer reward only (exact expected reward over cached
trajectories; not RL), on checkpointed segments. CPU.

Models: the §45 CLR1 students (frozen) with §49's write head (same training, seed 3). Because the model and the writer
are frozen, the segment trajectory does not depend on the halting decision, so segments 1..8 are cached once per graph:
the last thought h_k of each segment and the per-graph balanced accuracy of the max-combined answers after k segments
(TPR on the reachable chain, TNR on the look-alike, over the chain's full length).
Policy: stop probability pi_k = sigmoid(w . h_k + b) (reads only the thought; no step index), forced stop at 8.
Objective (final-answer reward only): maximise E[acc(stop) - c * stop], c = 0.01 per segment, computed exactly from the
cached trajectory (P(stop = k) = pi_k prod_{j<k} (1 - pi_j)); Adam, 300 steps, full batch.
Train: deep_chain graphs with 12/18/24-hop chains (R = 2, 3, 4; widths 2-3; 512 each; seed 21).
Eval: §49's halting probes (seed 9, R = 2, 3, 4). Greedy stop = first k with pi_k > 0.5. Compared with §49's closure
rule (first segment adding no new reachable answer) and with fixed S = 6, on the same eval graphs.
    python theory/exp24_learned_halt.py --out theory/results/exp24_learned_halt.json
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
from deep_chain import deep_chain_graphs  # noqa: E402
from eval_arch import load_arch  # noqa: E402
from exp23_write_halt import seg, train_head  # noqa: E402

T = lambda a: torch.as_tensor(a)
SMAX, COST = 8, 0.01


@torch.no_grad()
def cache(m, K, head, writer, g, s):
    n = len(g); L = s["reach"].shape[1]
    Q = T(np.concatenate([s["reach"], s["decoy"]], 1))
    start = g.start; best = None; H, acc, newpos = [], [], []
    for k in range(SMAX):
        ans, h, _ = seg(m, K, g, start, head, Q)
        prev = best
        best = ans if best is None else np.maximum(best, ans)
        p = best > 0
        acc.append(0.5 * (p[:, :L].mean(1) + 1 - p[:, L:].mean(1)))
        newpos.append(((best > 0) & ~(prev > 0)).sum(1) if prev is not None else p.sum(1))
        H.append(h.clone())
        start = writer(h)
    return torch.stack(H, 1), np.stack(acc, 1), np.stack(newpos, 1)


def stop_dist(pi):
    """pi: (n, SMAX) stop probabilities -> (n, SMAX) P(stop at k), forced stop at the last segment."""
    surv = torch.cumprod(torch.cat([torch.ones_like(pi[:, :1]), 1 - pi[:, :-1]], 1), 1)
    p = surv * pi
    p = torch.cat([p[:, :-1], surv[:, -1:]], 1)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="theory/results/exp24_learned_halt.json")
    a = ap.parse_args()
    Hs = [102, 111, 113, 118, 125, 130, 132, 140, 146, 157]
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    train = [deep_chain_graphs(np.random.default_rng(21 + R), 512, [2, 3], R) for R in (2, 3, 4)]
    evalp = {R: deep_chain_graphs(np.random.default_rng(9), 512, [2, 3], R) for R in (2, 3, 4)}
    for sd in Hs:
        name = f"pclr1_s{sd}"
        if name in res:
            continue
        path = f"theory/runs/{name}"
        m, c = load_arch(path, "cpu"); K = c["K"]
        head = json.load(open(os.path.join(path, "arch.json")))["clr"]["head"]
        writer = train_head(m, K, head)
        tr = [cache(m, K, head, writer, g, s) for g, s in train]
        Htr = torch.cat([t[0] for t in tr]); Atr = torch.as_tensor(np.concatenate([t[1] for t in tr])).float()
        torch.manual_seed(0)
        pol = torch.nn.Linear(Htr.shape[-1], 1)
        opt = torch.optim.Adam(pol.parameters(), lr=1e-2)
        costs = COST * torch.arange(1, SMAX + 1).float()
        for _ in range(300):
            pi = torch.sigmoid(pol(Htr)).squeeze(-1)
            loss = -(stop_dist(pi) * (Atr - costs)).sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        fixed_best = int(np.argmax((Atr.numpy() - costs.numpy()).mean(0)))      # best fixed segment count, chosen on the train cache
        o = {"fixed_best_segments": fixed_best + 1}
        for R, (g, s) in evalp.items():
            He, Ae, NPe = cache(m, K, head, writer, g, s)
            with torch.no_grad():
                pi = torch.sigmoid(pol(He)).squeeze(-1).numpy()
            learned = np.array([next((k for k in range(SMAX) if pi[i, k] > 0.5), SMAX - 1) for i in range(len(g))])
            # §49's rule: stop after the last segment that still added a reachable answer
            closure = np.array([next((k - 1 for k in range(1, SMAX) if NPe[i, k] == 0), SMAX - 1) for i in range(len(g))])
            rew = lambda st: float(np.mean(Ae[np.arange(len(g)), st] - COST * (st + 1)))
            acc = lambda st: float(np.mean(Ae[np.arange(len(g)), st]))
            six = np.full(len(g), 5)
            fb = np.full(len(g), fixed_best)
            o[str(6 * R)] = {"learned_median_segments": float(np.median(learned + 1)), "closure_median_segments": float(np.median(closure + 1)),
                             "reward_learned": rew(learned), "reward_closure": rew(closure), "reward_fixed6": rew(six),
                             "acc_learned": acc(learned), "acc_closure": acc(closure), "acc_fixed6": acc(six),
                             "reward_fixed_best": rew(fb), "acc_fixed_best": acc(fb),
                             "acc_by_segments": [float(x) for x in Ae.mean(0)]}
        res[name] = o
        json.dump(res, open(a.out, "w", newline="\n"), indent=1)
        print(name, {L_: (o[L_]["learned_median_segments"], round(o[L_]["reward_learned"], 3), round(o[L_]["reward_closure"], 3), round(o[L_]["reward_fixed_best"], 3)) for L_ in ("12", "18", "24")}, flush=True)


if __name__ == "__main__":
    main()
