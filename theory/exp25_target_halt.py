"""THEORY §51: the §50 aggregator fix -- single-target answers from the halting segment, with a halt learned from
final-answer reward. Toy analogue of PROTOCOL10's last-segment rule. CPU, cached trajectories.

Models: the §45 CLR1 students (frozen) with §49's write head. Per graph one target t: the reachable chain's node at
distance d ~ U{1..L} or its look-alike (50/50; seed per split). Segments 1..8 (single-anchor restarts) are cached: the
last thought h_k and the target's query logit z_k(t) of segment k alone.
Answer rules at a stop k:  last  = sign(z_k(t))  (non-accumulating)  |  max = sign(max_{j<=k} z_j(t))  (§50's rule).
Stop head (per rule): pi_k = sigmoid(w . [h_k, z_k(t)] + b); trained to maximise the exact expected reward
E[correct(stop) - 0.01 stop] over cached training graphs (12/18/24-hop chains, seeds 31-33; Adam 300 steps).
Held-out: seed 41 graphs of each length. Greedy stop: first k with pi_k > 0.5 (forced at 8).
Reported per rule: held-out reward and accuracy; median stopping segment by target-distance bin for reachable targets
(1-6, 7-12, 13-18, 19-24).
    python theory/exp25_target_halt.py --out theory/results/exp25_target_halt.json
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
from exp24_learned_halt import stop_dist  # noqa: E402

T = lambda a: torch.as_tensor(a)
SMAX, COST = 8, 0.01


@torch.no_grad()
def cache(m, K, head, writer, g, s, seed):
    n = len(g); bi = np.arange(n); L = s["reach"].shape[1]
    rng = np.random.default_rng(seed)
    d = rng.integers(1, L + 1, n); is_reach = rng.random(n) < 0.5
    tgt = np.where(is_reach, s["reach"][bi, d - 1], s["decoy"][bi, d - 1])
    Q = T(tgt[:, None])
    start = g.start; H, Z = [], []
    for k in range(SMAX):
        ans, h, _ = seg(m, K, g, start, head, Q)
        H.append(h.clone()); Z.append(ans[:, 0])
        start = writer(h)
    return torch.stack(H, 1), np.stack(Z, 1), d, is_reach


def correct_table(Z, is_reach, rule):
    zz = np.maximum.accumulate(Z, 1) if rule == "max" else Z
    return ((zz > 0) == is_reach[:, None]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="theory/results/exp25_target_halt.json")
    a = ap.parse_args()
    Hs = [102, 111, 113, 118, 125, 130, 132, 140, 146, 157]
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    tr_g = [deep_chain_graphs(np.random.default_rng(31 + i), 512, [2, 3], R) for i, R in enumerate((2, 3, 4))]
    ev_g = [deep_chain_graphs(np.random.default_rng(41 + i), 512, [2, 3], R) for i, R in enumerate((2, 3, 4))]
    costs = COST * torch.arange(1, SMAX + 1).float()
    for sd in Hs:
        name = f"pclr1_s{sd}"
        if name in res:
            continue
        path = f"theory/runs/{name}"
        m, c = load_arch(path, "cpu"); K = c["K"]
        head = json.load(open(os.path.join(path, "arch.json")))["clr"]["head"]
        writer = train_head(m, K, head)
        tr = [cache(m, K, head, writer, g, s, 131 + i) for i, (g, s) in enumerate(tr_g)]
        ev = [cache(m, K, head, writer, g, s, 141 + i) for i, (g, s) in enumerate(ev_g)]
        cat = lambda xs, j: np.concatenate([x[j] for x in xs])
        Htr, Ztr, dtr, rtr = torch.cat([x[0] for x in tr]), cat(tr, 1), cat(tr, 2), cat(tr, 3)
        Hev, Zev, dev_, rev = torch.cat([x[0] for x in ev]), cat(ev, 1), cat(ev, 2), cat(ev, 3)
        o = {}
        for rule in ("last", "max"):
            Ctr = torch.as_tensor(correct_table(Ztr, rtr, rule))
            zfeat = lambda Z, rule=rule: torch.as_tensor(np.maximum.accumulate(Z, 1) if rule == "max" else Z).float()[..., None]
            Xtr = torch.cat([Htr, zfeat(Ztr)], -1); Xev = torch.cat([Hev, zfeat(Zev)], -1)
            torch.manual_seed(0)
            pol = torch.nn.Linear(Xtr.shape[-1], 1)
            opt = torch.optim.Adam(pol.parameters(), lr=1e-2)
            for _ in range(300):
                loss = -(stop_dist(torch.sigmoid(pol(Xtr)).squeeze(-1)) * (Ctr - costs)).sum(1).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                pi = torch.sigmoid(pol(Xev)).squeeze(-1).numpy()
            stop = np.array([next((k for k in range(SMAX) if pi[i, k] > 0.5), SMAX - 1) for i in range(len(pi))])
            Cev = correct_table(Zev, rev, rule)
            corr = Cev[np.arange(len(stop)), stop]
            bins = {}
            for lo, hi in ((1, 6), (7, 12), (13, 18), (19, 24)):
                sel = rev & (dev_ >= lo) & (dev_ <= hi)
                bins[f"{lo}-{hi}"] = float(np.median(stop[sel] + 1)) if sel.any() else None
            o[rule] = {"reward": float(np.mean(corr - COST * (stop + 1))), "acc": float(corr.mean()),
                       "acc_reach": float(corr[rev].mean()), "acc_decoy": float(corr[~rev].mean()),
                       "median_stop_by_dist_reach": bins, "median_stop_decoy": float(np.median(stop[~rev] + 1))}
        res[name] = o
        json.dump(res, open(a.out, "w", newline="\n"), indent=1)
        print(name, {r: (round(o[r]["reward"], 3), round(o[r]["acc"], 3), o[r]["median_stop_by_dist_reach"]) for r in o}, flush=True)


if __name__ == "__main__":
    main()
