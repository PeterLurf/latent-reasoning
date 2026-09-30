"""THEORY §44 (NORTH_STAR open question 2): can a trained model restart one latent episode from a *set* of written
anchors (several START tokens) without retraining? CPU, measurement only, the 59 P30 ∪ L models of §26.

Episode 1: K steps from START. Anchors a1..a3 = targets of the 3 edges most attended by the sweep head at step K (§26).
Conditions (answers max-combined with episode 1, as in §26; long-chain probe(512), bal-acc by distance):
  * top1   : one more episode from a1                                           (single written anchor)
  * top3max: three more episodes, one from each of a1, a2, a3                   (§26 re2top3; 3 extra episodes)
  * set3   : ONE more episode whose prefix ends in START(a3), START(a2), START(a1) (a1 last, so the loop starts there)
  * set3r  : the same with a2, a3 replaced by random nodes of the graph that are not on the probed chains (control:
             do extra START tokens carry information, or are they ignored / harmful?)
    python theory/exp19_set_restart.py --out theory/results/exp19_set_restart.json
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
from exp12_reanchor import bal_by_dist  # noqa: E402
from latent.model import T_START  # noqa: E402
from rollout import load, probe, rollout  # noqa: E402

T = lambda a: torch.as_tensor(a)


@torch.no_grad()
def set_episode(m, K, src, dst, starts, q):
    """One episode whose prefix ends in START tokens for starts[:, 0..W-1]; the last one seeds the loop."""
    x = m.embed_prefix(src, dst, T(starts[:, -1]))
    te = m.type_emb.weight
    extra = [(m.node_emb(T(starts[:, w])) + te[T_START])[:, None] for w in range(starts.shape[1] - 1)]
    x = torch.cat([x[:, :-1]] + extra + [x[:, -1:]], 1)
    h, cache, _ = m.encode(x)
    for k in range(1, K + 1):
        h, cache, _ = m.latent_step(h, k, cache)
    return m.query(cache, q)


@torch.no_grad()
def run(m, K, head, g, br, seed=0):
    n = len(g); bi = np.arange(n)
    q = np.concatenate([g.nodes[bi, br[c]] for c in ("A", "B", "C", "D")], 1)
    Q = T(q)
    src, dst = T(g.src), T(g.dst)
    o1 = rollout(m, K, src, dst, T(g.start), K, qnodes=Q)
    base = o1["qlogit"].numpy()
    a = o1["attn1"][K - 1][:, head, :o1["P"] - 1]
    order = torch.argsort(a, 1, descending=True).numpy()
    anc = np.stack([g.dst[bi, order[:, r]] for r in range(3)], 1)                   # (n, 3): a1, a2, a3
    ep = lambda s: rollout(m, K, src, dst, T(s), K, qnodes=Q)["qlogit"].numpy()
    out = {}
    e1 = ep(anc[:, 0])
    out["top1"] = bal_by_dist(np.maximum(base, e1), n)
    mx = np.maximum(base, e1)
    for r in (1, 2):
        mx = np.maximum(mx, ep(anc[:, r]))
    out["top3max"] = bal_by_dist(mx, n)
    out["set3"] = bal_by_dist(np.maximum(base, set_episode(m, K, src, dst, anc[:, [2, 1, 0]], Q).numpy()), n)
    rng = np.random.default_rng(seed)
    onchain = set(); rand = np.zeros((n, 2), dtype=anc.dtype)
    for i in range(n):
        excl = set(q[i].tolist())
        pool = [v for v in set(g.src[i].tolist()) | set(g.dst[i].tolist()) if v not in excl]
        rand[i] = rng.choice(pool, 2, replace=len(pool) < 2)
    out["set3r"] = bal_by_dist(np.maximum(base, set_episode(m, K, src, dst, np.c_[rand, anc[:, 0]], Q).numpy()), n)
    out["single"] = bal_by_dist(base, n)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="theory/results/exp19_set_restart.json")
    ap.add_argument("--runs", nargs="*")
    a = ap.parse_args()
    R = json.load(open("theory/results/exp12_reanchor.json"))
    names = a.runs or sorted(R, key=lambda r: (R[r]["extrap"], r))
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    g, br, _ = probe(512)
    for name in names:
        if name in res:
            continue
        m, c = load("runs/" + name, "cpu")
        o = run(m, c["K"], R[name]["head"], g, br)
        o.update(K=c["K"], extrap=R[name]["extrap"])
        res[name] = o
        json.dump(res, open(a.out, "w", newline="\n"), indent=1)
        f = lambda k: "%.3f" % np.mean(o[k][7:10])
        print("%-10s %s | far8-10 top1 %s top3max %s set3 %s set3r %s" % (
            name, "EXT" if o["extrap"] else "   ", f("top1"), f("top3max"), f("set3"), f("set3r")), flush=True)


if __name__ == "__main__":
    main()
