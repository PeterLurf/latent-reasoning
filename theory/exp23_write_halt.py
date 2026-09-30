"""THEORY §49 (review §47, test 2; C5 v1-v2): a learned write head and a closure halt for checkpointed segments. CPU.

Models: the §45 CLR1 students (frozen). Per model:
  * write head: s_v = <W h_K, id_emb[v]> (linear map from the segment's last thought through the model's id table),
    trained on CPU (Adam, 400 steps, batch 256, fresh long-chain training graphs, seed 3) to predict the model's own
    attention-decoded anchor (target of the sweep head's most-attended edge at step K): self-distillation, no labels.
    Reported: held-out agreement with the attention anchor.
  * chains: S segments (single-anchor restarts, history cut) on §37's 24-hop probe with anchors written by the head
    (head_S*) vs by attention (attn_S*); bal-acc by distance.
  * closure halt: on probes with chains of 12, 18 and 24 hops (deep_chain R = 2, 3, 4; widths 2-3; 512 graphs; seed 9),
    run up to 8 head-written segments; a graph halts at the first segment after which no query's max-combined answer
    newly turns reachable; reported: halting segment per graph by chain length, and accuracy at halt.
    python theory/exp23_write_halt.py --out theory/results/exp23_write_halt.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.set_num_threads(int(os.environ.get("THEORY_THREADS", "2")))
from deep_chain import deep_chain_graphs  # noqa: E402
from eval_arch import load_arch  # noqa: E402
from latent.length_gen import long_chain_graphs  # noqa: E402
from rollout import rollout  # noqa: E402

T = lambda a: torch.as_tensor(a)


@torch.no_grad()
def seg(m, K, g, start, head, Q):
    o = rollout(m, K, T(g.src), T(g.dst), T(start), K, qnodes=Q)
    att = o["attn1"][K - 1][:, head, :o["P"] - 1]
    anc = g.dst[np.arange(len(g)), att.argmax(1).numpy()]
    return (o["qlogit"].numpy() if Q is not None else None), o["thoughts"][:, K], anc


def train_head(m, K, head, steps=400, bs=256, seed=3):
    rng = np.random.default_rng(seed)
    E = m.id_emb.weight.detach()
    Wl = torch.nn.Linear(m.d, E.shape[1])
    opt = torch.optim.Adam(Wl.parameters(), lr=3e-3)
    for _ in range(steps):
        g, br = long_chain_graphs(rng, bs, [2, 4, 6])
        _, h, anc = seg(m, K, g, g.start, head, None)
        loss = F.cross_entropy(Wl(h) @ E.T, T(anc).long())
        opt.zero_grad(); loss.backward(); opt.step()
    return lambda h: (Wl(h) @ E.T).argmax(1).numpy()


@torch.no_grad()
def chain(m, K, head, g, s, writer, S):
    n = len(g)
    Q = T(np.concatenate([s["reach"], s["decoy"]], 1)); L = s["reach"].shape[1]
    start = g.start; best = None; out = {}; newpos = []
    for k in range(1, S + 1):
        ans, h, anc = seg(m, K, g, start, head, Q)
        prev = best
        best = ans if best is None else np.maximum(best, ans)
        newpos.append(((best > 0) & ~(prev > 0)).sum(1) if prev is not None else (best > 0).sum(1))
        p = best > 0
        out[k] = [float(0.5 * (p[:, j].mean() + 1 - p[:, L + j].mean())) for j in range(L)]
        start = writer(h) if writer is not None else anc
    return out, np.stack(newpos, 1), best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="theory/results/exp23_write_halt.json")
    a = ap.parse_args()
    H = [102, 111, 113, 118, 125, 130, 132, 140, 146, 157]
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    g24, s24 = deep_chain_graphs(np.random.default_rng(7), 512, [2, 3], 4)
    ghold, _ = long_chain_graphs(np.random.default_rng(5), 512, [2, 4, 6])
    probes = {R: deep_chain_graphs(np.random.default_rng(9), 512, [2, 3], R) for R in (2, 3, 4)}
    for sd in H:
        name = f"pclr1_s{sd}"
        if name in res:
            continue
        path = f"theory/runs/{name}"
        m, c = load_arch(path, "cpu"); K = c["K"]
        head = json.load(open(os.path.join(path, "arch.json")))["clr"]["head"]
        writer = train_head(m, K, head)
        with torch.no_grad():
            _, h, anc = seg(m, K, ghold, ghold.start, head, None)
            agree = float((writer(h) == anc).mean())
        o = {"agree_heldout": agree}
        ch, _, _ = chain(m, K, head, g24, s24, writer, 6)
        ca, _, _ = chain(m, K, head, g24, s24, None, 6)
        o["head"] = {str(k): v for k, v in ch.items()}; o["attn"] = {str(k): v for k, v in ca.items()}
        halt = {}
        for R, (g, s) in probes.items():
            _, newpos, _ = chain(m, K, head, g, s, writer, 8)
            hs = []
            for i in range(len(g)):
                z = np.where(newpos[i, 1:] == 0)[0]
                hs.append(int(z[0]) + 1 if len(z) else 8)                 # segments run before halting
            halt[str(6 * R)] = {"median_halt_segments": float(np.median(hs)), "mean": float(np.mean(hs))}
        o["halt"] = halt
        res[name] = o
        json.dump(res, open(a.out, "w", newline="\n"), indent=1)
        print(name, "agree %.2f | head S6 d19-24 %.3f attn %.3f | halt median by length %s" % (
            agree, np.mean(ch[6][18:24]), np.mean(ca[6][18:24]),
            {L_: halt[L_]["median_halt_segments"] for L_ in halt}), flush=True)


if __name__ == "__main__":
    main()
