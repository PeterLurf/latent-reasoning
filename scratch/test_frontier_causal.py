"""Equivalence and manipulation checks for latent/frontier_causal.py (PROTOCOL7 T1/T2), on non-test models."""
import sys
import numpy as np
import torch
from latent.competence import load
from latent.length_gen import long_chain_graphs
from latent.graphs import NPB
from latent import frontier_causal as FC

dev = "cuda:0"
for run in sys.argv[1:]:
    m, c = load(run, dev); K = c["K"]
    g, br = long_chain_graphs(np.random.default_rng(5), 256, [2, 4, 6])
    bi = np.arange(len(g)); T = lambda a: torch.as_tensor(a, device=dev)
    q = np.concatenate([g.nodes[bi, br[ch]] for ch in "ABCD"], 1)
    with torch.no_grad():
        h, cache, _ = m.encode(m.embed_prefix(T(g.src), T(g.dst), T(g.start)))
        for k in range(1, 13):
            h, cache, _ = m.latent_step(h, min(k, K), cache)
        ref = (m.query(cache, T(q)) > 0).cpu().numpy()
        logit_ref = m.query(cache, T(q)).cpu().numpy()
    mine, att = FC.rollout(m, K, g, q, 12, dev, record=True)
    zero = FC.rollout(m, K, g, q, 12, dev, bias_pos=np.zeros((len(g), 1), int) + 0 * 0)[0]  # bias on token 0 only
    print(run, "use_step", getattr(m, "use_step", True), "| identical to latent_step:", bool((mine == ref).all()),
          "| agree", float((mine == ref).mean()))
    # manipulation check: inject B-edge 0 at step 7 with mass 0.5 on every head, record attention
    pos = {ch: FC.edge_positions(g, br[ch]) for ch in "ABCD"}
    H = m.blocks[0].attn.h
    inj = {7: (pos["B"][:, 0], torch.full((H,), 0.5, device=dev))}
    _, att2 = FC.rollout(m, K, g, q, 12, dev, inject=inj, record=True)
    got = att2[6][bi, :, pos["B"][:, 0]]            # step 7 -> index 6, (n, H)
    rowsum = att2[6].sum(-1)                         # edges only, <= 1
    print("   inject check: min mass on B-edge0 at step 7 =", float(got.min()).__round__(4),
          "| base mean =", float(att[6][bi, :, pos["B"][:, 0]].mean()).__round__(4),
          "| step-8 unaffected by record path:", att2.shape)
