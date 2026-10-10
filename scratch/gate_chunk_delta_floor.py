"""g2 detail (K = 10, no ckpt / cache, fp64): per-parameter rel diff chunk-vs-loop, and for mem.s_add the loop's own fp64 rounding scale:
loop vs the loop with the delta heads' state accumulated in a different association (chunk C = 1 / 16 / 32 vs 64) -> how far apart do exact-arithmetic-equal
fp64 computations land on this scalar?"""
import json
import os
import sys

sys.path.insert(0, "D:/Latent"); os.chdir("D:/Latent")
import torch  # noqa: E402

sys.argv = ["x"]; sys.path.insert(0, "D:/Latent/scratch"); import gate_chunk_delta as G  # noqa: E402
DM = G.DM


def run(m, D, x, msk, cd):
    DM.CHUNK_DELTA = cd; m.zero_grad(set_to_none=True); S = m.mem.zero_state(2, "cpu").double()
    for _ in range(10):
        S = m.write(S, x, msk)
    m.dec_loss(S, D, "cpu").backward()
    return {n_: p_.grad.detach().clone() for n_, p_ in m.named_parameters() if p_.grad is not None}


torch.set_num_threads(3); _f = torch.Tensor.float; torch.Tensor.float = lambda self, *a, **k: self.double(); DM.FAST_ADD = True
P, m = G.prosmem(True); D = P.load("prosqa/p8data/p8_na_train.json")[:2]; x, msk = m.think_inputs(D, "cpu")
gr = {cd: run(m, D, x, msk, cd) for cd in (0, 64, 32, 16, 1)}
r64 = {k: G.rel(gr[0][k], gr[64][k]) for k in gr[0]}; top = sorted(r64.items(), key=lambda kv: -kv[1])[:6]
out = {"top_rel_chunk64_vs_loop": top, "s_add_grad": gr[0]["mem.s_add"].tolist(), "s_del_grad": gr[0]["mem.s_del"].tolist() if "mem.s_del" in gr[0] else None,
       "s_add_rel_vs_loop_by_C": {str(c): G.rel(gr[0]["mem.s_add"], gr[c]["mem.s_add"]) for c in (64, 32, 16, 1)},
       "s_add_rel_C64_vs_C32": G.rel(gr[64]["mem.s_add"], gr[32]["mem.s_add"]), "s_add_rel_C16_vs_C1": G.rel(gr[16]["mem.s_add"], gr[1]["mem.s_add"]),
       "max_rel_all_params_C1_vs_loop": max(G.rel(gr[0][k], gr[1][k]) for k in gr[0])}
print(json.dumps(out, indent=1))
