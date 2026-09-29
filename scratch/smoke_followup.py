import time

import numpy as np

import latent.followup as F
from latent.graphs import sample_graphs

t0 = time.time()
A = F.load_assets("runs/r3_pc_gated_gpt2", "cuda:1")
k = A["k_star"]
rng = np.random.default_rng(0)
g = sample_graphs(rng, 40, widths=[4], split="discovery")
sig = lambda x: 1 / (1 + np.exp(-x))

r = F.dose_records(A, g, k)
print("dose: shapes", r["pat"].shape, "dose0==base", np.abs(r["pat"][:, 0] - r["base"][None]).max())
qb = r["q_branch"]
edited = np.stack([qb == r["att_idx"][:, [w]] for w in range(r["W"])], 1)          # (N, W, Q)
for ki, kind in enumerate(r["kinds"]):
    tg = [float((sig(r["base"]) - sig(r["pat"][ki, di]))[edited].mean()) for di in range(len(r["doses"]))]
    print(f"  {kind:8s} targeted by dose {np.round(tg, 3)}  proj/diff norm {r['proj_norm'][ki].mean() / r['diff_norm'].mean():.3f}")
print("  clean counterfactual gap", float((sig(r["base"]) - sig(r["don"]))[edited].mean()))

s = F.swap_records(A, g, k, n_pairs=1)
print("swap: shapes", s["pat"].shape)
c = F.continuation_records(A, sample_graphs(rng, 60, widths=[4], split="discovery"), k, j=2)
print("continuation: n", c["n"], {m: round(float((sig(c["edit_full_Dn"]) - sig(c["R"]))[c[f"mask_{m}"]].mean()), 3)
                                     for m in ("C_rec", "C_don", "B_shared", "other_attached")},
      "natural", {m: round(float((sig(c["Nt"]) - sig(c["R"]))[c[f"mask_{m}"]].mean()), 3) for m in ("C_rec", "C_don", "B_shared")})
cr = F.cache_records(A, g, k, n_slots=2)
print("cache: TLG==don", float(np.abs(cr["TLG"] - cr["don"]).max()), "conds", list(cr["conds"]))
a = F.attention_records(A, g, n_slots=2)
print("attn: shapes", a["att_boost"].shape)
gr = F.gain_restore_records(A, g, k, n_slots=2)
print("gain: keys", [x for x in gr if x.endswith(("base", "pat", "don"))])
j = F.jacobian_records(A, g, k, n_slots=2)
print("jac: A", j["A"].shape, "Jnorm mean", float(j["Jnorm"].mean()))
print("total", round(time.time() - t0, 1), "s")
