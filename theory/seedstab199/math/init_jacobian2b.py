"""N1b: P restricted to span(keys) (where ALL delta writes live), INJECT keys vs READ-SLOTS slot keys (M = 8, wte(' .') + N(0, 0.02^2) after [EOT, expr]),
open-loop path independence at init for both, and the per-pass-decay variant's spectrum. CPU 3 threads idle."""
import json, sys
import numpy as np, torch, torch.nn.functional as F
from gpt2lib import *
torch.manual_seed(1)
rng = np.random.default_rng(1991); out = {}
def run(maps, D0, K):
    Ds = [d.clone() for d in D0]
    for _ in range(K): Ds = [Ds[h] @ maps[h][0] + maps[h][1] for h in range(6)]
    return Ds
zero = [torch.zeros(64, 64, dtype=torch.float64) for _ in range(6)]
def flat(Ds): return torch.cat([d.flatten() for d in Ds])
slot_e = sd["wte.weight"][TID[" ."]][None].repeat(8, 1) + 0.02 * torch.randn(8, D)
for n in [2, 4, 8, 12]:
    res = {"inject": [], "slots": []}; maps_all = {"inject": [], "slots": []}
    for it in range(12):
        root, _ = generate(rng, n); syms = [TID[s] for s in render(root)]
        kd, vd = delta_kv([EOT] + syms + [TID[" ."]])                        # INJECT: every real token of the think input writes
        emb = torch.cat([sd["wte.weight"][torch.tensor([EOT] + syms)], slot_e], 0)
        ks, vs = delta_kv_emb(emb); ks, vs = ks[-8:], vs[-8:]                # READ-SLOTS: only the 8 slot positions write
        for name, (kk, vv) in (("inject", (kd, vd)), ("slots", (ks, vs))):
            maps = [pass_maps(kk[:, h], vv[:, h]) for h in range(6)]; maps_all[name].append(maps)
            rr = [restricted(maps[h][0], kk[:, h]) for h in range(6)]
            G = kk.transpose(0, 1) @ kk.permute(1, 2, 0); L = kk.shape[0]
            offc = float(G[:, ~torch.eye(L, dtype=torch.bool)].abs().mean())
            res[name].append(dict(L=L, top_full=float(np.mean([torch.linalg.svdvals(m[0]).max() for m in maps])),
                                  top_spanK=float(np.mean([r[0] for r in rr])), rho_spanK=float(np.mean([r[1] for r in rr])), dim_spanK=float(np.mean([r[2] for r in rr])),
                                  key_abs_cos=offc))
    agg = {}
    for name in res:
        a = {k: float(np.mean([r[k] for r in res[name]])) for k in res[name][0]}
        ms = maps_all[name]; c16 = []; c8 = []; ratios = []
        for i in range(len(ms)):
            donor = run(ms[(i + 1) % len(ms)], zero, 8)
            c16.append(float(F.cosine_similarity(flat(run(ms[i], zero, 16)), flat(run(ms[i], donor, 16)), dim=0)))
            Ds = [zero]
            for t in range(6): Ds.append(run(ms[i], Ds[-1], 1))
            nr = [float(flat([Ds[t + 1][h] - Ds[t][h] for h in range(6)]).norm()) for t in range(6)]
            ratios.append([np.log(nr[t] / nr[t - 1]) for t in (2, 3, 4)])
        cx = [float(F.cosine_similarity(flat(run(ms[i], zero, 16)), flat(run(ms[(i + 1) % len(ms)], zero, 16)), dim=0)) for i in range(len(ms))]
        a["cross_item_cos16"] = float(np.mean(cx)); a["PDI16"] = float((1 - np.mean(c16)) / (1 - np.mean(cx)))
        a["open_loop_delta_cos16"] = float(np.mean(c16)); a["step_ratio_t234"] = [round(float(x), 3) for x in np.mean(np.array(ratios), 0)]
        agg[name] = a
    out[n] = agg; print(n, json.dumps(agg)); sys.stdout.flush()
json.dump(out, open("init_jacobian2b.json", "w", newline="\n"), indent=1)
