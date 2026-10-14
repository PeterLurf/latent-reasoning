"""SEEDSTAB2 rev 2, check L2: what can SVR's REVISED feature map compute on its own?
Revised map (draft rev 2, sec 4.1): f = concat_h r_h / c_h (c_h a stop-gradient, state-independent per-head constant), then ONE global cosine,
z_c = tau <w_c/|w_c|, f/|f|>, tau = 10, NO bias. argmax_c z_c = argmax_c <w_c/|w_c|, f>: LINEAR in the state at a fixed query.
Old map (rev 1): per-head RMS normalisation of r_h (+ bias) - the reviewer showed it computes + / x mod 11 on superposed codes.
All codes / keys / values / query / readout are LEARNED end to end (free per-operand codes, the most adversarial setting), all inputs, best
training accuracy over Adam steps. H 12 heads, D 64. Stores (what the memory could hold after the think pass wrote per-operand codes):
  super   r_h = phi_h(a) + psi_h(b)                                        (superposed operand codes; Lemma L)
  bilin   Delta_h = v_h(b) k_h(a)^T + v'_h(a) k'_h(b)^T + singles; r_h = Delta_h q_h  (outer-product store: one op is expected to be possible)
  pair3   three operands, all ordered pairwise outer products + singles; target a NESTED op (two reductions)   (degree-2 store)
  erase3  pair3 + one delta-rule erase/write with operand c's key: Delta <- Delta (I - beta k_c k_c^T) + beta v_c k_c^T   (degree-3 terms)
Targets: one op (a o b) for super / bilin; nested (a o b) + c for pair3 / erase3. o in {+, x}. Two seeds each.
Output: linmap_check.json beside this file."""
import json, os
import torch
torch.set_num_threads(3)
P, H, D, TAU = 11, 12, 64, 10.0


def data(op, three):
    nz = op == "*"; av = range(1, P) if nz else range(P)
    f = (lambda a, b: (a * b) % P) if nz else (lambda a, b: (a + b) % P)
    if not three:
        T = torch.tensor([(a, b) for a in av for b in av]); y = torch.tensor([f(a, b) for a, b in T.tolist()]); return T, y
    T = torch.tensor([(a, b, c) for a in av for b in av for c in range(P)]); y = torch.tensor([(f(a, b) + c) % P for a, b, c in T.tolist()]); return T, y


def run(store, op, mapname, seed, iters=3000):
    g = torch.Generator().manual_seed(seed); three = store in ("pair3", "erase3"); T, y = data(op, three); n_op = 3 if three else 2
    rn = lambda *s: torch.nn.Parameter(torch.randn(*s, generator=g) / D ** 0.5)  # noqa: E731
    single = rn(n_op, P, H, D)                                              # superposed single-operand read contributions
    pairs = [(i, j) for i in range(n_op) for j in range(n_op) if i != j]
    K = rn(len(pairs), P, H, D); V = rn(len(pairs), P, H, D); q = rn(H, D); beta = torch.nn.Parameter(torch.zeros(H))
    Ke, Ve = rn(P, H, D), rn(P, H, D)                                       # the erase token's key / value (operand c)
    W = torch.nn.Parameter(torch.randn(P, H * D, generator=g) * 0.01); bb = torch.nn.Parameter(torch.zeros(P))
    ps = [single, K, V, q, beta, Ke, Ve, W, bb]; opt = torch.optim.Adam(ps, lr=0.02); best = 0.0
    for _ in range(iters):
        r = sum(single[i][T[:, i]] for i in range(n_op))                    # (N, H, D)
        if store != "super":                                                # closed-form reads: Delta q = sum_p v_p (k_p . q); no D x D tensor
            qn = torch.nn.functional.normalize(q, dim=-1)
            Kp = [K[p_][T[:, i]] for p_, (i, j) in enumerate(pairs)]; Vp = [V[p_][T[:, j]] for p_, (i, j) in enumerate(pairs)]
            coef = [(k_ * qn[None]).sum(-1) for k_ in Kp]                     # (N, H): k_p . q
            if store == "erase3":                                           # Delta' q = Delta q - beta (Delta kc)(kc . q) + beta vc (kc . q)
                kc = torch.nn.functional.normalize(Ke[T[:, 2]], dim=-1); vc = Ve[T[:, 2]]; b_ = (2 * torch.sigmoid(beta))[None, :]
                kcq = (kc * qn[None]).sum(-1)
                coef = [cf - b_ * (k_ * kc).sum(-1) * kcq for cf, k_ in zip(coef, Kp)]
                r = r + (b_ * kcq)[..., None] * vc
            r = r + sum(cf[..., None] * v_ for cf, v_ in zip(coef, Vp))
        if mapname == "perhead":                                            # rev-1 map: per-head RMS normalisation + bias
            f = (r / torch.sqrt((r ** 2).mean(-1, keepdim=True) + 1e-6)).reshape(len(y), -1)
            z = TAU * torch.nn.functional.normalize(f, dim=-1) @ torch.nn.functional.normalize(W, dim=-1).T + bb
        else:                                                               # rev-2 map: stop-gradient per-head constants, global cosine, no bias
            c = torch.sqrt((r.detach() ** 2).mean((0, 2)) + 1e-6)          # per-head RMS over the batch, detached (a state-independent constant per step)
            f = (r / c[None, :, None]).reshape(len(y), -1)
            z = TAU * torch.nn.functional.normalize(f, dim=-1) @ torch.nn.functional.normalize(W, dim=-1).T
        loss = torch.nn.functional.cross_entropy(z, y); opt.zero_grad(); loss.backward(); opt.step()
        best = max(best, float((z.argmax(1) == y).float().mean()))
    return best


import sys
out = {}
plan = [("super", "+"), ("super", "*"), ("bilin", "+"), ("bilin", "*"), ("pair3", "+"), ("pair3", "*"), ("erase3", "+"), ("erase3", "*")]
only = sys.argv[1].split(",") if len(sys.argv) > 1 else None             # e.g. "pair3_*_perhead,erase3_+_linear" (rev 2: resume after a timeout)
outname = sys.argv[2] if len(sys.argv) > 2 else "linmap_check.json"
for store, op in plan:
    for mp in ("linear", "perhead"):
        key = "%s_%s_%s" % (store, op, mp)
        if only is not None and key not in only:
            continue
        iters = 3000 if store in ("super", "bilin") else 2000
        r_ = [run(store, op, mp, s, iters) for s in (0, 1)]; out[key] = r_
        print("%-7s %s %-8s best train acc %s" % (store, op, mp, ["%.3f" % v for v in r_]), flush=True)
        json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), outname), "w", newline="\n"), indent=1)
