"""Review of SEEDSTAB2 Lemma L: does PER-HEAD normalisation of a superposed store phi_h(a) + psi_h(b) let a linear (cosine) readout compute
op(a, b) mod 11? Learn free codes phi, psi per head and the readout end to end on all pairs; report the best training accuracy per feature map:
  raw      : concat_h (phi_h(a) + psi_h(b))                 (Lemma L's setting: must stay < 1)
  global   : raw / ||raw||, no bias                         (one common positive scale: argmax unchanged, must stay < 1)
  perhead  : concat_h RMS-normalised (phi_h(a) + psi_h(b))  (the draft's SVR features)
Readout: cosine with tau = 10 (+ bias b_c in raw / perhead, as in the draft)."""
import json, os, sys
import torch
torch.set_num_threads(3)
P, H, D = 11, 12, 64
INV = {v: pow(v, P - 2, P) for v in range(1, P)}
OPS = {"+": lambda a, b: (a + b) % P, "*": lambda a, b: (a * b) % P}


def run(op, mode, seed, iters=4000):
    g = torch.Generator().manual_seed(seed)
    bv = list(range(1, P)) if op == "*" else list(range(P))
    A = torch.tensor([a for a in range(P) for b in bv]); B = torch.tensor([b for a in range(P) for b in bv])
    y = torch.tensor([OPS[op](int(a), int(b)) for a, b in zip(A, B)])
    phi = torch.nn.Parameter(torch.randn(P, H, D, generator=g)); psi = torch.nn.Parameter(torch.randn(P, H, D, generator=g))
    W = torch.nn.Parameter(torch.randn(P, H * D, generator=g) * 0.01); bb = torch.nn.Parameter(torch.zeros(P))
    opt = torch.optim.Adam([phi, psi, W, bb], lr=0.02); best = 0.0
    for it in range(iters):
        x = phi[A] + psi[B]                                                   # (N, H, D) superposed store
        if mode == "perhead":
            x = x / torch.sqrt((x ** 2).mean(-1, keepdim=True) + 1e-6)
        f = x.reshape(len(y), -1)
        z = 10.0 * torch.nn.functional.normalize(f, dim=-1) @ torch.nn.functional.normalize(W, dim=-1).T
        if mode != "global":
            z = z + bb
        if mode == "raw":
            z = f @ W.T + bb                                                  # unnormalised linear readout (Lemma L proper)
        loss = torch.nn.functional.cross_entropy(z, y); opt.zero_grad(); loss.backward(); opt.step()
        best = max(best, float((z.argmax(1) == y).float().mean()))
    return best


out = {}
for op in OPS:
    for mode in ("raw", "global", "perhead"):
        r = [run(op, mode, s) for s in (0, 1)]; out["%s_%s" % (op, mode)] = r; print(op, mode, r, flush=True)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "perhead_norm.json"), "w", newline="\n"), indent=1)
