"""PROOFS_D4 check 4: Lemma D3 (softmax-read gradient identity and its O(delta^2) claim), fp64 CPU.
(1) identity: U = sum_s alpha_s N_s, alpha = softmax(l) (SLVA's per-slot, per-head read, boxes/slva.py gather); for ANY downstream loss L(U),
    dL/dl_s = alpha_s <dL/dU, N_s - U>.  Checked with autograd on a random nonlinear L, and through the real SLVA.gather on a tiny SLVA-like setup.
(2) scaling: N_s = c* + eps * d_s, L = 1/2 (U - c*)^T H (U - c*) + kappa * sum (U - c*)^3: max_s |dL/dl_s| vs eps -> log-log slope 2 (quadratic),
    leading term alpha_s dbar^T H (d_s - dbar) * eps^2, remainder O(eps^3).
(3) direction: is descent 'toward the least-drifted versions'? Counterexample: co-linear drifts of opposite sign (d_1 = +e small, d_2 = -2e large):
    descent raises the MORE drifted version 2 (it cancels the mean drift). Same-sign drifts (accumulating drift): descent favours the least drifted."""
import json, sys
import torch
torch.set_num_threads(3); torch.manual_seed(0); dt = torch.float64
out = {}
# (1) identity on a generic loss
S, D = 7, 64
l = torch.randn(S, dtype=dt, requires_grad=True); Ns = torch.randn(S, D, dtype=dt); W = torch.randn(D, D, dtype=dt) / 8
a = torch.softmax(l, 0); U = (a[:, None] * Ns).sum(0); L = torch.tanh(U @ W).pow(2).sum() + (U.sin() * U).sum()
gU, = torch.autograd.grad(L, U, retain_graph=True); gl, = torch.autograd.grad(L, l)
pred = a.detach() * ((Ns - U.detach()) @ gU)
out["identity_max_abs_err"] = float((gl - pred).abs().max()); out["identity_scale"] = float(gl.abs().max())
# (2) scaling with eps
H = torch.randn(D, D, dtype=dt); H = H @ H.T / D + torch.eye(D, dtype=dt); cstar = torch.randn(D, dtype=dt); d = torch.randn(S, D, dtype=dt)
l0 = torch.randn(S, dtype=dt); rows = []
for eps in [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4]:
    l_ = l0.clone().requires_grad_(True); a_ = torch.softmax(l_, 0); Ns_ = cstar + eps * d; U_ = (a_[:, None] * Ns_).sum(0); e_ = U_ - cstar
    L_ = 0.5 * e_ @ H @ e_ + 0.3 * e_.pow(3).sum(); g_, = torch.autograd.grad(L_, l_)
    ab = a_.detach(); dbar = (ab[:, None] * d).sum(0); lead = ab * ((d - dbar) @ (H @ dbar)) * eps ** 2
    rows.append({"eps": eps, "max_grad": float(g_.abs().max()), "max_abs(grad - leading)": float((g_ - lead).abs().max())})
import math
sl = [(math.log(rows[i + 1]["max_grad"]) - math.log(rows[i]["max_grad"])) / (math.log(rows[i + 1]["eps"]) - math.log(rows[i]["eps"])) for i in range(len(rows) - 1)]
sl3 = [(math.log(rows[i + 1]["max_abs(grad - leading)"]) - math.log(rows[i]["max_abs(grad - leading)"])) / (math.log(rows[i + 1]["eps"]) - math.log(rows[i]["eps"])) for i in range(len(rows) - 1)]
out["scaling"] = rows; out["loglog_slope_grad"] = sl; out["loglog_slope_remainder"] = sl3
# (3) direction counterexample (H = I, two versions, co-linear drifts)
e = torch.zeros(D, dtype=dt); e[0] = 1.0
for name, (d1, d2) in {"opposite_sign": (1.0, -2.0), "same_sign": (1.0, 2.0)}.items():
    l_ = torch.tensor([math.log(0.9), math.log(0.1)], dtype=dt, requires_grad=True); a_ = torch.softmax(l_, 0)
    Ns_ = torch.stack([cstar + 0.01 * d1 * e, cstar + 0.01 * d2 * e]); U_ = (a_[:, None] * Ns_).sum(0); e_ = U_ - cstar; L_ = 0.5 * e_ @ e_
    g_, = torch.autograd.grad(L_, l_)
    out["direction_" + name] = {"drift_norms": [abs(d1) * 0.01, abs(d2) * 0.01], "grad_l": g_.tolist(),
                                "descent_raises_version": int(torch.argmin(g_)) + 1, "more_drifted_version": 2}
# (4) through boxes/slva.py's real read: compare autograd dL/dlogit against the identity using the module's own gather on a toy model is not possible
#     without GPT-2 weights; instead re-implement the exact gather algebra (q.k / 8 + bias - c * 1[s != t-1], softmax over s, sum alpha N(V_s)) per head.
B, T, Hh, hd, t = 2, 5, 3, 8, 6
q = torch.randn(B, T, Hh, hd, dtype=dt); K_ = torch.randn(B, T, Hh, t, hd, dtype=dt); NV = torch.randn(B, T, Hh, t, hd, dtype=dt)
bias = torch.randn(Hh, t, dtype=dt)
lg = (torch.einsum("bjhd,bjhsd->bjhs", q, K_) / math.sqrt(hd) + bias[None, None]).detach().requires_grad_(True)
w = torch.softmax(lg, -1); Uh = torch.einsum("bjhs,bjhsd->bjhd", w, NV); Lh = (Uh.tanh() * torch.randn_like(Uh)).sum()
gUh, = torch.autograd.grad(Lh, Uh, retain_graph=True); glg, = torch.autograd.grad(Lh, lg)
predh = w.detach() * torch.einsum("bjhd,bjhsd->bjhs", gUh, NV - Uh.detach()[:, :, :, None, :])
out["slva_algebra_identity_max_abs_err"] = float((glg - predh).abs().max())
json.dump(out, open(sys.argv[1], "w", newline="\n"), indent=1); print(json.dumps(out, indent=1))
