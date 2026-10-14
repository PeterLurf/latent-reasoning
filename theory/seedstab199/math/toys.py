"""N3 / N5 toy checks (numpy, CPU, idle): (A) Theorem 1 + Lemma 5.1 on random closed-loop maps F(S) = S P + eps G(S) + C;
(B) the log-ratio progress objective's gradient is scale-free while the margin's is kappa^(t-1)-attenuated; (C) Theorem 2: RMS-normalised integrator."""
import ctypes, json
ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x40)
import numpy as np
rng = np.random.default_rng(5); out = {}
d = 32
def make(kP):
    Q, _ = np.linalg.qr(rng.standard_normal((d, d))); P = kP * Q                     # ||P|| = kP exactly
    W = rng.standard_normal((d, d)) / np.sqrt(d); W /= np.linalg.norm(W, 2)        # ||W|| = 1 -> Lip(tanh(W .)) <= 1
    C = rng.standard_normal(d); return P, W, C
def traj(P, W, C, eps, T, S0=None):
    S = np.zeros(d) if S0 is None else S0.copy(); out_ = [S.copy()]
    for _ in range(T): S = P @ S + eps * np.tanh(W @ S) + C; out_.append(S.copy())
    return np.array(out_)
# (A) bound ||S_{t+1} - S_t|| <= (||P|| + eps)^t ||S_1 - S_0|| and path independence ||S_t - S'_t|| <= kappa^t ||S_0 - S'_0||
viol = 0; worst = 0.0
for trial in range(200):
    kP, eps = rng.uniform(0.3, 0.95), rng.uniform(0, 0.04); P, W, C = make(kP); k = kP + eps; X = traj(P, W, C, eps, 30)
    inc = np.linalg.norm(np.diff(X, axis=0), axis=1); bound = k ** np.arange(30) * inc[0]
    viol += int((inc > bound * (1 + 1e-9)).any()); worst = max(worst, float(np.max(inc / bound)))
    Y = traj(P, W, C, eps, 30, S0=rng.standard_normal(d) * 10); gap = np.linalg.norm(X - Y, axis=1)
    viol += int((gap > k ** np.arange(31) * gap[0] * (1 + 1e-9)).any())
out["A_bound_violations_of_400"] = viol; out["A_max_ratio_inc_to_bound"] = worst
# (B) d/d eps of the pass-t increment norm vs t kappa^(t-1) ||S_1-S_0||; and of log(increment ratio) (scale-free)
P, W, C = make(0.6); res = []
for eps in (0.02, 0.2):
    h = 1e-6; X0, X1 = traj(P, W, C, eps, 12), traj(P, W, C, eps + h, 12)
    i0, i1 = np.linalg.norm(np.diff(X0, axis=0), axis=1), np.linalg.norm(np.diff(X1, axis=0), axis=1)
    dinc = (i1 - i0) / h; dlogratio = (np.log(i1[1:] / i1[:-1]) - np.log(i0[1:] / i0[:-1])) / h
    res.append(dict(eps=eps, kappa_bound=0.6 + eps, d_inc_deps=[float("%.3g" % v) for v in dinc[[1, 3, 6, 10]]],
                    t_kappa_tm1_inc0=[float("%.3g" % (t * (0.6 + eps) ** (t - 1) * i0[0])) for t in (1, 3, 6, 10)],
                    d_logratio_deps=[float("%.3g" % v) for v in dlogratio[[0, 2, 5, 9]]]))
out["B_gradients"] = res
# (C) RMS-normalised integrator A_t = sum_s w_s (+ noise): share of the newest write in the normalised read, constant vs growing writes;
#     post-completion drift of the read direction after j extra writes of relative size rho
dd = 256; T = 16; sh_c, sh_g, nz = [], [], []
for _ in range(400):
    c = rng.standard_normal((T, dd)); c /= np.linalg.norm(c, axis=1, keepdims=True)       # one code per step (random, near-orthogonal)
    A = np.cumsum(c, 0); r = A / np.linalg.norm(A, axis=1, keepdims=True); sh_c.append([float(r[t] @ c[t]) for t in range(T)])
    Ag = np.zeros(dd); g = []                                                              # growing writes |w_t| = |A_{t-1}| (share ~ 1/sqrt 2)
    for t in range(T):
        w = c[t] * max(1.0, np.linalg.norm(Ag)); Ag = Ag + w; g.append(float(Ag @ c[t] / np.linalg.norm(Ag)))
    sh_g.append(g)
    xi = rng.standard_normal((T, dd)) * (0.1 / np.sqrt(dd)); E = np.cumsum(xi, 0); nz.append(np.linalg.norm(E, axis=1))
out["C_share_newest_const_writes_t1_2_4_8_16"] = [round(float(np.mean(sh_c, 0)[t - 1]), 3) for t in (1, 2, 4, 8, 16)]
out["C_share_1_over_sqrt_t"] = [round(1 / np.sqrt(t), 3) for t in (1, 2, 4, 8, 16)]
out["C_share_newest_growing_writes"] = [round(float(np.mean(sh_g, 0)[t - 1]), 3) for t in (1, 2, 4, 8, 16)]
out["C_growing_writes_state_norm_ratio_per_pass"] = float(np.sqrt(2))
out["C_noise_norm_over_sigma_t1_4_16"] = [round(float(np.mean(nz, 0)[t - 1] / 0.1), 3) for t in (1, 4, 16)]
# post-completion: A_n = sum of n unit codes, then j extra writes of norm rho in random directions -> 1 - cos(read_n, read_{n+j})
drift = {}
for n in (4, 8):
    for rho in (0.1, 0.3):
        v = []
        for _ in range(300):
            c = rng.standard_normal((n, dd)); c /= np.linalg.norm(c, axis=1, keepdims=True); A = c.sum(0); r0 = A / np.linalg.norm(A)
            e = rng.standard_normal((8, dd)); e *= rho / np.linalg.norm(e, axis=1, keepdims=True)
            v.append([1 - float((A + e[:j].sum(0)) @ r0 / np.linalg.norm(A + e[:j].sum(0))) for j in (1, 4, 8)])
        drift["n%d_rho%.1f" % (n, rho)] = [round(float(x), 4) for x in np.mean(v, 0)]
out["C_post_completion_1mcos_j1_4_8"] = drift
print(json.dumps(out, indent=1)); json.dump(out, open("toys.json", "w", newline="\n"), indent=1)
