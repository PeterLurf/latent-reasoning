"""SEEDSTAB2 check L (numpy only): an AFFINE readout of a SUPERPOSED store of two operand codes, s_c(a, b) = M1[c, a] + M2[c, b] + m[c],
cannot compute a mod-11 reduction (Lemma L in the draft, proved by the matching-sum argument). Sanity check: fit the most general such readout
(multinomial logistic on one-hot(a) (+) one-hot(b), full batch, no regularisation, long Adam) and report its best training accuracy per op.
The proof says accuracy < 1 for + - * / (at most |classes| - 1 of the |classes| matchings can be all-correct... see draft); the controls
'first' (op = a) and a sum-decomposable table reach 1.0. Also checks that a rank-10 bilinear (Fourier) score implements + mod 11 exactly.
Output: lp_sumdecomp.json beside this file."""
import json, os
import numpy as np

P = 11
INV = {v: pow(v, P - 2, P) for v in range(1, P)}
OPS = {"+": lambda a, b: (a + b) % P, "-": lambda a, b: (a - b) % P, "*": lambda a, b: (a * b) % P, "/": lambda a, b: (a * INV[b]) % P,
       "ctrl_first": lambda a, b: a, "ctrl_max": lambda a, b: max(a, b)}


def fit(f, bvals, iters=20000, lr=0.05, seed=0):
    pairs = [(a, b) for a in range(P) for b in bvals]; y = np.array([f(a, b) for a, b in pairs])
    X = np.zeros((len(pairs), 2 * P + 1)); X[np.arange(len(pairs)), [a for a, _ in pairs]] = 1; X[np.arange(len(pairs)), [P + b for _, b in pairs]] = 1; X[:, -1] = 1
    rng = np.random.default_rng(seed); W = 0.01 * rng.standard_normal((X.shape[1], P)); m1 = np.zeros_like(W); m2 = np.zeros_like(W); best = 0.0
    for it in range(1, iters + 1):
        Z = X @ W; Z -= Z.max(1, keepdims=True); Pm = np.exp(Z); Pm /= Pm.sum(1, keepdims=True)
        acc = float((Z.argmax(1) == y).mean()); best = max(best, acc)
        G = Pm.copy(); G[np.arange(len(y)), y] -= 1; g = X.T @ G / len(y)
        m1 = 0.9 * m1 + 0.1 * g; m2 = 0.999 * m2 + 0.001 * g * g
        W -= lr * (m1 / (1 - 0.9 ** it)) / (np.sqrt(m2 / (1 - 0.999 ** it)) + 1e-8)
    return best, len(pairs)


out = {}
for name, f in OPS.items():
    bv = list(range(1, P)) if name == "/" else list(range(P))
    best, npairs = fit(f, bv); out[name] = {"best_train_acc": best, "pairs": npairs}; print("%-10s best train acc %.3f over %d pairs" % (name, best, npairs))
ok = all(int(np.argmax([sum(np.cos(2 * np.pi * k * (c - a - b) / P) for k in range(1, 6)) for c in range(P)])) == (a + b) % P for a in range(P) for b in range(P))
out["bilinear_rank10_add_exact"] = bool(ok); print("rank-10 bilinear (+ mod 11) exact:", ok)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "lp_sumdecomp.json"), "w", newline="\n"), indent=1)
