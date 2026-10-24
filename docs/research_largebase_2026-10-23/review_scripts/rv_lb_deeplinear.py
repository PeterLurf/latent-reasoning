"""REVIEW_LB check D ($0, CPU, numpy, 1 thread): does LB_draft D2's escape time tau_i ~ ln(s_i / a_0) / (eta s_i) hold for DeltaMem's init?

Two-layer linear map y_hat = W2 W1 x, whitened x (n 32), target y = A x with A = sum_i s_i u_i v_i^T, s = (3, 1, 0.3, 0.1).
Inits:
  SAXE  small balanced (W1, W2 ~ N(0, 1e-4)): the setting the draft cites (arXiv 1312.6120)
  WO0   DeltaMem's: W1 = O(1) (pretrained-copied k / v; orthogonal rows, gain 1), W2 = 0 exactly (zero-init wo)
Optimisers: full-batch GD (eta 0.05) and Adam (lr 1e-3, betas 0.9 / 0.999, minibatch 64 so the gradient is noisy).
Reported: steps for mode i (u_i^T W2 W1 v_i) to reach 0.5 s_i; and tau_i x s_i (constant under Saxe's 1 / s law).
Usage: python -B rv_lb_deeplinear.py [--out rv_lb_deeplinear.json]
"""
import argparse
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402

N, HID, S = 32, 32, (3.0, 1.0, 0.3, 0.1)


def run(init, opt, steps=60000, seed=0):
    r = np.random.default_rng(seed); U, _ = np.linalg.qr(r.normal(size=(N, N))); V, _ = np.linalg.qr(r.normal(size=(N, N)))
    A = sum(s * np.outer(U[:, i], V[:, i]) for i, s in enumerate(S))
    if init == "SAXE":
        W1 = r.normal(scale=1e-4, size=(HID, N)); W2 = r.normal(scale=1e-4, size=(N, HID))
    else:
        Q, _ = np.linalg.qr(r.normal(size=(HID, N))); W1 = Q.copy(); W2 = np.zeros((N, HID))
    m = [np.zeros_like(W1), np.zeros_like(W2)]; v = [np.zeros_like(W1), np.zeros_like(W2)]; hit = [None] * len(S)
    for t in range(1, steps + 1):
        if opt == "GD":
            E = W2 @ W1 - A; g1, g2 = W2.T @ E, E @ W1.T                     # whitened full batch: grad of 0.5 ||W2 W1 - A||_F^2
            W1 -= 0.05 * g1; W2 -= 0.05 * g2
        else:
            X = r.normal(size=(N, 64)); E = (W2 @ W1 - A) @ X / 64; g1, g2 = W2.T @ E @ X.T, E @ (W1 @ X).T
            for j, (W, gg) in enumerate(((W1, g1), (W2, g2))):
                m[j] = 0.9 * m[j] + 0.1 * gg; v[j] = 0.999 * v[j] + 0.001 * gg ** 2
                W -= 1e-3 * (m[j] / (1 - 0.9 ** t)) / (np.sqrt(v[j] / (1 - 0.999 ** t)) + 1e-8)
        if t % 10 == 0:
            P = W2 @ W1
            for i, s in enumerate(S):
                if hit[i] is None and U[:, i] @ P @ V[:, i] >= 0.5 * s:
                    hit[i] = t
            if all(h is not None for h in hit):
                break
    return {"tau": hit, "tau_x_s": [None if h is None else h * s for h, s in zip(hit, S)]}


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "rv_lb_deeplinear.json"))
    a = ap.parse_args(argv); res = {"s": S}
    for init in ("SAXE", "WO0"):
        for opt in ("GD", "ADAM"):
            res["%s_%s" % (init, opt)] = run(init, opt)
    with open(a.out, "w", newline="\n") as f:
        json.dump(res, f, indent=1); f.write("\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
