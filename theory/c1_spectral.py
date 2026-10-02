"""THEORY §69 amendment 69-2 (descriptive): spectral learning of a vector-output weighted automaton (Balle et al. 2014) on
the group word problem from the same final-answer data as theory/c1_rl.py.

Data: n problems from a run's training distribution (update index uniform, length from the curriculum as in
c1_rl.length, ops uniform over G, label = final element). Hankel blocks over prefixes P = Sigma (length-1 strings), suffix
eps, outputs Y = G: H[p, y] = empirical P(y | p), H_a[p, y] = empirical P(y | p a) (unseen strings -> 0), so only
problems of length <= 2 are used. Rank-|G| SVD H ~ U S V^T; A_a = (H V)^+ H_a V; decoder B = (H V)^+ H. Execution: state
after x_1 = row x_1 of H V, then A_{x_2} ... A_{x_N}; prediction argmax(state B). Certificate: BFS from the |G| (start
state, true element) pairs under every A_a, states deduplicated after rounding to 1e-6 of the start states' mean norm;
certified iff the reachable set closes within 20 |G| states and every reached pair decodes correctly.
    python theory/c1_spectral.py [--groups S3 S4 A5] [--ns 1000 10000 100000 1000000] [--seeds 200 201 202]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c1_group import comp_table  # noqa: E402

UPDATES = {"S3": 8000, "S4": 16000, "A5": 16000}


def lengths(rng, n, updates, curriculum):
    """Vectorised c1_rl.length over a uniform update index."""
    f = rng.integers(0, updates, n) / updates
    if curriculum == "zhang":
        return np.where(f < 0.32, 2, np.where(f < 0.48, rng.integers(1, 5, n), rng.integers(1, 9, n)))
    stage = np.minimum(3, (4 * f).astype(int))
    choice = rng.integers(0, 1 << 30, n)
    sizes = np.array([1, 2, 3, 4])[stage]
    return np.array([1, 2, 4, 8])[choice % sizes]


def fit(group, n, curriculum, seed):
    COMP, e, A = comp_table(group)
    rng = np.random.default_rng(seed)
    L = lengths(rng, n, UPDATES[group], curriculum)
    n1, n2 = int((L == 1).sum()), int((L == 2).sum())
    p1 = rng.integers(0, A, n1)
    C1 = np.zeros((A, A)); np.add.at(C1, (p1, COMP[e, p1]), 1)
    p2, a2 = rng.integers(0, A, n2), rng.integers(0, A, n2)
    C2 = np.zeros((A, A, A)); np.add.at(C2, (a2, p2, COMP[COMP[e, p2], a2]), 1)
    H = C1 / np.maximum(C1.sum(1, keepdims=True), 1)
    Ha = C2 / np.maximum(C2.sum(2, keepdims=True), 1)                       # (a, p, y)
    _, _, Vt = np.linalg.svd(H)
    V = Vt[:A].T
    HVp = np.linalg.pinv(H @ V)
    Aa = np.stack([HVp @ Ha[a] @ V for a in range(A)])                    # (A, r, r)
    B = HVp @ H
    start = H @ V                                                           # state after op b = row b
    return COMP, e, A, Aa, B, start, {"n_len1": n1, "n_len2": n2, "pairs_seen": int((C2.sum(2) > 0).sum()),
                                      "pairs_total": A * A, "len1_seen": int((C1.sum(1) > 0).sum())}


def certify(COMP, e, A, Aa, B, start):
    ref = max(float(np.linalg.norm(start, axis=1).mean()), 1e-12)
    key = lambda s, g: (tuple(np.round(s / ref, 6)), g)                  # noqa: E731
    seen, frontier, bad = set(), [], 0
    for b in range(A):
        k = key(start[b], int(COMP[e, b]))
        if k not in seen:
            seen.add(k); frontier.append((start[b], int(COMP[e, b]))); bad += int(np.argmax(start[b] @ B) != COMP[e, b])
    while frontier:
        if len(seen) > 20 * A:
            return False, bad, len(seen), "not closed"
        s, g = frontier.pop()
        for a in range(A):
            ns, ng = s @ Aa[a], int(COMP[g, a])
            k = key(ns, ng)
            if k not in seen:
                seen.add(k); frontier.append((ns, ng)); bad += int(np.argmax(ns @ B) != ng)
    return bad == 0, bad, len(seen), "closed"


def hard_acc(COMP, e, A, Aa, B, start, rng):
    acc = {}
    for N in (8, 9, 32, 128):
        ops = rng.integers(0, A, (2000, N)); st = COMP[e, ops[:, 0]]; s = start[ops[:, 0]]
        for t in range(1, N):
            s = np.einsum("bi,bij->bj", s, Aa[ops[:, t]]); st = COMP[st, ops[:, t]]
            s = s / np.maximum(np.linalg.norm(s, axis=1, keepdims=True), 1e-300) * np.linalg.norm(start, axis=1).mean()
        acc[N] = float((np.argmax(s @ B, 1) == st).mean())
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="*", default=["S3", "S4", "A5"])
    ap.add_argument("--curricula", nargs="*", default=["std", "zhang"])
    ap.add_argument("--ns", type=int, nargs="*", default=[1000, 10000, 100000, 1000000])
    ap.add_argument("--seeds", type=int, nargs="*", default=[200, 201, 202])
    ap.add_argument("--out", default=None)
    a = ap.parse_args(); res = {}
    for g in a.groups:
        for cur in a.curricula:
            for n in a.ns:
                for s in a.seeds:
                    COMP, e, A, Aa, B, start, info = fit(g, n, cur, s)
                    cert, bad, nst, how = certify(COMP, e, A, Aa, B, start)
                    acc = hard_acc(COMP, e, A, Aa, B, start, np.random.default_rng(s + 7))
                    r = dict(info, certified=bool(cert), bad_pairs=bad, reached=nst, closure=how, acc=acc)
                    res["%s_%s_n%d_s%d" % (g, cur, n, s)] = r
                    print(g, cur, "n", n, "seed", s, "certified", cert, "bad", bad, "reached", nst, how,
                          "pairs %d/%d" % (info["pairs_seen"], A * A), "acc", {k: round(v, 3) for k, v in acc.items()}, flush=True)
    if a.out:
        json.dump(res, open(a.out, "w", newline="\n"), indent=1)


if __name__ == "__main__":
    main()
