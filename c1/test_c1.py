"""Unit tests for THEORY §67 (c1/core.py, c1/train.py evaluation).  python -m c1.test_c1"""
import itertools
import math

import numpy as np
import torch

from c1.core import COMP, M, PERMS, Tab, audit, hard_rollout, marginal_loss, sample, truth
from c1.train import mechanism, nulls


def perfect_tab(corrupt=False):
    t = Tab()
    with torch.no_grad():
        T = torch.full((6, M, M), -30.0); D = torch.full((M, 6), -30.0)
        for a in range(6):
            for s in range(6):
                T[a, s, COMP[a, s]] = 30.0
        for s in range(6):
            D[s, s] = 30.0
        if corrupt:
            T[2, 3] = -30.0; T[2, 3, 0 if COMP[2, 3] != 0 else 1] = 30.0
        t.T.copy_(T); t.D.copy_(D)
    return t


def main():
    # group
    assert PERMS[0] == (0, 1, 2)
    for a in range(6):
        assert COMP[a, 0] == a and COMP[0, a] == a
    for a, b, c in itertools.product(range(6), repeat=3):
        assert COMP[a, COMP[b, c]] == COMP[COMP[a, b], c]
    assert any(COMP[a, b] != COMP[b, a] for a in range(6) for b in range(6))
    assert truth([3]) == 3 and truth([1, 2]) == COMP[2, 1]
    # exact marginalisation = brute-force path sum
    torch.manual_seed(0)
    logT = torch.randn(6, M, M).log_softmax(-1); logD = torch.randn(M, 6).log_softmax(-1)
    ops = torch.tensor([[1, 4, 2], [0, 5, 5]]); y = torch.tensor([3, 1])
    for b in range(2):
        tot = 0.0
        for path in itertools.product(range(M), repeat=3):
            p, prev = 1.0, 0
            for t, c in enumerate(path):
                p *= math.exp(logT[ops[b, t], prev, c]); prev = c
            tot += p * math.exp(logD[prev, y[b]])
        lb = marginal_loss(logT, logD, ops[b:b + 1], y[b:b + 1]).item()
        assert abs(lb + math.log(tot)) < 1e-4, (lb, -math.log(tot))
    # reset uses only the last op from the initial code
    lr = marginal_loss(logT, logD, ops[:1], y[:1], reset=True).item()
    want = -torch.logsumexp(logT[ops[0, -1], 0] + logD[:, y[0]], 0).item()
    assert abs(lr - want) < 1e-5
    # perfect table: certified, exact at depth 128, nulls at chance, mechanism exact
    t = perfect_tab(); au = audit(t)
    assert au["certified"] and au["transition_errors"] == 0, au
    rng = np.random.default_rng(1); o, yy = sample(rng, 3000, 128)
    _, ans, _ = hard_rollout(t, torch.tensor(o))
    assert (ans.numpy() == yy).all()
    nl = nulls(t, "cpu", 6000)
    for N, d in nl.items():
        for k, v in d.items():
            assert v["acc"] < 0.2 or N == "8" and False, (N, k, v)
    mech = mechanism(t, "cpu", n=500)
    assert mech["donor_acc"] == 1.0 and mech["same_semantic_agreement"] == 1.0 and mech["irrelevant_change_rate"] is None, mech
    # one corrupted transition is caught
    au2 = audit(perfect_tab(corrupt=True))
    assert au2["transition_errors"] == 1 and not au2["certified"], au2
    print("C1 TESTS PASSED")


if __name__ == "__main__":
    main()
