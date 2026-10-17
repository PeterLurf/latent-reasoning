"""PROOFS_D4 check 2: the trainer's realised n law. nested170.train_items draws n ~ U{1..n_max}, then rejects items in EXCL (the section 170 test set,
1000 per n, + 2200 SERIAL n 8 items, exactly as boxes/vbank.py main builds it) and redraws BOTH n and the item. So P(n) is proportional to the
acceptance rate a(n) = P(generate(n) not in EXCL). Read-only import of boxes.nested170; CPU."""
import json, sys
sys.path.insert(0, "D:/Latent")
import numpy as np
import boxes.nested170 as N

TE = N.test_set(1000); EXCL = {tuple(it["syms"]) for it in TE}; r_sp = np.random.default_rng(17300)
for _ in range(2200):
    EXCL.add(tuple(N.render(N.generate(r_sp, 8)[0])))
rng = np.random.default_rng(424242); acc = {}; M = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
for n in range(1, 9):
    k = 0
    for _ in range(M):
        r, t = N.generate(rng, n)
        k += tuple(N.render(r)) not in EXCL
    acc[n] = k / M
Z = sum(acc.values()); pi = {n: a / Z for n, a in acc.items()}
# direct check: the trainer's own sampler (same rejection loop), 64 * 300 items
rng2 = np.random.default_rng(7); its = []
for _ in range(300):
    its += N.train_items(rng2, 64, 8, EXCL)
emp = {n: float(np.mean([it["n"] == n for it in its])) for n in range(1, 9)}
out = {"accept_rate": acc, "pi_accepted": pi, "train_items_empirical": emp, "draws_per_n": M}
json.dump(out, open(sys.argv[1], "w", newline="\n"), indent=1)
print(json.dumps(out, indent=1))
