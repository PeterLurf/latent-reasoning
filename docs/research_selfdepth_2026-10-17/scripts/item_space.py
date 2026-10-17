"""SELF-DEPTH-CHAIN check 2 ($0, CPU, 1 thread): per n, fresh round items from a round seed vs the §170 exclusion set
(test_set(1000) n 1..16 + the 2,200-item SERIAL pool), within-pool duplicates, and the slot count T = len(syms) + 2 (support).
Also: the §170 test set extended to n_max 20 is identical on n <= 16 (same rng order) - checked."""
import os, sys, json
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, "D:/Latent")
import numpy as np
import boxes.nested170 as N

TE = N.test_set(1000); EXCL = {tuple(it["syms"]) for it in TE}; r_sp = np.random.default_rng(17300)
for _ in range(2200):
    EXCL.add(tuple(N.render(N.generate(r_sp, 8)[0])))
TE20 = N.test_set(1000, n_max=20)
same16 = all(a["syms"] == b["syms"] and a["t"] == b["t"] for a, b in zip(TE, TE20[:len(TE)]))
EX20 = {tuple(it["syms"]) for it in TE20[len(TE):]}
out = {"excl_size": len(EXCL), "test20_identical_on_n_le_16": same16, "by_n": {}}
NPER = int(sys.argv[2]) if len(sys.argv) > 2 else 50000
# training support (D3-T draws n ~ U{1..8} in the final stage): T over 200k draws
rng = np.random.default_rng(1); Ttr = np.array([len(N.render(N.generate(rng, int(rng.integers(1, 9)))[0])) + 2 for _ in range(200000)])
out["train_T_n_le_8"] = {"max": int(Ttr.max()), "p99": float(np.percentile(Ttr, 99)), "mean": float(Ttr.mean())}
for n in range(1, 21):
    rng = np.random.default_rng(21400 + n); seen = set(); hit = dup = hit20 = 0; T = []
    for _ in range(NPER if n >= 5 else 5000):
        r, t = N.generate(rng, n); s = tuple(N.render(r)); T.append(len(s) + 2)
        hit += s in EXCL; hit20 += s in EX20; dup += s in seen; seen.add(s)
    k = len(T); T = np.array(T)
    out["by_n"][str(n)] = {"draws": k, "excl_hits": hit, "excl_hit_rate": hit / k, "ext20_hits": hit20, "dup_rate": dup / k,
                           "T_mean": float(T.mean()), "T_p50": float(np.median(T)), "T_p99": float(np.percentile(T, 99)), "T_max": int(T.max()),
                           "share_T_gt_33": float((T > 33).mean())}
    print(n, out["by_n"][str(n)], flush=True)
json.dump(out, open(sys.argv[1], "w", newline="\n"), indent=1)
print("excl", len(EXCL), "same16", same16, "train_T", out["train_T_n_le_8"])
