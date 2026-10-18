"""SD2 draft check ($0, CPU): (1) test_set(1000, n_max 24)[:20000] == test_set(1000, n_max 20) bitwise (syms, t); and == test_set(1000) on n <= 16;
(2) T = len(syms) + 2 per n 1..24 (mean / max), the eval rollout input length; (3) duplicates within n 17..24 and vs SERIAL. Read-only on the repo."""
import os, sys, json
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, "D:/Latent")
import numpy as np
import boxes.nested170 as N
A = N.test_set(1000, 24); B = N.test_set(1000, 20); C = N.test_set(1000)
eq20 = all(a["syms"] == b["syms"] and a["t"] == b["t"] and a["n"] == b["n"] for a, b in zip(A[:20000], B)) and len(B) == 20000
eq16 = all(a["syms"] == c["syms"] and a["t"] == c["t"] for a, c in zip(A[:16000], C)) and len(C) == 16000
T = {}
for n in range(1, 25):
    L = [len(it["syms"]) + 2 for it in A if it["n"] == n]
    T[n] = {"mean": round(float(np.mean(L)), 2), "max": int(max(L)), "p99": int(np.percentile(L, 99)), "min": int(min(L))}
sp = set(); r_sp = np.random.default_rng(17300)
for _ in range(2200):
    sp.add(tuple(N.render(N.generate(r_sp, 8)[0])))
syms = [tuple(it["syms"]) for it in A]
dups = len(syms) - len(set(syms)); serial_hits = sum(s in sp for s in syms)
out = {"eq_n1_20_vs_test_set_20": eq20, "eq_n1_16_vs_test_set_16": eq16, "items": len(A), "T_by_n": T, "dups_in_test24": dups, "serial_hits_in_test24": serial_hits,
       "y_dist_n21_24": {n: np.bincount([it["t"] for it in A if it["n"] == n], minlength=11).tolist() for n in range(21, 25)}}
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "testset_ext.json"), "w", newline="\n"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "y_dist_n21_24"}))
print("majority-class share n 21-24:", {n: max(v) / 1000 for n, v in out["y_dist_n21_24"].items()})
