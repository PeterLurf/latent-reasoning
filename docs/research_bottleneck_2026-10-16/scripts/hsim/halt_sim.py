# Error model for content halting (NO-OP-ALIVE) vs fixed K' oracles. Per pass while a > 0: correct (pi), stall (sig), wrong value (mu), collapse (dl).
# After done: decode residue nu per pass (false alive op -> halt delayed). Cores: C content-keyed, K count-keyed (reduces only at t <= n), E8 edge-pinned.
import numpy as np, json, sys
TMAX = 24
def run(core, n, sig, mu, dl, nu, rng, M=200000, edge=8, stalls_at=None):
    a = np.full(M, n); wrong = np.zeros(M, bool); t_done = np.full(M, -1); halt = np.full(M, -1)
    for t in range(1, TMAX + 1):
        alive = a > 0
        u = rng.random(M)
        forced_stall = stalls_at is not None and t in stalls_at
        if core == "K":
            can = alive & (t <= n)                       # clock rule: reduce only while the bank index t <= n
        else:
            can = alive.copy()
        st = can & (u < sig) | (can & forced_stall)
        wv = can & ~st & (u >= sig) & (u < sig + mu)
        cl = can & ~st & (u >= sig + mu) & (u < sig + mu + dl)
        if core == "E8" and t == edge:
            cl = cl | (can & (a > 1))                    # edge collapse: write a single value at t = edge
        red = can & ~st & ~cl
        a = np.where(cl, 0, np.where(red, a - 1, a)); wrong |= wv | (cl & (rng.random(M) < 10 / 11))
        newly = (a == 0) & (t_done < 0); t_done[newly] = t
        # halt check on the decoded state of V_t: done and no residue this pass
        res = rng.random(M) < nu
        h = (a == 0) & ~res & (halt < 0); halt[h] = t
    return a, wrong, t_done, halt
def summary(core, n, sig, mu, dl, nu, seed=0, stalls_at=None):
    rng = np.random.default_rng(seed); a, wrong, td, h = run(core, n, sig, mu, dl, nu, rng, stalls_at=stalls_at)
    shift = len(stalls_at) if stalls_at else 0
    hal = h >= 0; ok = ~wrong
    acc_halt_cens_fail = float((hal & ok).mean())         # never-halting = failure
    acc_k = lambda K: float(((td >= 0) & (td <= K) & ok).mean())
    return {"core": core, "n": n, "sig": sig, "mu": mu, "dl": dl, "nu": nu, "stalls": shift,
            "P_exact": float((h == n + shift).mean()), "P_premature": float((hal & (h < n + shift)).mean()),
            "P_late": float((h > n + shift).mean()), "P_never": float((~hal).mean()),
            "acc_halt": acc_halt_cens_fail, "acc_Kn": acc_k(n + shift), "acc_Kn2": acc_k(n + shift + 2),
            "mean_halt_event": float(h[hal].mean()) if hal.any() else None}
rows = []
for core in ("C", "K", "E8"):
    for n in (4, 8, 10, 12):
        for sig, mu, dl, nu in [(0.0, 0.0, 0.0, 0.0), (0.02, 0.01, 0.0, 0.01), (0.05, 0.01, 0.005, 0.01), (0.10, 0.02, 0.01, 0.02)]:
            rows.append(summary(core, n, sig, mu, dl, nu))
# forced single stall at ceil(n/2) (STALL test), clean otherwise except a small natural error rate
for core in ("C", "K"):
    for n in (4, 6, 8):
        rows.append(summary(core, n, 0.02, 0.01, 0.0, 0.01, stalls_at={int(np.ceil(n / 2))}))
json.dump(rows, open("halt_sim.json", "w"), indent=1)
hdr = "core n  sig   mu    dl    nu   S | exact  prem   late   never | acc_h  acc_Kn acc_Kn2 | mean_h"
print(hdr)
for r in rows:
    print("%-4s %-2d %.3f %.3f %.3f %.3f %d | %.3f  %.3f  %.3f  %.3f | %.3f  %.3f  %.3f | %s" % (r["core"], r["n"], r["sig"], r["mu"], r["dl"], r["nu"], r["stalls"],
          r["P_exact"], r["P_premature"], r["P_late"], r["P_never"], r["acc_halt"], r["acc_Kn"], r["acc_Kn2"], "%.2f" % r["mean_halt_event"] if r["mean_halt_event"] else "-"))
