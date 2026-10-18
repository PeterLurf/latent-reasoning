"""PROOFS_TS check P3 + P6 (CPU, no model, 1 thread, Idle): context feasibility, token lengths, training shapes, eval / labelling step counts.
(a) test_set(1000, n_max 24): P = len(syms) + 2 ([EOT] expr ' ='), C = len(N.cot_ids(root)) (incl. EOT); positions fed 0..P + C - 2 -> feasible iff
    P + C <= 1025. Also P + C computed independently from the shape (each step: L_k = len(render) after a placeholder reduction).
(b) the eval loop's budget rule min(max C + 50, 1025 - P) per (prompt length, chunk of 32) batch: where the cap binds, and the largest position fed.
(c) fresh SHAPE-ONLY draws (rng 23188; never shown to a model): feasibility at n 24..28 (2,000 per n); pool-size extremes: max P + C over 96,000 n-12
    and 64,000 n-16 draws (T_max_by_n of a 384k pool), the 64 longest of them (mem_check_self shape), and the max over 384,000 n-8 draws (the base's
    48k x 64 plain items hold ~384k n-8 items) and the max L_0 there (the trained copy length).
(d) structure: C / n^2, P mean, value-token share n / C.
python -B pc_tokens.py"""
import ctypes
import json
import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"; sys.dont_write_bytecode = True
try:
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x40)
except Exception:
    pass
sys.path.insert(0, "D:/Latent")
import numpy as np  # noqa: E402

import boxes.nested170 as N  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "pc_tokens.json")


def shape_lens(root):
    """(L_0, [L_1..L_n]) from the shape only: placeholder value 7 at each reduced leaf."""
    nd = N._copy(root); L0 = len(N.render(nd)); out = []
    while nd.op is not None:
        path = N._leftmost_ready(nd); tgt = nd
        for p in path:
            tgt = getattr(tgt, p)
        tgt.op = tgt.l = tgt.r = None; tgt.v = 7; out.append(len(N.render(nd)))
    return L0, out


def PC(root):
    L0, Ls = shape_lens(root); return L0 + 2, sum(Ls) + len(Ls)          # P, C (n - 1 ' =' + EOT = n terminators)


def main():
    res = {}
    TE = N.test_set(1000, n_max=24); per = {}; batches = {}
    for n in range(1, 25):
        its = [it for it in TE if it["n"] == n]; P = np.array([len(it["syms"]) + 2 for it in its]); C = np.array([len(N.cot_ids(it["root"])) for it in its])
        P2, C2 = zip(*[PC(it["root"]) for it in its]); assert (np.array(P2) == P).all() and (np.array(C2) == C).all()
        tot = P + C; feas = tot <= 1025
        per[n] = {"P_mean": float(P.mean()), "C_mean": float(C.mean()), "C_over_n2": float(C.mean() / n ** 2), "tot_mean": float(tot.mean()), "tot_max": int(tot.max()),
                  "n_infeasible": int((~feas).sum()), "n_tot_eq_1025": int((tot == 1025).sum()), "n_tot_eq_1024": int((tot == 1024).sum()),
                  "L0_max": int(max(len(it["syms"]) for it in its)), "L0_mean": float(np.mean([len(it["syms"]) for it in its])), "value_share": float(n / C.mean())}
        # (b) eval batches: group by len(syms), chunks of 32 in test order; overflow items not generated
        by = {}
        for it, c_, t_ in zip(its, C, tot):
            if t_ <= 1025:
                by.setdefault(len(it["syms"]), []).append(int(c_))
        nb = 0; ncap = 0; maxpos = 0
        for L, cs in by.items():
            for s in range(0, len(cs), 32):
                ch = cs[s:s + 32]; Pp = L + 2; bud = min(max(ch) + 50, 1025 - Pp); nb += 1; ncap += bud == 1025 - Pp
                maxpos = max(maxpos, Pp + bud - 1)                           # the loop feeds the budget-th token too (wasted)
        batches[n] = {"batches": nb, "cap_binds": ncap, "max_position_fed_if_budget_exhausted": maxpos}
        print(n, per[n], batches[n], flush=True)
    res["test_per_n"] = per; res["eval_batches"] = batches
    rng = np.random.default_rng(23188)
    fr = {}
    for n in range(24, 29):
        t_ = np.array([sum(PC(N.generate(rng, n)[0])) for _ in range(2000)]); fr[n] = {"feasible_share": float((t_ <= 1025).mean()), "max": int(t_.max()), "mean": float(t_.mean())}
        print("fresh", n, fr[n], flush=True)
    res["fresh_feasible"] = fr
    ext = {}
    for n, k in ((8, 384_000), (12, 96_000), (16, 64_000), (13, 64_000)):
        tots = np.empty(k, np.int64); L0s = np.empty(k, np.int64)
        for i in range(k):
            r_ = N.generate(rng, n)[0]; P_, C_ = PC(r_); tots[i] = P_ + C_; L0s[i] = P_ - 2
        top = np.sort(tots)[-64:]
        ext[n] = {"draws": k, "tot_max": int(tots.max()), "tot_q999": float(np.quantile(tots, 0.999)), "tot_q99": float(np.quantile(tots, 0.99)), "top64_mean": float(top.mean()),
                  "top64_min": int(top.min()), "L0_max": int(L0s.max()), "L0_q99": float(np.quantile(L0s, 0.99)), "L0_mean": float(L0s.mean())}
        print("extreme", n, ext[n], flush=True)
        res["pool_extremes"] = ext; json.dump(res, open(OUT, "w", newline="\n"), indent=1)
    json.dump(res, open(OUT, "w", newline="\n"), indent=1)


if __name__ == "__main__":
    main()
