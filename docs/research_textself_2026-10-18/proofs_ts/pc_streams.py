"""PROOFS_TS check P4 (CPU, no model): the rng_s coin replay and pool arithmetic of TS_draft §1.5 / §3.2.
Replays rng_s = default_rng(seed + 21800) with vbank.draw_step's consumption (one random() per step; on a coin < p_self, 64 scalar
integers(0, self_kextra + 1) draws) and compares coin AND the logged U values with each latent FT cell's draws_log.json (read only).
Also: the hypergeometric per-n consumption of 383,488 of 384,000 interleaved records. python -B pc_streams.py"""
import ctypes
import json
import math
import os
import sys

sys.dont_write_bytecode = True
try:
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x40)   # IDLE_PRIORITY_CLASS
except Exception:
    pass
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "pc_streams.json")
RUNS = "D:/Latent/boxes/runs_fsq"


def replay(seed, steps=12000, p_self=0.5, kextra=4, bs=64, vector_u=False):
    r = np.random.default_rng(seed + 21800); coins = []; Us = []
    for _ in range(steps):
        c = bool(r.random() < p_self); coins.append(c)
        if c:
            Us.append(list(r.integers(0, kextra + 1, size=bs)) if vector_u else [int(r.integers(0, kextra + 1)) for _ in range(bs)])
        else:
            Us.append(None)
    return coins, Us


def main():
    res = {"cells": {}}
    for cell in ("s215_ftself_s1", "s216_r2_s1", "s216_r1_s2"):
        a = json.load(open(os.path.join(RUNS, cell, "args.json"))); dl = json.load(open(os.path.join(RUNS, cell, "draws_log.json")))
        coins, Us = replay(a["seed"], a["steps"], a["p_self"], a["self_kextra"], a["bs"])
        steps_logged = [d["step"] for d in dl]
        lc = {d["step"]: bool(d["coin"]) for d in dl}; lu = {d["step"]: d.get("U") for d in dl if d["coin"]}
        agree_coin = sum(lc.get(s + 1) == coins[s] for s in range(a["steps"]))
        agree_u = sum(1 for s in range(a["steps"]) if coins[s] and lu.get(s + 1) == [int(x) for x in Us[s]])
        n_self = sum(coins); log_self = sum(lc.values())
        # last logged n_self_steps in log.jsonl
        nss = None
        for l_ in open(os.path.join(RUNS, cell, "log.jsonl")):
            try:
                r_ = json.loads(l_)
            except Exception:
                continue
            if "n_self_steps" in r_:
                nss = r_["n_self_steps"]
        # vector-U variant: does a size=64 call consume the stream identically?
        coins_v, Us_v = replay(a["seed"], a["steps"], a["p_self"], a["self_kextra"], a["bs"], vector_u=True)
        # coin-only variant (no U draws): how many self steps / coin agreement with the latent?
        r0 = np.random.default_rng(a["seed"] + 21800); co = [bool(r0.random() < a["p_self"]) for _ in range(a["steps"])]
        res["cells"][cell] = {"seed": a["seed"], "p_self": a["p_self"], "self_kextra": a["self_kextra"], "records": len(dl), "first_last_step": [min(steps_logged), max(steps_logged)],
                              "replay_self_steps": n_self, "logged_self_steps": log_self, "log_jsonl_last_n_self_steps": nss,
                              "coin_agree": agree_coin, "U_agree_self_steps": agree_u, "self_items": n_self * a["bs"],
                              "vector_U_same_coins": coins_v == coins, "coin_only_self_steps": sum(co), "coin_only_agree_steps": sum(x == y for x, y in zip(co, coins))}
        print(cell, res["cells"][cell], flush=True)
    # binomial law of the self-step count (for the 'equal in law' alternative, Q10)
    res["binom_sd_self_steps"] = math.sqrt(12000 * 0.25)
    # hypergeometric: consumption 383,488 of an interleaved 384,000
    Ntot, take = 384_000, 5992 * 64; hg = {}
    for name, K in (("T1 per n (96,000)", 96_000), ("T2 new n (64,000)", 64_000), ("T2 old n (32,000)", 32_000)):
        p = K / Ntot; mean = take * p; var = take * p * (1 - p) * (Ntot - take) / (Ntot - 1); hg[name] = {"mean": mean, "sd": math.sqrt(var)}
    res["hypergeom"] = hg; res["take"] = take; res["spare"] = Ntot - take
    # composition check: T2 pool 4 x 64,000 + 4 x 32,000
    res["T2_pool_total"] = 4 * 64_000 + 4 * 32_000; res["T1_pool_total"] = 4 * 96_000
    print(json.dumps({k: v for k, v in res.items() if k != "cells"}, indent=1))
    json.dump(res, open(OUT, "w", newline="\n"), indent=1)


if __name__ == "__main__":
    main()
