"""R3 draft OC ($0, CPU). Every prior and noise level is [ASSUMED]; the effect sizes come from committed STAB_self values (score_s213.hist24).
Stage A (mechanism test; one FRZ cell per lineage, paired with the existing no-lever twin: s1 FT-SELF, s2 R1-s2):
  per seed  D = STAB_self(FRZ) - STAB_self(twin) = delta + e,  e ~ N(0, sqrt(2) * s_rep)  (two independent FT runs; item SE ~0.008 folded in)
  SEED-PASS = D >= 0.08  and  STAB_self(FRZ) <= STAB_base + 0.10 (two-sided: OVERSHOOT is not a pass)  and  ABSORB (Bernoulli p_abs)
  ADDR-CARRIES = SEED-PASS in both seeds.
  delta: none 0 / 0; half 0.104 / 0.055; full 0.207 / 0.109 (s1 = FT-CTRL 0.696 - FT-SELF 0.489; s2 = (base 0.684 - 0.086 FT share, ASSUMED
  equal to s1's) - R1-s2 0.489).
Stage B (round 3 proper, one lineage; R3-FRZ, with R3-PLAIN as the ablation arm): primary R3-STABLE-DEPTH = AR (ABSORB-3 and RETAIN-3) and
  K24 (K'24-STABLE and not FT-DIVERGED) and MEMH (STAB_self(R3-FRZ) >= STAB_self(R2-s1) - 0.03) and DBS (mean a, n 21-24 >= 0.80).
  Under no lever effect the round-3 STAB drop d ~ N(0.05, 0.04) (R2's own drop 0.052) [ASSUMED]; under the effect d ~ N(0, 0.02).
  DBS: M3 ~ N(mu, s_lin) with mu 0.8745 (H_shift = R2's n 17-20 mean) or 0.935 (H_prop panel, derived in R3_draft section 3), s_lin 0.06-0.12.
Usage: python -B r3_oc_sim.py  (writes r3_oc_sim.json beside this script)."""
import json
import math
import os

import numpy as np

PHI = lambda z: 0.5 * (1 + math.erf(z / math.sqrt(2)))  # noqa: E731
BASE = {"s1": 0.782, "s2": 0.684}; TWIN = {"s1": 0.489, "s2": 0.489}
DELTA = {"none": (0.0, 0.0), "half": (0.1035, 0.0545), "full": (0.207, 0.109)}
P_ABS = {"none": 0.95, "half": 0.90, "full": 0.85}
BAR, OVER = 0.08, 0.10


def seed_pass_exact(delta, s_rep, base, twin, p_abs):
    sp = math.sqrt(2) * s_rep
    lo = (BAR - delta) / sp; hi = (base + OVER - twin - delta) / sp       # D must lie in [BAR, base + OVER - twin]
    return p_abs * (PHI(hi) - PHI(lo))


def stage_a(s_rep, eff):
    p1 = seed_pass_exact(DELTA[eff][0], s_rep, BASE["s1"], TWIN["s1"], P_ABS[eff])
    p2 = seed_pass_exact(DELTA[eff][1], s_rep, BASE["s2"], TWIN["s2"], P_ABS[eff])
    return {"seed_s1": p1, "seed_s2": p2, "ADDR_CARRIES_both": p1 * p2, "ADDR_PARTIAL_exactly_one": p1 * (1 - p2) + p2 * (1 - p1)}


def stage_a_mc(s_rep, eff, n=200_000, seed=31000):
    r = np.random.default_rng(seed); ok = np.ones(n, bool)
    for i, s in enumerate(("s1", "s2")):
        D = DELTA[eff][i] + r.normal(0, math.sqrt(2) * s_rep, n); stab = TWIN[s] + D
        ok &= (D >= BAR) & (stab <= BASE[s] + OVER) & (r.random(n) < P_ABS[eff])
    return float(ok.mean())


def stage_b(eff, s_rep, s_lin, mu):
    p_ar = 0.75
    p_k24 = 0.80 if eff else 0.60
    dm, ds = (0.0, 0.02) if eff else (0.05, 0.04)
    p_memh = PHI((0.03 - dm) / math.sqrt(ds ** 2 + 2 * s_rep ** 2))       # d + noise of two STAB readings (R2's and R3's) <= 0.03
    p_dbs = 1 - PHI((0.80 - mu) / s_lin)
    return {"AR": p_ar, "K24": p_k24, "MEMH": p_memh, "DBS": p_dbs, "R3_STABLE_DEPTH": p_ar * p_k24 * p_memh * p_dbs}


out = {"def": __doc__, "stage_a": {}, "stage_b": {}}
for s_rep in (0.02, 0.04, 0.06):
    for eff in ("none", "half", "full"):
        e = stage_a(s_rep, eff); e["mc_both"] = stage_a_mc(s_rep, eff)
        out["stage_a"]["s_rep %.2f | %s" % (s_rep, eff)] = {k: round(v, 4) for k, v in e.items()}
for eff in (False, True):
    for s_lin in (0.06, 0.12):
        for mu in (0.8745, 0.935):
            out["stage_b"]["effect %s | s_lin %.2f | mu %.4f" % (eff, s_lin, mu)] = {k: round(v, 4) for k, v in stage_b(eff, 0.04, s_lin, mu).items()}
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "r3_oc_sim.json"), "w", newline="\n"), indent=1)
for sec in ("stage_a", "stage_b"):
    print("==", sec)
    for k, v in out[sec].items():
        print("  %-40s %s" % (k, v))
