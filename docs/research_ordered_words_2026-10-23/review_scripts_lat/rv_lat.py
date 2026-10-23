"""REVIEW_LAT checks (CPU only, <= 2 threads, $0, no blinded file). Writes rv_lat.json next to this file.
  (A) exposure: binomial bounds; TEXT's d = 2 rise vs P0d's flat d <= 2 bins (read from the logged pilot jsons).
  (B) back-references by depth (TOPO, frames 0-5): which depths ever read a COMPUTED earlier state (S_j, j >= 1) from memory.
  (C) R2 under a per-depth model: the draft's pooled max(K'0, K'1) <= DIRECT + 0.10 vs a hop-per-pass serial model whose K'1
      (think pass + decoder pass = 2 sequential reads) solves d = 2.
  (D) R3 under a carried-fraction mixture (fraction f follows, the rest outputs own): the f at which R3 passes; LEAK-SEG false fail.
  (E) d = 3 kept share at t = 1 (the draft uses t = ceil(d/2) = 2, which is degenerate).
  (F) a FLOP cross-check of the LAT s/step projection.
    python -B docs/research_ordered_words_2026-10-23/review_scripts_lat/rv_lat.py"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "2"
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)
import boxes.ow_words as OW  # noqa: E402

SEED = 23101   # review-only seed (not a registered set)


def part_a():
    def lb(n, p):
        return n * p - 3.09 * math.sqrt(n * p * (1 - p))
    p0d = json.load(open(os.path.join(ROOT, "scratch", "gate_s226", "pilot_p0d.json")))["cells"]["P0d-FULLFT"]
    txt = json.load(open(os.path.join(ROOT, "scratch", "gate_s227", "pilot_text.json")))
    t_rise = [(r["step"], r["by_d"]["2"]["acc"]) for r in txt["decode"] if "2" in r["by_d"]]
    return {"E_48k_U1..8": {"mean": 6000, "sd": math.sqrt(48000 * 0.125 * 0.875), "lb_0.1pct": lb(48000, 0.125)},
            "E_24k_U1..8": {"mean": 3000, "lb_0.1pct": lb(24000, 0.125)},
            "E_8k_U1,2": {"mean": 4000, "lb_0.1pct": lb(8000, 0.5)},
            "E_12k_U1..4": {"mean": 3000, "lb_0.1pct": lb(12000, 0.25)},
            "band_rule_2x2508": 2 * 2508,
            "p0d_dle2_pooled_bins_5k_to_11k": [round(p0d["num_acc_1k_bins"][str(b)], 3) for b in range(5, 11)],
            "p0d_implied_d2_acc_bins (pooled*2 - d1 0.998)": [round(2 * p0d["num_acc_1k_bins"][str(b)] - 0.998, 3) for b in range(5, 11)],
            "text_d2_decode_by_step": t_rise,
            "text_d1_decode_by_step": [(r["step"], r["by_d"]["1"]["acc"]) for r in txt["decode"]]}


def part_b(n=4000):
    s = OW.Stream(SEED, frames=OW.FRAME_TRAIN, order_mode="topo"); out = {}
    for d in range(1, 9):
        mem_ref = 0; any_s = 0; muls = 0
        for _ in range(n):
            it = s.draw(d); refs = [j for op, j in zip(it["ops"], it["src_j"]) if op.endswith("_s")]
            any_s += bool(refs); mem_ref += any(j >= 1 for j in refs); muls += "mul_s" in it["ops"]
        out[str(d)] = {"share_any_back_ref": any_s / n, "share_ref_to_computed_state_j>=1": mem_ref / n, "share_with_mul_s": muls / n}
    return out


def binom(rng, n, p, size, deff=1.3):
    ne = max(1, int(n / deff)); return rng.binomial(ne, min(max(p, 0.0), 1.0), size) / ne


def part_c(reps=20000):
    rng = np.random.default_rng(SEED + 1); B = list(range(2, 9)); per = 1000
    dirr = {d: (0.18 if d == 2 else 0.11) for d in B}                          # P0d-like: d = 2 ~0.1-0.2, deeper near chance
    lat = {d: 0.85 for d in B}
    models = {
        "hop-per-pass serial (K'0 = DIRECT; K'1 solves d = 2 only: 1 think pass + the decoder pass)": ({d: dirr[d] for d in B}, {d: (0.95 if d == 2 else dirr[d]) for d in B}),
        "two-hops-per-pass serial (K'1 solves d <= 3 or 4)": ({d: dirr[d] for d in B}, {d: (0.95 if d <= 3 else (0.5 if d == 4 else dirr[d])) for d in B}),
        "ideal (K'0 = K'1 = DIRECT at every d)": ({d: dirr[d] for d in B}, {d: dirr[d] for d in B}),
    }
    out = {}
    for name, (k0, k1) in models.items():
        L = np.mean([binom(rng, per, lat[d], reps) for d in B], 0); D = np.mean([binom(rng, per, dirr[d], reps) for d in B], 0)
        K0 = np.mean([binom(rng, per, k0[d], reps) for d in B], 0); K1 = np.mean([binom(rng, per, k1[d], reps) for d in B], 0)
        K1x = np.mean([binom(rng, per, k1[d], reps) for d in B if d >= 3], 0); Dx = np.mean([binom(rng, per, dirr[d], reps) for d in B if d >= 3], 0)
        M = np.maximum(K0, K1); R2 = (L - M >= 0.30) & (M <= D + 0.10)
        Mf = np.maximum(K0, K1x); R2f = (L - Mf >= 0.30) & (np.maximum(K0 - D, K1x - Dx) <= 0.10)
        out[name] = {"E[pooled K'1]": float(K1.mean()), "E[pooled DIRECT]": float(D.mean()), "P(R2) draft (pooled B)": float(R2.mean()),
                     "P(R2) fixed (K'0 on B, K'1 on B with d >= 3, each vs DIRECT on the same depths)": float(R2f.mean())}
    out["_note"] = "B = {2..8} (n* = 1, as P0d implies); 1,000 items per depth; design effect 1.3; LAT 0.85 at every depth"
    return out


def part_d(reps=20000):
    rng = np.random.default_rng(SEED + 2); NP = 2000; F_ideal = 0.17; out = {"mixture": {}}
    for f in (0.25, 0.30, 0.35, 0.36, 0.38, 0.40, 0.45, 0.50, 0.60):
        fol = binom(rng, NP, f, reps); flo = np.maximum(binom(rng, 3 * NP, f * F_ideal, reps), binom(rng, NP, f * 0.14, reps))
        r3 = (fol - flo >= 0.30)
        out["mixture"]["f=%.2f" % f] = {"follow": f, "floor": f * F_ideal, "own_share": 1 - f, "P(R3 follow bar)": float(r3.mean()),
                                         "P(R3) + follow_U >= 0.50": float((r3 & (fol >= 0.50)).mean())}
    out["mixture_note"] = ("items: fraction f behave as an ideal carrier (output in U; matched floor 0.17 on them), the rest keep their own answer "
                           "(own is outside U and outside every floor target), so follow = f, floor = 0.17 f; R3's bar 0.83 f >= 0.30 -> f >= 0.36")
    for lk in (0.20, 0.25, 0.30, 0.40):
        fol = binom(rng, NP, 0.75, reps); flo = binom(rng, 3 * NP, 0.17, reps); leak = binom(rng, NP, lk, reps)
        out["leakseg_false_fail_strong_carrier_leak=%.2f" % lk] = float(((fol - flo >= 0.30) & (leak <= 0.25)).mean())
    return out


def cf(rec, don, t, mode):
    S = list(don["values"][:t + 1]) if mode == "a" else list(rec["values"][:t]) + [don["values"][t]]
    for k in range(t + 1, rec["d"] + 1):
        op, b, j = rec["ops"][k - 1], rec["operands"][k - 1], rec["src_j"][k - 1]
        S.append(OW.apply(op, S[k - 1], S[j] if op.endswith("_s") else b))
    return S[-1]


def part_e(n=3000):
    r = np.random.default_rng(SEED + 3); s = OW.Stream(SEED + 4, frames=OW.FRAME_TRAIN, order_mode="topo"); out = {}
    for t in (1, 2):
        kept = 0; tried = 0
        while tried < n:
            rec = s.draw(3); don = OW.redraw(r, rec)
            if don is None or don["sig"] == rec["sig"]:
                continue
            tried += 1; own, dfin, ca, cb = rec["answer"], don["answer"], cf(rec, don, t, "a"), cf(rec, don, t, "b")
            kept += len({own, dfin, ca}) == 3 and cb not in (own, dfin)
        out["d3_t%d" % t] = {"kept": kept, "of": tried, "share": kept / tried}
    return out


def part_f():
    # forward+backward ~ 6 x non-embedding params x tokens; GPT-2 small non-embedding ~85M, half per 6-block half
    Lq = {1: 36.2, 2: 45.9, 3: 57.6, 4: 72.1, 5: 85.0, 6: 98.8, 7: 111.7, 8: 126.9}      # lat_checks B think lengths
    aux = {d: 10.0 * d for d in Lq}; P, H = 85e6, 42.5e6; K = 8; bs = 64
    def tp(cache):
        v = []
        for d, L in Lq.items():
            think = (L * H + K * L * H) if cache else K * L * P
            v.append(think + (L + 4) * P + (aux[d] + 4 * d) * P)
        return float(np.mean(v))
    nest_L, nest_aux = 29.1, 56.0                                                      # §184: batch-max think 29, mean aux ~56 tokens
    nest = K * nest_L * P + nest_L * P + nest_aux * P
    s184_s2, s184_s1 = 35033 / 48000, 0.889
    out = {"tokparam_nested_uncached": nest, "tokparam_ow_uncached": tp(False), "tokparam_ow_cached": tp(True)}
    for k in ("uncached", "cached"):
        rr = out["tokparam_ow_" + k] / nest
        out["ratio_" + k] = rr; out["s_per_step_from_s2_" + k] = s184_s2 * rr; out["s_per_step_from_s1_" + k] = s184_s1 * rr
    out["effective_TFLOPs_implied_by_s184_s2"] = 6 * nest * bs / s184_s2 / 1e12
    out["LAT_48k_hours_cached_s1_host"] = 48000 * out["s_per_step_from_s1_cached"] / 3600
    return out


def main():
    out = {"A_exposure": part_a(), "B_back_refs": part_b(), "C_R2_k1": part_c(), "D_R3_mixture_leak": part_d(), "E_d3_t1": part_e(), "F_flops": part_f()}
    with open(os.path.join(HERE, "rv_lat.json"), "w", newline="\n") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
