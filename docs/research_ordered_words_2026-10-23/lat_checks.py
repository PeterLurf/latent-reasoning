"""LAT_v1_draft $0 checks (CPU only, <= 2 threads, no GPU, no blinded file). Writes lat_checks.json next to this file.
  (A) exposure: reconstructs the per-d step exposure of §227 TEXT and §226 P0d from codi_ps's step-draw law (rng = default_rng(seed + 22331),
      d = rng.integers(1, d_max + 1) per step; DIRECT / TEXT cells draw no K) and their logged stage boundaries; validated against the logged
      exposure_by_d totals. Then: TEXT's per-d exposure at each decode record, P0d's at stop, and the planned uniform-d schedules.
  (B) lengths + cost: OW TOPO (frames 0-5) think / decoder / aux token lengths by d vs §170 nested (n ~ U{1..8}, mixed batches padded to the
      batch max) -> the projected INJECT-OW s/step from s184_reweight_s2's measured 0.730 s/step (3090) under a linear-in-tokens cost model.
  (C) pairs (TOPO, frames 0-5, new seeds): kept share under the union rule, cf_b != cf(a), exact length, the last-6-token LEAK-SEG vacuity
      check, the matched redraw floor and the within-d permutation floor for ideal carriers.
  (D) shortcut panel on TOPO frames 0-5.
  (E) OC: readings R1 / R2 / R3 and the label C1-LANG under mechanism nulls.
    python -B docs/research_ordered_words_2026-10-23/lat_checks.py"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "2"
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402
import tiktoken  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
import boxes.ow_words as OW  # noqa: E402

ENC = tiktoken.get_encoding("gpt2")
SEED_TEST2, SEED_PAIRS2, SEED_FLOOR2, SEED_PERM2 = 22960, 22970, 22975, 22980     # unused in boxes/*.py, theory/*.py (grep 2026-10-23)


# ------------------------------------------------------------------ (A) exposure
def replay(seed, stage_ends, last, ds=(1, 2, 3, 4, 5, 6, 7, 8)):
    """stage_ends: [(step_of_advance, from_dmax, to_dmax)]; returns cum[d][step] (steps with depth d among steps 1..step)."""
    rng = np.random.default_rng(seed + 22331); dmax = 1; adv = {s: to for s, _, to in stage_ends}; seq = np.zeros(last + 1, int)
    for s in range(1, last + 1):
        seq[s] = int(rng.integers(1, dmax + 1))
        if s in adv:
            dmax = adv[s]
    cum = {d: np.cumsum(seq == d) for d in ds}
    return cum


def part_a():
    out = {}
    p0d = json.load(open(os.path.join(ROOT, "scratch", "gate_s226", "pilot_p0d.json")))["cells"]["P0d-FULLFT"]
    txt = json.load(open(os.path.join(ROOT, "scratch", "gate_s227", "pilot_text.json")))
    st_p = [(r["step"], r["from"], r["to"]) for r in p0d["stages"]]; st_t = [(r["step"], r["from"], r["to"]) for r in txt["stages"]]
    cp = replay(1, st_p, p0d["last_step"]); ct = replay(1, st_t, txt["last_step"])
    out["validation"] = {
        "p0d_logged": p0d["exposure_by_d"]["steps"], "p0d_replayed": {str(d): int(cp[d][-1]) for d in (1, 2, 3) },
        "text_logged": txt["exposure_by_d"]["steps"], "text_replayed": {str(d): int(ct[d][-1]) for d in (1, 2, 3, 4, 5)}}
    ok = all(int(v) == int(cp[int(d)][-1]) for d, v in p0d["exposure_by_d"]["steps"].items()) and \
        all(int(v) == int(ct[int(d)][-1]) for d, v in txt["exposure_by_d"]["steps"].items())
    out["validation"]["replay_matches_logged"] = bool(ok)
    rows = []
    for r in txt["decode"]:
        s = r["step"]; rows.append({"step": s, "d_max": r["d_max"], "acc": {d: v["acc"] for d, v in r["by_d"].items()},
                                     "exposure": {d: int(ct[int(d)][s]) for d in r["by_d"] if int(d) <= 8}})
    out["text_decode_vs_exposure"] = rows
    first = {}
    for d in range(2, 6):
        for thr in (0.80, 0.95):
            hit = [x for x in rows if x["acc"].get(str(d), 0) >= thr]
            first["d%d_ge%.2f" % (d, thr)] = {"step": hit[0]["step"], "E_d": hit[0]["exposure"][str(d)]} if hit else None
    out["text_first_reach"] = first
    intro = {d: next((s for s, _, to in st_t if to == d), 0) for d in range(2, 7)}
    out["text_global_steps_past_intro_to_ge0.95"] = {str(d): (first["d%d_ge0.95" % d]["step"] - intro[d]) if first.get("d%d_ge0.95" % d) else None for d in range(2, 6)}
    out["p0d_exposure_at_stop"] = {str(d): int(cp[d][-1]) for d in (1, 2, 3)}
    # planned schedules: uniform d ~ U{1..8} for 48k (cloud) and U{1,2} for 8k (pilot); binomial 0.1% lower bound
    def lb(n, p):
        return int(n * p - 3.09 * math.sqrt(n * p * (1 - p)))
    out["planned"] = {"cloud_U1..8_48k": {"E_mean": 48000 / 8, "E_lb_0.1pct": lb(48000, 1 / 8)},
                      "pilot_U1,2_8k": {"E_mean": 8000 / 2, "E_lb_0.1pct": lb(8000, 0.5)},
                      "cloud_U1..8_at_16k_futility_check": {"E_mean": 16000 / 8, "E_lb_0.1pct": lb(16000, 1 / 8)}}
    return out


# ------------------------------------------------------------------ (B) lengths and cost
def part_b():
    import boxes.nested170 as N7
    rng = np.random.default_rng(1); out = {"ow": {}, "nested": {}}
    sfx = ENC.encode(" The answer is:"); aux_pre = 4
    ow_W = {}
    for d in range(1, 13):
        its = [OW.make(rng, d, order_mode="topo") for _ in range(400)]
        Lq = np.array([1 + len(ENC.encode(it["text"])) for it in its])                       # [EOT] + question
        La = np.array([sum(aux_pre + len(ENC.encode(" " + r)) + 1 for r in it["R_step"]) for it in its])
        Lc = np.array([len(ENC.encode(" " + it["cot"] + " The answer is %d." % it["answer"])) + 1 for it in its])
        out["ow"][str(d)] = {"think_len_mean": float(Lq.mean()), "think_len_max": int(Lq.max()), "aux_tokens_mean": float(La.mean()),
                             "text_target_tokens_mean": float(Lc.mean()), "dec_len_mean": float(Lq.mean() + len(sfx))}
        ow_W[d] = (Lq, La)
    # nested (§170 stream law): mixed n ~ U{1..8} per item, 64 per batch, padded to the batch max
    nl = {}; na = {}
    for n in range(1, 9):
        L = []; A = []
        for _ in range(400):
            r, t = N7.generate(rng, n); s = N7.render(r); L.append(1 + len(s) + 1)
            A.append(sum(aux_pre + len(e) + 1 for e, _ in N7.aux_steps(r)))
        nl[n] = np.array(L); na[n] = np.array(A)
        out["nested"][str(n)] = {"think_len_mean": float(np.mean(L)), "aux_tokens_mean": float(np.mean(A))}
    K = 8; bmax_L = []; bmax_A = []
    for _ in range(300):
        ns = rng.integers(1, 9, size=64); Ls = [nl[n][rng.integers(0, 400)] for n in ns]; As = [na[n][rng.integers(0, 400)] for n in ns]
        bmax_L.append(max(Ls)); bmax_A.append(np.mean(As))
    W_nest = (K + 1) * np.mean(bmax_L) + np.mean(bmax_L) + np.mean(bmax_A)               # K think + decoder (padded) + aux (mean, packed per pass)
    W_ow = np.mean([(K + 1) * ow_W[d][0].mean() + ow_W[d][0].mean() + ow_W[d][1].mean() for d in range(1, 9)])   # one d per step, no cross-d padding
    W_ow_cl = np.mean([0.5 * ow_W[d][0].mean() + (K + 1) * 0.5 * ow_W[d][0].mean() + 0.5 * ow_W[d][0].mean() + ow_W[d][1].mean() for d in range(1, 9)])
    W_ow_pilot = np.mean([(K + 1) * ow_W[d][0].mean() + ow_W[d][0].mean() + ow_W[d][1].mean() for d in (1, 2)])
    s184 = 35033 / 48000
    out["cost_model"] = {
        "def": "work = (K + 1) think passes x think length + decoder length + aux tokens (token-linear, [ASSUMED]); nested: batch-max padded think, "
               "mean aux; OW: one d per step; cache_lower: blocks 0-5 once (0.5 L) + the upper half per pass",
        "s184_s2_sec_per_step_3090": s184, "nested_batchmax_think_len": float(np.mean(bmax_L)), "W_nested": float(W_nest),
        "W_ow_U1..8": float(W_ow), "ratio": float(W_ow / W_nest), "proj_sec_per_step_3090": float(s184 * W_ow / W_nest),
        "W_ow_cache_lower": float(W_ow_cl), "proj_sec_per_step_3090_cache_lower": float(s184 * W_ow_cl / W_nest),
        "W_ow_pilot_U1,2": float(W_ow_pilot), "proj_pilot_sec_per_step_3090": float(s184 * W_ow_pilot / W_nest)}
    return out


# ------------------------------------------------------------------ (C) pairs and floors
def cf(rec, don, t, mode):
    S = list(don["values"][:t + 1]) if mode == "a" else list(rec["values"][:t]) + [don["values"][t]]
    for k in range(t + 1, rec["d"] + 1):
        op, b, j = rec["ops"][k - 1], rec["operands"][k - 1], rec["src_j"][k - 1]
        S.append(OW.apply(op, S[k - 1], S[j] if op.endswith("_s") else b))
    return S[-1]


def part_c(n_cand=3000):
    r = np.random.default_rng(SEED_PAIRS2); s = OW.Stream(SEED_PAIRS2 + 1, frames=OW.FRAME_TRAIN, order_mode="topo"); rf = np.random.default_rng(SEED_FLOOR2)
    rp = np.random.default_rng(SEED_PERM2); out = {}
    for d in range(3, 9):
        t = math.ceil(d / 2); kept = []; n_try = 0; len_mis = 0; last6_same = 0
        while n_try < n_cand:
            rec = s.draw(d); don = OW.redraw(r, rec)
            if don is None or don["sig"] == rec["sig"]:
                continue
            n_try += 1
            own, dfin, ca, cb = rec["answer"], don["answer"], cf(rec, don, t, "a"), cf(rec, don, t, "b")
            if len({own, dfin, ca}) == 3 and cb not in (own, dfin):
                ta, tb = ENC.encode(rec["text"]), ENC.encode(don["text"]); len_mis += len(ta) != len(tb); last6_same += ta[-7:] == tb[-7:]
                kept.append((rec, don, ca, cb))
        k = len(kept); row = {"t": t, "candidates": n_try, "kept": k, "kept_share": k / n_try}
        if k:
            row.update({"len_mismatch": len_mis, "share_last7_tokens_identical": last6_same / k,
                        "share_cfb_ne_cfa": float(np.mean([ca != cb for _, _, ca, cb in kept]))})
            # matched floor: 3 redraw donors on the recipient's skeleton; an ideal carrier outputs its cf for that donor
            hit_a, hit_b, hit_d2 = [], [], []
            for rec, don, ca, cb in kept[:1000]:
                U = {ca, cb}
                for _ in range(3):
                    d2 = OW.redraw(rf, rec)
                    if d2 is None:
                        continue
                    hit_a.append(cf(rec, d2, t, "a") in U); hit_b.append(cf(rec, d2, t, "b") in U); hit_d2.append(d2["answer"] in U)
            row["matched_floor_ideal_cfa_carrier"] = float(np.mean(hit_a)); row["matched_floor_ideal_cfb_carrier"] = float(np.mean(hit_b))
            row["matched_floor_relay_donor2_final"] = float(np.mean(hit_d2)); row["matched_floor_own_collapse"] = 0.0
            # permutation floor (§221 style): an ideal cf(a) carrier's outputs scored against random other pairs' unions within d
            pred = np.array([ca for _, _, ca, _ in kept[:1000]]); Us = [{ca, cb} for _, _, ca, cb in kept[:1000]]; m = len(pred); acc = []
            for _ in range(200):
                perm = rp.permutation(m); acc.append(np.mean([pred[i] in Us[perm[i]] for i in range(m) if perm[i] != i]))
            row["perm_floor_ideal_cfa_carrier"] = float(np.mean(acc)); row["union_size_mean"] = float(np.mean([len(u) for u in Us]))
        out[str(d)] = row
    return out


# ------------------------------------------------------------------ (D) shortcut panel
def part_d():
    import re
    s = OW.Stream(SEED_TEST2, frames=OW.FRAME_TRAIN, order_mode="topo"); out = {}
    for d in range(1, 9):
        its = [s.draw(d) for _ in range(2000)]; y = np.array([it["answer"] for it in its])
        nums = [[int(x) for x in re.findall(r"\b(\d+)\b", it["text"])] for it in its]
        maj = np.bincount(y, minlength=11).argmax()
        rules = {"majority": np.full(len(y), maj), "first_number": np.array([n[0] for n in nums]), "last_number": np.array([n[-1] for n in nums]),
                 "sum_mod11": np.array([sum(n) % 11 for n in nums]), "prod_mod11": np.array([int(np.prod([v % 11 for v in n])) % 11 for n in nums])}
        out[str(d)] = {k: float((v == y).mean()) for k, v in rules.items()}
        out[str(d)]["max_rule"] = max(out[str(d)].values())
    return out


# ------------------------------------------------------------------ (E) OC
def oc(reps=20000, seed=22990):
    rng = np.random.default_rng(seed); DEFF = 1.3
    def binom(n, p, size):
        ne = max(1, int(n / DEFF)); return rng.binomial(ne, min(max(p, 0), 1), size) / ne
    # B: 3 depths (min) -> 3,000 test items; pairs at d >= 4: 2 depths -> 2,000; LEAK-SEG on the same pairs
    NB, NP = 3000, 2000
    F = 0.17                                                                # measured ideal-carrier floor, (C) d 4-8: 0.13-0.19
    sc = {
        "no_effect (LAT = DIRECT 0.15; K'0/1 = K; follow = floor)": dict(lat=0.15, dirr=0.15, k01=0.15, fol=F, flo=F, leak=F),
        "strong serial, cf(a) carrier": dict(lat=0.85, dirr=0.15, k01=0.15, fol=0.75, flo=F, leak=F),
        "strong serial, cf_b carrier (back-refs re-read from text)": dict(lat=0.85, dirr=0.15, k01=0.15, fol=0.75, flo=F, leak=F, fol_a_only=0.10),
        "marginal serial (LAT 0.64, K'0/1 0.22, follow 0.52)": dict(lat=0.64, dirr=0.15, k01=0.22, fol=0.52, flo=F, leak=F),
        "at the bars (LAT 0.60, DIRECT 0.30, K'0/1 0.30, follow F + 0.30)": dict(lat=0.60, dirr=0.30, k01=0.30, fol=F + 0.30, flo=F, leak=F),
        "partial removal (K'0/1 0.45: passes help, one pass still strong)": dict(lat=0.85, dirr=0.15, k01=0.45, fol=0.75, flo=F, leak=F),
        "pause / extra compute (no item state; swap -> own)": dict(lat=0.80, dirr=0.15, k01=0.15, fol=F, flo=F, leak=F),
        "relay / precompute (swap -> donor-final; excluded from U)": dict(lat=0.80, dirr=0.15, k01=0.15, fol=F, flo=F, leak=F),
        "question-recompute scaffold (state unused, K'0/1 near K; §207 mode)": dict(lat=0.80, dirr=0.15, k01=0.70, fol=F, flo=F, leak=F),
        "weak carrier + floor collapse (follow 0.30; matched floor 0.02, perm floor F)": dict(lat=0.70, dirr=0.15, k01=0.15, fol=0.30, flo=F, flo_matched=0.02, leak=F),
        "carrier-leak (follow 0.70 but LEAK-SEG 0.40)": dict(lat=0.80, dirr=0.15, k01=0.15, fol=0.70, flo=F, leak=0.40),
    }
    res = {}
    for name, p in sc.items():
        lat = binom(NB, p["lat"], reps); dr = binom(NB, p["dirr"], reps); k0 = binom(NB, p["k01"], reps); k1 = binom(NB, p["k01"], reps)
        fol = binom(NP, p["fol"], reps); flo = np.maximum(binom(NP * 3, p.get("flo_matched", p["flo"]), reps), binom(NP, p["flo"], reps))   # max(matched, permutation)
        leak = binom(NP, p["leak"], reps)
        R1 = (lat - dr >= 0.30) & (lat >= 0.60)
        R2 = (lat - np.maximum(k0, k1) >= 0.30) & (np.maximum(k0, k1) <= dr + 0.10)
        R3 = (fol - flo >= 0.30) & (leak <= 0.25)
        row = {"P(R1)": float(R1.mean()), "P(R2)": float(R2.mean()), "P(R3)": float(R3.mean()), "P(C1-LANG)": float((R1 & R2 & R3).mean()),
               "P(DEEP-NOT-SERIAL)": float((R1 & R2 & ~R3).mean())}
        if "fol_a_only" in p:
            fa = binom(NP, p["fol_a_only"] + F, reps); row["P(R3) if scored on cf(a) only"] = float(((fa - flo >= 0.30) & (leak <= 0.25)).mean())
        if "flo_matched" in p:
            fm = binom(NP * 3, p["flo_matched"], reps); row["P(R3) with the matched floor alone"] = float(((fol - fm >= 0.30) & (leak <= 0.25)).mean())
        res[name] = row
    res["_settings"] = {"reps": reps, "design_effect": DEFF, "band_items": NB, "pairs": NP, "floor": "max(matched redraw floor x3 donors, permutation floor)",
                        "R1": "LAT - DIRECT >= 0.30 and LAT >= 0.60", "R2": "LAT(K) - max(K'0, K'1) >= 0.30 and max(K'0, K'1) <= DIRECT + 0.10",
                        "R3": "follow_U - floor >= 0.30 and LEAK-SEG <= 0.25"}
    return res


def main():
    out = {"A_exposure": part_a(), "B_lengths_cost": part_b(), "C_pairs_floors": part_c(), "D_shortcuts": part_d(), "E_oc": oc()}
    with open(os.path.join(HERE, "lat_checks.json"), "w", newline="\n") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: (v if k != "E_oc" else {n: r for n, r in v.items()}) for k, v in out.items() if k in ("E_oc",)}, indent=1))


if __name__ == "__main__":
    main()
