"""THEORY §200 scoring (GSM8K-AUG PILOT, Tier 1; rules fixed in §200 before any code or run; Theory is the only scorer).
Cells (boxes/runs_fsq/): A gsm_lat_aug_s1, B gsm_lat_augrs_s1 (latent); C gsm_txt_aug_s1, D gsm_txt_augrs_s1 (TEXT-COT). FINAL checkpoint everywhere.
eval.json fields (all accuracies are fractions in [0, 1] on the 1,319 GSM8K test items unless named otherwise):
  acc; acc_valid_best, valid_best_step (descriptive); acc_by_m {m: acc}; fit {fit_fix, fit_rs}; struct {gap_mstrat, seen_acc, unseen_acc};
  filler {base, k16, k32}; symbolic {main, p1, p2}; dedupe {n_flagged, acc_excl_flagged}; cost {meter_flops_per_item, s_per_item};
  latent only: acc_by_kprime {"0","1","2","4","6","8","10","12","16"}; idle {m: {n, k_m2, k10}}; path_indep {delta_cos_final, answer_agree, ...}.
Readings (thresholds in accuracy points / 100): R1 A-C, B-D above / matches / below at +-0.028; R2 per latent cell ABOVE-PCCOT-1SEED (>= 0.5228),
STAGE-2-ELIGIBLE (>= 0.4948 and R1 matches or above); R3 B-A, D-C RS-HELPS / RS-HURTS / RS-NEUTRAL (+-0.028), LATENT-SPECIFIC;
R4 THINK-NECESSARY (K'0 and K'1 <= 0.5 K'10); R6 FILLER excess (latent drop - text drop, A vs C): SIZE-KEYED (>= 0.04 at k32 and >= 0.015 at k16) /
LENGTH-ROBUST (<= 0.015 at k32) / mixed; R7 STABLE-12 (K'12 >= K'10 - 0.02), STABLE-16 (K'16 >= K'10 - 0.05);
LOOP-NOT-FORMED (latent): K'10 - K'0 < 0.028 -> that cell's R1 / R2 / R3 read LOOP-NOT-FORMED. Predictions P1, P2, P4-P8 checked beside.
    python theory/score_s200.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import basin_entry as BE  # noqa: E402

R = os.path.join(ROOT, "boxes", "runs_fsq")
CELLS = {"A": "gsm_lat_aug_s1", "B": "gsm_lat_augrs_s1", "C": "gsm_txt_aug_s1", "D": "gsm_txt_augrs_s1"}
LAT = ("A", "B")
T, EPS = 0.028, 1e-9
PCCOT, ABOVE = 0.4948, 0.5228


def load(k):
    p = os.path.join(R, CELLS[k], "eval.json")
    return json.load(open(p)) if os.path.exists(p) else None


def three(d, pos, neg):
    return pos if d >= T - EPS else (neg if d <= -T + EPS else "matches" if pos == "above" else "RS-NEUTRAL")


def loop_not_formed(e):
    kp = e["acc_by_kprime"]
    return kp["10"] - kp["0"] < T - EPS


def main():
    E = {k: load(k) for k in CELLS}; out = {"cells": {}}; lines = []
    for k in CELLS:
        if E[k] is None:
            continue
        d = CELLS[k]; ent = BE.summary(d) if os.path.exists(os.path.join(R, d, "log.jsonl")) else None
        c = {"acc": E[k]["acc"], "entry_descriptive": ent}
        if k in LAT:
            kp = E[k]["acc_by_kprime"]; c["LNF"] = loop_not_formed(E[k])
            c["R4"] = kp["0"] <= 0.5 * kp["10"] + EPS and kp["1"] <= 0.5 * kp["10"] + EPS
            c["R7"] = {"STABLE-12": kp["12"] >= kp["10"] - 0.02 - EPS, "STABLE-16": kp["16"] >= kp["10"] - 0.05 - EPS}
            c["idle"] = E[k].get("idle"); c["path_indep"] = E[k].get("path_indep")
        f = E[k].get("fit") or {}; c["recall_share"] = None if f.get("fit_rs") is None else f["fit_fix"] - f["fit_rs"]
        out["cells"][k] = c
    lnf = {k: out["cells"][k]["LNF"] for k in LAT if k in out["cells"]}
    for lat, txt in (("A", "C"), ("B", "D")):                               # R1
        if E[lat] and E[txt]:
            r = "LOOP-NOT-FORMED" if lnf[lat] else three(E[lat]["acc"] - E[txt]["acc"], "above", "below")
            out["R1_%s%s" % (lat, txt)] = r; lines.append("R1 %s-%s %s (%.3f vs %.3f)" % (lat, txt, r, E[lat]["acc"], E[txt]["acc"]))
    for lat, txt in (("A", "C"), ("B", "D")):                               # R2
        if E[lat]:
            if lnf[lat]:
                r2 = "LOOP-NOT-FORMED"
            else:
                r1 = out.get("R1_%s%s" % (lat, txt))
                r2 = ("ABOVE-PCCOT-1SEED; " if E[lat]["acc"] >= ABOVE - EPS else "") + ("STAGE-2-ELIGIBLE" if E[lat]["acc"] >= PCCOT - EPS and r1 in ("above", "matches") else "not stage-2-eligible")
            out["R2_%s" % lat] = r2; lines.append("R2 %s %s (%.3f; PCCoT 0.4948)" % (lat, r2, E[lat]["acc"]))
    if all(E[k] for k in CELLS):                                            # R3
        rl = "LOOP-NOT-FORMED" if (lnf["A"] or lnf["B"]) else three(E["B"]["acc"] - E["A"]["acc"], "RS-HELPS", "RS-HURTS")
        rt = three(E["D"]["acc"] - E["C"]["acc"], "RS-HELPS", "RS-HURTS")
        out["R3"] = {"latent": rl, "text": rt, "LATENT-SPECIFIC": rl == "RS-HELPS" and rt == "RS-NEUTRAL"}; lines.append("R3 latent %s, text %s" % (rl, rt))
    if E["A"] and E["C"] and E["A"].get("filler") and E["C"].get("filler"):  # R6
        fa, fc = E["A"]["filler"], E["C"]["filler"]
        ex = {kk: (fa["base"] - fa[kk]) - (fc["base"] - fc[kk]) for kk in ("k16", "k32")}
        r6 = "SIZE-KEYED" if ex["k32"] >= 0.04 - EPS and ex["k16"] >= 0.015 - EPS else ("LENGTH-ROBUST" if ex["k32"] <= 0.015 + EPS else "mixed")
        out["R6"] = {"reading": r6, "excess": ex}; lines.append("R6 %s (excess k16 %.3f, k32 %.3f)" % (r6, ex["k16"], ex["k32"]))
    pr = {}                                                                 # predictions (descriptive checks)
    if "A" in out["cells"]:
        rs = out["cells"]["A"]["recall_share"]; pr["P1"] = None if rs is None else ("holds" if rs <= 0.05 + EPS else "FALSIFIED" if rs > 0.10 + EPS else "between")
        pr["P5"] = "holds" if out["cells"]["A"]["R4"] else "FALSIFIED"; pr["P7"] = "holds" if all(out["cells"]["A"]["R7"].values()) else "FALSIFIED"
    if "R3" in out:
        pr["P2"] = "holds" if out["R3"]["latent"] == "RS-NEUTRAL" and out["R3"]["text"] == "RS-NEUTRAL" else ("FALSIFIED" if E["B"]["acc"] - E["A"]["acc"] >= T - EPS else "partly")
    if E["A"] and E["B"] and E["A"].get("struct") and E["B"].get("struct"):
        g = E["B"]["struct"]["gap_mstrat"] - E["A"]["struct"]["gap_mstrat"]; pr["P4"] = "holds" if abs(g) <= 0.05 + EPS else ("FALSIFIED" if g < -0.05 - EPS else "outside (B gap larger)")
    if "R6" in out:
        x = out["R6"]["excess"]["k32"]; pr["P6"] = "SIZE-KEYED as conjectured" if x >= 0.04 - EPS else ("LENGTH-ROBUST" if x <= 0.015 + EPS else "mixed band: linear transfer FALSIFIED")
    if all(E[k] and E[k].get("symbolic") for k in CELLS):
        drop = {k: E[k]["acc"] - E[k]["symbolic"]["main"] for k in CELLS}
        pr["P8"] = "holds" if all(v <= 0.03 + EPS for v in drop.values()) and drop["B"] <= drop["A"] + EPS else "FALSIFIED"; out["symbolic_drop"] = drop
    out["predictions"] = pr; out["reading"] = " | ".join(lines) or "pending"
    print("READING:", out["reading"]); print("PREDICTIONS:", pr)
    for k in LAT:
        if k in out["cells"] and out["cells"][k]["idle"]:
            print("IDLE-PASSES %s:" % k, {m: (round(v["k_m2"], 3), round(v["k10"], 3), v["n"]) for m, v in sorted(out["cells"][k]["idle"].items(), key=lambda x: int(x[0]))})
    json.dump(out, open(os.path.join(ROOT, "theory", "results", "score_s200.json"), "w", newline=chr(10)), indent=1, default=str)


if __name__ == "__main__":
    main()
