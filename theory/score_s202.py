"""THEORY §202 scoring (WARM-START-FROM-COT; rules fixed in §202 before any code or run; Theory is the only scorer).
Cells (boxes/runs_fsq/): W gsm_lat_warm_s1 (latent, init from §200 C), C+ gsm_txt_aug_cont_s1 (TEXT-COT, init from C); comparators A gsm_lat_aug_s1, C gsm_txt_aug_s1.
eval.json fields as §200 (acc; acc_by_kprime for latent; fit {fit_fix, fit_rs}; ...). FINAL checkpoint. T = 0.028.
W-LIFT: W - A >= T WARM-HELPS / <= -T WARM-HURTS / else NEUTRAL. R1-W: W - C+ above / matches / below at +-T. LOOP-FORMED: K'10 - K'0 >= T.
THINK-NECESSARY: K'0 <= 0.5 K'10 and K'1 <= 0.5 K'10. Predictions P-WL, P-W1..P-W3, P-W6 checked beside.
    python theory/score_s202.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "boxes", "runs_fsq")
CELLS = {"W": "gsm_lat_warm_s1", "Cp": "gsm_txt_aug_cont_s1", "A": "gsm_lat_aug_s1", "C": "gsm_txt_aug_s1"}
T, EPS = 0.028, 1e-9


def load(k):
    p = os.path.join(R, CELLS[k], "eval.json")
    return json.load(open(p)) if os.path.exists(p) else None


def three(d, pos, mid, neg):
    return pos if d >= T - EPS else (neg if d <= -T + EPS else mid)


def main():
    E = {k: load(k) for k in CELLS}; out = {}; lines = []; pr = {}
    if E["W"] is None:
        print("READING: pending (W not landed)"); json.dump({"reading": "pending"}, open(os.path.join(ROOT, "theory", "results", "score_s202.json"), "w", newline=chr(10))); return
    w = E["W"]; kp = w["acc_by_kprime"]
    loop = kp["10"] - kp["0"] >= T - EPS
    tn = kp["0"] <= 0.5 * kp["10"] + EPS and kp["1"] <= 0.5 * kp["10"] + EPS
    out.update(W_acc=w["acc"], loop_formed=loop, think_necessary=tn, kprime=kp)
    lines.append("W %.3f; LOOP-FORMED %s (K'10 %.3f - K'0 %.3f); THINK-NECESSARY %s (K'1 %.3f)" % (w["acc"], loop, kp["10"], kp["0"], tn, kp["1"]))
    if E["A"]:
        lift = w["acc"] - E["A"]["acc"]; out["W_LIFT"] = three(lift, "WARM-HELPS", "NEUTRAL", "WARM-HURTS"); out["lift"] = lift
        lines.append("W-LIFT %s (W - A %+.3f; exposure differs: W 24k text + 12k latent, A 24k latent)" % (out["W_LIFT"], lift))
        pr["P-WL"] = "holds" if out["W_LIFT"] == "WARM-HELPS" else "FALSIFIED"
    if E["Cp"]:
        d = w["acc"] - E["Cp"]["acc"]; out["R1_W"] = three(d, "above", "matches", "below")
        lines.append("R1-W %s (W - C+ %+.3f; C+ %.3f; C+ - C %+.3f)" % (out["R1_W"], d, E["Cp"]["acc"], E["Cp"]["acc"] - (E["C"]["acc"] if E["C"] else float("nan"))))
        pr["P-W3"] = "holds" if 0.24 - EPS <= w["acc"] <= 0.35 + EPS and out["R1_W"] in ("matches", "below") else "FALSIFIED"
    pr["P-W1"] = "holds" if loop else "FALSIFIED"; pr["P-W2"] = "holds" if tn else "FALSIFIED"
    f = w.get("fit") or {}
    if f.get("fit_rs") is not None:
        rs = f["fit_fix"] - f["fit_rs"]; out["recall_share"] = rs; pr["P-W6"] = "holds" if rs >= 0.20 - EPS else "FALSIFIED"
    out["predictions"] = pr; out["reading"] = " | ".join(lines)
    print("READING:", out["reading"]); print("PREDICTIONS:", pr)
    json.dump(out, open(os.path.join(ROOT, "theory", "results", "score_s202.json"), "w", newline=chr(10)), indent=1, default=str)


if __name__ == "__main__":
    main()
