"""THEORY §209 scoring (CODI-AT-OUR-BUDGET; rules fixed in §209 before any code or run; Theory is the only scorer).
Cell boxes/runs_fsq/gsm_codi_4ep_s1/eval.json: {"acc", "n", "valid_hist": [[step, acc], ...], "epochs", "k_latent_ablation": {"0": acc, "6": acc}} (k ablation optional).
In-house equal-budget references (§200): A gsm_lat_aug_s1 (latent), C gsm_txt_aug_s1 (text CoT). r = (CODI4 - A) / (C - A).
ARCHITECTURE-BOUND: CODI4 >= 0.251 (r >= 0.6); BUDGET-BOUND: CODI4 <= A + T = 0.175; else MIXED. A dead / short run (epochs < 4.0 - 1e-6) -> UNREAD.
    python theory/score_s209.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "boxes", "runs_fsq")
T, EPS, ARCH, EPOCHS = 0.028, 1e-9, 0.251, 4.0


def jl(d):
    f = os.path.join(R, d, "eval.json")
    return json.load(open(f)) if os.path.exists(f) else None


def main():
    e, A, C = jl("gsm_codi_4ep_s1"), jl("gsm_lat_aug_s1"), jl("gsm_txt_aug_s1"); out = {}
    if e is None:
        out["reading"] = "pending (not landed)"
    elif (e.get("epochs") or 0) < EPOCHS - 1e-6:
        out["reading"] = "UNREAD (run ended at %.2f epochs < 4; rule b, no default label)" % (e.get("epochs") or 0)
    else:
        x, a, c = e["acc"], A["acc"], C["acc"]; r = (x - a) / (c - a)
        lab = "ARCHITECTURE-BOUND" if x >= ARCH - EPS else ("BUDGET-BOUND" if x <= a + T + EPS else "MIXED")
        vh = e.get("valid_hist") or []; rising = len(vh) >= 2 and vh[-1][1] > vh[-2][1] + EPS
        out.update(acc=x, A=a, C=c, r=r, label=lab, codi_minus_text=x - c, valid_rising_at_end=rising, k_latent_ablation=e.get("k_latent_ablation"))
        out["reading"] = "%s (CODI4 %.3f; r %.2f on the A %.3f -> C %.3f scale; vs text %+.3f; valid still rising %s)" % (lab, x, r, a, c, x - c, rising)
    print("READING:", out["reading"])
    json.dump(out, open(os.path.join(ROOT, "theory", "results", "score_s209.json"), "w", newline=chr(10)), indent=1)


if __name__ == "__main__":
    main()
