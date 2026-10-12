"""THEORY note 187-7 scoring (SIZE-PROBE; descriptive, eval-only; written before any measurement; Theory is the only scorer).
File <run>/eval_s187_sizeprobe.json: {"by_t": {t: {"full": acc, "norm": acc, "dir": acc, "full_adddel": acc, "n_fit": int, "n_test": int}}, "chance": 0.1, ...}
(additive read r_t at the main decoder's read position after t passes; n 3..12 at nesting <= 2, balanced; 10-class logreg_acc probes).
At t = 4: PREMISE = norm >= 0.30 for NOFADE and §184 s1 and s2, and norm <= 0.15 for RMS.
SUPPORTED = RMS full <= 0.15 while §184 full >= 0.30 (both seeds); WEAKENED = RMS full >= 0.30; else mixed.
Descriptive: dir for all models, DEN vs RMS on full / norm, norm by t, norm_pre (187-7a: the pre-normalisation read).
187-7a: the primary tensor is the post-normalisation read the decoder sees, so RMS norm is at chance by construction.
    python theory/score_s187_7.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "boxes", "runs_fsq")
MODELS = {"nofade": "s182_nofade_s1", "den": "s185_den_s1", "rms": "s185_rms_s1", "s184_s1": "s184_reweight_s1", "s184_s2": "s184_reweight_s2"}
T, HI, LO, EPS = "4", 0.30, 0.15, 1e-9


def load(d):
    p = os.path.join(R, d, "eval_s187_sizeprobe.json")
    if not os.path.exists(p):
        return None
    x = json.load(open(p)); assert abs(float(x.get("chance", 0.1)) - 0.1) < 1e-9 and T in x["by_t"], (d, list(x["by_t"]))
    return x["by_t"]


def main():
    res = {k: load(d) for k, d in MODELS.items()}
    miss = [k for k, v in res.items() if v is None]
    out = {"by_model": res}
    if miss:
        rd = "pending (%s)" % ", ".join(miss)
    else:
        g = lambda k, p: float(res[k][T][p])                                 # noqa: E731
        premise = all(g(k, "norm") >= HI - EPS for k in ("nofade", "s184_s1", "s184_s2")) and g("rms", "norm") <= LO + EPS
        if g("rms", "full") <= LO + EPS and all(g(k, "full") >= HI - EPS for k in ("s184_s1", "s184_s2")):
            acc = "SUPPORTED"
        elif g("rms", "full") >= HI - EPS:
            acc = "WEAKENED"
        else:
            acc = "mixed"
        rd = "PREMISE %s; account %s at t=4: " % (premise, acc) + "; ".join(
            "%s full %.3f norm %.3f dir %.3f%s" % (k, g(k, "full"), g(k, "norm"), g(k, "dir"),
                                                 "" if res[k][T].get("norm_pre") is None else " norm_pre %.3f" % g(k, "norm_pre")) for k in MODELS)
        rd += " [187-7a: RMS norm is at chance by construction (post-norm read); norm_pre is descriptive]"
        out["premise"], out["account"] = premise, acc
        out["norm_by_t"] = {k: {t: res[k][t]["norm"] for t in sorted(res[k], key=int)} for k in MODELS}
    out["reading"] = rd; print("READING:", rd)
    json.dump(out, open(os.path.join(ROOT, "theory", "results", "score_s187_7.json"), "w", newline=chr(10)), indent=1, default=str)


if __name__ == "__main__":
    main()
