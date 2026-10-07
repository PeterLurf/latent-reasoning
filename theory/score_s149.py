"""THEORY §149 scoring (rules fixed in §149; written before any §149 run; Theory is the only scorer).
SOTA-PROBE: our donor SERIAL, ablation and contraction on RELEASED GPT-2 GSM8K-Aug latent reasoners, at release (eval only, no training).
Models (dirs boxes/runs_fsq/s149_{model}/eval.json): coconut (connordilgren/gpt2-gsm8k-coconut ckpt 33; 6 thoughts), simcot_coconut (internlm/SIM_COT-GPT2-Coconut
ckpt 28; 10 thoughts), simcot_codi (internlm/SIM_COT-GPT2-CODI; 6 latents), codi (zen-E/CODI-gpt2; 6 latents). Test: bench/data/gsm8k_aug/test.txt (1,319).
SANITY GATE: test accuracy within +-2 points of our bench harness (33.1 / 44.8 / 39.4 / 42.5); a model failing it is not read.
SERIAL(t) (aligned models: coconut pair (2t-1, 2t) <-> step t; simcot_coconut pair t <-> step t; simcot_codi latent t-1 <-> step t; t <= 3): donor / recipient
pairs with the SAME equation template, recipient correct unpatched, step t referenced downstream, donor's step-t value != recipient's, and the substituted
answer != the recipient's own (excluded share reported); swap the aligned thought(s) from the donor's run; expected = the recipient's chain re-evaluated with
step t := the donor's value. SERIAL(t) = donor-consistent >= 0.90 and recipient-consistent < 0.10, read only with n >= 30 pairs. BYPASS(t) = recipient-consistent
>= 0.5 (recomputed from the visible question, L1). codi: alignment probe-defined, SERIAL descriptive. RHO: thought t perturbed by eps = 0.1 x its norm (after
prj for CODI-style), later thoughts' deviation ratios, the first-3 geometric mean; EXPANDING > 1, CONTRACTIVE <= 0.28 (our bound, carried over), else MARGINAL.
Note 149-1: cells with n < 30 read UNDERPOWERED (distinct from PARTIAL; n reported per cell); only simcot_coconut / simcot_codi (code-documented alignment),
t = 1-3, feed a per-model C1 verdict (CAUSAL = every read cell SERIAL with >= 2 read; BYPASSED = >= 2 BYPASS cells; else PARTIAL); coconut / codi and all
ablation cells are descriptive.
    python theory/score_s149.py
"""
import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "boxes", "runs_fsq")
MODELS = {"coconut": 33.1, "simcot_coconut": 44.8, "simcot_codi": 39.4, "codi": 42.5}          # our bench accuracy (GSM8K_REPRO), the sanity anchor
ALIGNED = {"coconut", "simcot_coconut", "simcot_codi"}
VERDICT_MODELS = {"simcot_coconut", "simcot_codi"}                            # note 149-1: only code-documented alignments feed a C1 verdict
ERRORS = []


def load(d, f="eval.json"):
    p = os.path.join(R, d, f)
    return json.load(open(p)) if os.path.exists(p) else None


def rho_k3(rk, dk):
    use = [rk[k] for k in range(min(3, len(rk), len(dk))) if dk[k] > 1e-6 and rk[k] > 0]
    return math.exp(sum(math.log(x) for x in use) / len(use)) if use else None


def model_read(name):
    r = load("s149_%s" % name)
    if r is None:
        return None
    acc = r.get("test_acc")
    out = {"test_acc": acc, "gate": None if acc is None else bool(abs(100 * acc - MODELS[name]) <= 2.0)}
    if not out["gate"]:
        out["reading"] = "not read (sanity gate: %s vs bench %.1f)" % (acc, MODELS[name]); return out
    out["ablation"] = r.get("ablation")
    ser = {}
    for t, v in (r.get("donor") or {}).items():                           # {"1": {"same_template": {...}, "all": {...}}, ...}
        st = (v or {}).get("same_template") or {}
        n = st.get("n", 0) or 0
        d_, rc = st.get("donor_consistent"), st.get("recipient_consistent")
        if n < 30 or d_ is None:
            ser[t] = {"n": n, "verdict": "UNDERPOWERED"}                   # note 149-1: distinct from PARTIAL
        else:
            ser[t] = {"n": n, "donor": d_, "recipient": rc, "excluded_share": st.get("excluded_share"),
                      "verdict": "SERIAL" if (d_ >= 0.90 and rc < 0.10) else ("BYPASS" if rc >= 0.5 else "PARTIAL"),
                      "all_pairs": (v or {}).get("all")}
    out["serial"] = ser
    if name not in ALIGNED:
        out["serial_status"] = "descriptive (probe-defined alignment)"
    cf = r.get("contraction") or {}
    rho = rho_k3(cf.get("rho_k") or [], cf.get("dist_med_k") or []) if cf else None
    out["rho"] = rho; out["rho_class"] = None if rho is None else ("EXPANDING" if rho > 1 else ("CONTRACTIVE" if rho <= 0.28 else "MARGINAL"))
    out["jacobian_top"] = cf.get("jacobian_top")
    vs = [v["verdict"] for k_, v in ser.items() if v["verdict"] != "UNDERPOWERED" and int(k_) <= 3]
    if name in VERDICT_MODELS:                                               # note 149-1: >= 2 read cells; BYPASSED needs >= 2 BYPASS cells
        out["c1_at_release"] = ("UNDERPOWERED (< 2 read cells)" if len(vs) < 2 else
                                "CAUSAL (SERIAL at every read step)" if all(x == "SERIAL" for x in vs) else
                                "BYPASSED (>= 2 BYPASS cells)" if sum(x == "BYPASS" for x in vs) >= 2 else "PARTIAL")
    else:
        out["c1_at_release"] = "descriptive only (note 149-1)"
    return out


def main():
    res = {name: model_read(name) for name in MODELS}
    for name, x in res.items():
        print(name, json.dumps(x)[:600] if x else None)
    pred = {"coconut": "EXPANDING", "codi": "CONTRACTIVE"}                  # 2607.09698's prediction, tested descriptively
    res["lyapunov_prediction"] = {k: {"predicted": v, "observed": (res.get(k) or {}).get("rho_class")} for k, v in pred.items()}
    print("2607.09698 prediction:", res["lyapunov_prediction"])
    if ERRORS:
        print("ERRORS:", ERRORS)
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "score_s149.json"), "w", newline=chr(10)), indent=1, default=float)


if __name__ == "__main__":
    main()
