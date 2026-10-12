"""THEORY note 187-8 scoring (READ-CLAMP; descriptive, eval-only; written before any measurement; Theory is the only scorer).
File <run>/eval_s187_readclamp.json (one process; both arms on the same items / card):
  {"unclamped": {"items": [row], "indist": {"handle_mean": x, "k8_pooled": y}},
   "clamped":   {"items": [row], "indist": {...}, "frac_clamped": {...}}, ...}
  row = {"n": int, "nest_depth": int, "handle_by_t": {t: 0/1}, "clamped_t3_6": bool (clamped arm only)}, n 9..12, all test items.
187-3's computation: mean handle over t 3..6 at nesting <= 2 per n; onset = first t with handle < 0.80 (T + 1 if none).
CONTROL = |indist handle_mean| and |k8_pooled| changes each <= 0.03, else INCONCLUSIVE; BINDING = >= 5% of n 10-12 items clamped at t 3-6, else NOT-BINDING;
NORM-CHANNEL = clamped n12 >= clamped n9 - 0.05 and clamped onsets n 10-12 within one pass; NOT-NORM = |clamped n12 - unclamped n12| < 0.05; else partial.
A label needs both §184 seeds to agree. 187-8b: the label is read from eval_s187_readclamp_padfree.json; the original file is descriptive. 187-8a: a 'rescaled' arm (every read to the n <= 8 mean norm) is read only if the clamp is NOT-BINDING in both seeds;
its own CONTROL applies (INCONCLUSIVE = uninformative).
    python theory/score_s187_8.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "boxes", "runs_fsq")
MODELS = ("s184_reweight_s1", "s184_reweight_s2")
FILE = "eval_s187_readclamp_padfree.json"                                    # 187-8b: the pad-free re-run decides; the original file is descriptive
EPS = 1e-9


def table(rows):
    out = {}
    for n in (9, 10, 11, 12):
        it = [r for r in rows if r["n"] == n and r["nest_depth"] <= 2]
        if not it:
            return None
        T = max(int(t) for r in it for t in r["handle_by_t"])
        h = [sum(r["handle_by_t"][str(t)] for r in it if str(t) in r["handle_by_t"]) / max(1, sum(str(t) in r["handle_by_t"] for r in it)) for t in range(1, T + 1)]
        out[n] = {"items": len(it), "mean_t3_6": sum(h[2:6]) / 4, "onset": next((t for t, v in enumerate(h, 1) if v < 0.80), T + 1)}
    return out


def model(d, arm="clamped"):
    p = os.path.join(R, d, FILE)
    if not os.path.exists(p):
        return {"status": "open"}
    x = json.load(open(p))
    if arm not in x:
        return {"status": "no %s arm" % arm}
    u, c = table(x["unclamped"]["items"]), table(x[arm]["items"])
    iu, ic = x["unclamped"]["indist"], x[arm]["indist"]
    ctrl = abs(ic["handle_mean"] - iu["handle_mean"]) <= 0.03 + EPS and abs(ic["k8_pooled"] - iu["k8_pooled"]) <= 0.03 + EPS
    deep = [r for r in x[arm]["items"] if r["n"] >= 10]; bind = 1.0 if arm == "rescaled" else sum(bool(r.get("clamped_t3_6")) for r in deep) / max(1, len(deep))   # 187-8a: rescale always binds
    ons = [c[n]["onset"] for n in (10, 11, 12)]
    if not ctrl:
        lab = "INCONCLUSIVE"
    elif bind < 0.05 - EPS:
        lab = "NOT-BINDING"
    elif c[12]["mean_t3_6"] >= c[9]["mean_t3_6"] - 0.05 - EPS and max(ons) - min(ons) <= 1:
        lab = "NORM-CHANNEL"
    elif abs(c[12]["mean_t3_6"] - u[12]["mean_t3_6"]) < 0.05 - EPS:
        lab = "NOT-NORM"
    else:
        lab = "partial"
    return {"status": "done", "label": lab, "control": ctrl, "binding_frac": bind, "unclamped": u, "clamped": c, "indist": {"unclamped": iu, "clamped": ic}}


def main():
    res = {d: model(d) for d in MODELS}
    if any(res[d]["status"] != "done" for d in MODELS):
        rd = "pending (%s)" % ", ".join("%s %s" % (d, res[d]["status"]) for d in MODELS)
    else:
        labs = [res[d]["label"] for d in MODELS]
        rd = (labs[0] if labs[0] == labs[1] else "seeds disagree (%s / %s): no label" % tuple(labs)) + "; " + "; ".join(
            "%s: t3-6 n9..12 unclamped %s -> clamped %s, onsets %s -> %s, binding %.2f, control %s" % (
                d[-2:], [round(res[d]["unclamped"][n]["mean_t3_6"], 3) for n in (9, 10, 11, 12)], [round(res[d]["clamped"][n]["mean_t3_6"], 3) for n in (9, 10, 11, 12)],
                [res[d]["unclamped"][n]["onset"] for n in (9, 10, 11, 12)], [res[d]["clamped"][n]["onset"] for n in (9, 10, 11, 12)], res[d]["binding_frac"], res[d]["control"]) for d in MODELS)
    if all(res[d].get("label") == "NOT-BINDING" for d in MODELS):                # 187-8a: the RESCALE arm is read only then
        rr = {d: model(d, "rescaled") for d in MODELS}; res["rescaled"] = rr
        lr = [rr[d].get("label") for d in MODELS]
        rd += " | RESCALE (187-8a): " + ((lr[0] if lr[0] == lr[1] else "seeds disagree (%s / %s)" % tuple(lr)) if all(rr[d]["status"] == "done" for d in MODELS) else "missing")
    global FILE
    f0, FILE = FILE, "eval_s187_readclamp.json"                           # 187-8b: the pad-deflated original, descriptive
    orig = {d: model(d) for d in MODELS}; FILE = f0; res["pad_deflated_original"] = orig
    if all(orig[d]["status"] == "done" for d in MODELS):
        rd += " | pad-deflated original (descriptive): %s" % " / ".join(orig[d]["label"] for d in MODELS)
    res["reading"] = rd; print("READING:", rd)
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "score_s187_8.json"), "w", newline=chr(10)), indent=1, default=str)


if __name__ == "__main__":
    main()
