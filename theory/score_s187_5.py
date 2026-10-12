"""THEORY note 187-5 / amendment 187-5a scoring (descriptive, eval-only; written before any 187-5 output exists; Theory is the only scorer).
KEEP-SINK position offsets (187-5a): on the question-carrying paths position 0 keeps wpe[0] and p >= 1 takes wpe[p + off]; the aux prompt keeps wpe[:T].
File <run>/eval_s187_posoffset_keep0.json (boxes/halt173 posoffset): by_offset {off: {n: {answer_acc_kprime_n2, handle_by_t {t: acc}}}}, n 3..8, off 0/3/6/9.
Per model, off 0 vs off 9:
  CONTROL-BROKEN = n 3-5 pooled answer acc drops by > 0.03;
  POSITION-KEYED = control holds, n 7-8 pooled drops by >= 0.10, and the n 7-8 mean handle onset (first t <= n with handle < 0.80; n + 1 if none) is earlier;
  CONTENT-KEYED  = control holds, n 7-8 pooled drops by < 0.03;  else mixed.
The reading needs both models to agree. eval_s187_posoffset.json (all-position shift) is reported as ALL-SHIFT, descriptive only (confounded with losing wpe[0]).
    python theory/score_s187_5.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "boxes", "runs_fsq")
MODELS = ("s184_reweight_s1", "s182_nofade_s1")
OFFS, CTRL, DEEP, EPS = ("0", "3", "6", "9"), (3, 4, 5), (7, 8), 1e-9


def pooled(o, ns):
    return sum(o[str(n)]["answer_acc_kprime_n2"] for n in ns) / len(ns)


def onset(o, n):
    h = o[str(n)]["handle_by_t"]
    return next((t for t in range(1, n + 1) if h[str(t)] < 0.80), n + 1)


def model(d, f="eval_s187_posoffset_keep0.json"):
    p = os.path.join(R, d, f)
    if not os.path.exists(p):
        return {"status": "open"}
    b = json.load(open(p))["by_offset"]
    assert all(k in b for k in OFFS), (d, list(b))
    o0, o9 = b["0"], b["9"]
    ctrl = {k: pooled(b[k], CTRL) for k in OFFS}; deep = {k: pooled(b[k], DEEP) for k in OFFS}
    ons = {k: sum(onset(b[k], n) for n in DEEP) / len(DEEP) for k in OFFS}
    dc, dd = ctrl["0"] - ctrl["9"], deep["0"] - deep["9"]
    if dc > 0.03 + EPS:
        lab = "CONTROL-BROKEN"
    elif dd >= 0.10 - EPS and ons["9"] < ons["0"] - EPS:
        lab = "POSITION-KEYED"
    elif dd < 0.03 - EPS:
        lab = "CONTENT-KEYED"
    else:
        lab = "mixed"
    return {"status": "done", "label": lab, "ctrl_n3_5": ctrl, "deep_n7_8": deep, "onset_n7_8": ons, "ctrl_drop_0_9": dc, "deep_drop_0_9": dd}


def main():
    res = {d: model(d) for d in MODELS}; res["all_shift"] = {d: model(d, "eval_s187_posoffset.json") for d in MODELS}
    labs = [res[d].get("label") for d in MODELS]
    if any(res[d]["status"] != "done" for d in MODELS):
        rd = "pending (%s)" % ", ".join("%s %s" % (d, res[d]["status"]) for d in MODELS)
    else:
        rd = labs[0] if labs[0] == labs[1] else "models disagree (%s / %s): no reading" % tuple(labs)
        rd += "; " + "; ".join("%s: n 3-5 %.3f -> %.3f, n 7-8 %.3f -> %.3f (off 0 -> 9), onset %.2f -> %.2f" % (
            d, res[d]["ctrl_n3_5"]["0"], res[d]["ctrl_n3_5"]["9"], res[d]["deep_n7_8"]["0"], res[d]["deep_n7_8"]["9"], res[d]["onset_n7_8"]["0"], res[d]["onset_n7_8"]["9"]) for d in MODELS)
    res["reading"] = rd; print("READING (KEEP-SINK):", rd)
    for d in MODELS:
        a = res["all_shift"][d]
        if a["status"] == "done":
            print("  ALL-SHIFT (descriptive, sink removed) %s: n 3-5 %s" % (d, {k: round(v, 3) for k, v in a["ctrl_n3_5"].items()}))
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "score_s187_5.json"), "w", newline=chr(10)), indent=1, default=str)


if __name__ == "__main__":
    main()
