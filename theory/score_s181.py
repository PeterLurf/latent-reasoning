"""THEORY §181 scoring (rules fixed in §181; written before any §181 implementation or run; Theory is the only scorer).
INTERLEAVE: §169's INJECT on §161's mix with k write passes before EVERY answer token (input [EOT, problem, " =", ans_<j]), the memory carried across tokens; arms k1 / k2; 48k.
HIT(arm) = 4x4 exact >= 0.9 AND THINK-NECESSARY per token (drop_j = col_j(full) - col_j(skip token j's passes), j = 3..8: median >= 0.3, none < -0.05) AND k' = 0 at every
token <= 0.1. FAIL = 4x4 < 0.5 ("open (under-trained)" if train_loss_tail.second_half < 0.98 x first_half). Else PARTIAL. The k1 / k2 pair is read per Audit.
Dirs boxes/runs_fsq/s181_interleave_k{1,2}_s1.
    python theory/score_s181.py
"""
import json
import os
import statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "boxes", "runs_fsq")
BASE = {"steps": 48000, "seed": 1, "pass_index": "sin_gain"}
ARMS = {"k1": 1, "k2": 2}
JS = range(3, 9)                                                            # 4x4 columns 3..8 (1-indexed, LSB first)
ERRORS = []
EPS = 1e-9


def load(d, f="eval.json"):
    p = os.path.join(R, d, f)
    return json.load(open(p)) if os.path.exists(p) else None


def arm(name):
    k = ARMS[name]; d = "s181_interleave_%s_s1" % name; a = load(d, "args.json") or {}; r = load(d)
    if r is None:
        return {"status": "running" if a else "open"}
    bad = {kk: a.get(kk) for kk, w in dict(BASE, interleave=k).items() if str(a.get(kk)) != str(w)}
    x = r.get("s181") or {}
    if x.get("k") is not None and str(x.get("k")) != str(k):
        bad["s181.k"] = x.get("k")
    if bad:
        ERRORS.append((d, bad)); return {"status": "error", "error": bad}
    acc = x.get("acc_4x4"); cols = x.get("acc_by_column"); sm = x.get("skip_matrix") or {}; k0 = (x.get("acc_by_kprime") or {}).get("0")
    tail = x.get("train_loss_tail") or {}
    under = None if not tail or tail.get("second_half") is None or not tail.get("first_half") else bool(tail["second_half"] < 0.98 * tail["first_half"])
    out = {"status": "done", "acc_4x4": acc, "acc_by_column": cols, "acc_by_kprime": x.get("acc_by_kprime"), "ext_4x8": x.get("ext_4x8"),
           "under_trained": under, "skip_matrix": sm, "mem_norm_by_token": x.get("mem_norm_by_token"), "compute": x.get("compute")}
    if acc is None:
        out["reading"] = "pending (acc_4x4 missing)"; return out
    drops = {}
    if cols and len(cols) >= 8:
        for j in JS:
            row = sm.get(str(j))
            drops[j] = (float(cols[j - 1]) - float(row[j - 1])) if row and len(row) >= j and row[j - 1] is not None else None
    out["drops"] = drops
    if acc >= 0.9:
        if not drops or None in drops.values() or k0 is None:
            out["reading"] = "pending (4x4 %.3f >= 0.9; skip drops %s, k'=0 %s)" % (acc, drops, k0); return out
        med = statistics.median(drops.values()); neg = [j for j, v in drops.items() if v < -0.05 - EPS]   # EPS: the drops are float differences
        tn = med >= 0.3 - EPS and not neg and k0 <= 0.1 + EPS
        out.update({"median_drop": med, "neg_drops": neg, "kprime0": k0, "think_necessary": tn})
        out["reading"] = ("HIT: 4x4 %.3f, median drop %.3f over columns 3-8, k'=0 %.3f" % (acc, med, k0) if tn else
                          "PARTIAL (accurate, not THINK-NECESSARY): 4x4 %.3f, median drop %.3f, negative %s, k'=0 %.3f" % (acc, med, neg, k0))
        out["hit"] = tn
    elif acc < 0.5:
        out["hit"] = False
        out["reading"] = ("open (under-trained): 4x4 %.3f, loss still falling" % acc) if under else "FAIL: 4x4 %.3f; columns %s" % (acc, cols)
    else:
        out["hit"] = False; out["reading"] = "PARTIAL: 4x4 %.3f; columns %s" % (acc, cols)
    return out


def main():
    res = {"arms": {n: arm(n) for n in ARMS}}
    h = {n: v.get("hit") for n, v in res["arms"].items()}
    done = all(v.get("status") == "done" and "pending" not in v.get("reading", "pending") and "open" not in v.get("reading", "") for v in res["arms"].values())
    if not done:
        pair = "pending"
    elif h["k1"] and h["k2"]:
        pair = "both HIT: the extra within-token pass is only compute"
    elif h["k1"]:
        pair = "k1 HIT only: the repeated within-token pass hurts"
    elif h["k2"]:
        pair = "k2 HIT only: one pass per token is too shallow"
    else:
        pair = "neither HIT: the failure is not input repetition"
    res["pair"] = pair
    for n, v in res["arms"].items():
        print(n, v.get("reading", v.get("status")))
    print("PAIR:", pair)
    if ERRORS:
        print("ERRORS:", ERRORS)
    res["errors"] = [str(e) for e in ERRORS]
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "score_s181.json"), "w", newline=chr(10)), indent=1, default=float)


if __name__ == "__main__":
    main()
