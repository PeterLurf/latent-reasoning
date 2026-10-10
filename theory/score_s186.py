"""THEORY §186 scoring (rules fixed in §186; written before any §186 implementation or run; Theory is the only scorer). Adapted from score_s183.
INTERLEAVE-BLIND-K4: §183 exactly with --interleave 4 (4 passes per answer token; the blind per-token decoder). §181's rules: HIT = 4x4 >= 0.9 AND median drop_j
(cols 3-8) >= 0.3 with none < -0.05 (the k'=0 conjunct is trivial under a blind reader); FAIL = 4x4 < 0.5 (open if under-trained); else PARTIAL.
Compute reading vs §183 (blind k1; its cols 3-7 mean read live): PARTLY = cols 3-7 mean >= control + 0.2 AND median drop >= 0.1; else NOT, and then the
mult interleave line closes (the §186 stop). A HIT opens s2.
Dir boxes/runs_fsq/s186_ilblind_k4_s{seed}.
    python theory/score_s186.py
"""
import json
import os
import statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "boxes", "runs_fsq")
BASE = {"steps": 48000, "pass_index": "sin_gain", "interleave": 4, "blind_decoder": True}
CTRL = "s183_ilblind_k1_s1"
EPS = 1e-9
ERRORS = []


def load(d, f="eval.json"):
    p = os.path.join(R, d, f)
    return json.load(open(p)) if os.path.exists(p) else None


def mid(cols):
    return sum(float(c) for c in cols[2:7]) / 5 if cols and len(cols) >= 7 else None


def cell(s, ctrl_mid):
    d = "s186_ilblind_k4_s%d" % s; a = load(d, "args.json") or {}; r = load(d)
    if r is None:
        return {"status": "running" if a else "open"}
    bad = {k: a.get(k) for k, w in dict(BASE, seed=s).items() if str(a.get(k)) != str(w)}
    x = r.get("s181") or {}
    if str(x.get("k")) != "4":
        bad["s181.k"] = x.get("k")
    if bad:
        ERRORS.append((d, bad)); return {"status": "error", "error": bad}
    acc = x.get("acc_4x4"); cols = x.get("acc_by_column"); sm = x.get("skip_matrix") or {}; k0 = (x.get("acc_by_kprime") or {}).get("0")
    tail = x.get("train_loss_tail") or {}
    under = None if not tail or tail.get("second_half") is None or not tail.get("first_half") else bool(tail["second_half"] < 0.98 * tail["first_half"])
    out = {"status": "done", "acc_4x4": acc, "acc_by_column": cols, "acc_by_kprime": x.get("acc_by_kprime"), "ext_4x8": x.get("ext_4x8"), "under_trained": under,
           "mid_3_7": mid(cols), "control_mid_3_7": ctrl_mid}
    drops = {}
    if cols and len(cols) >= 8:
        for j in range(3, 9):
            row = sm.get(str(j))
            drops[j] = (float(cols[j - 1]) - float(row[j - 1])) if row and len(row) >= j and row[j - 1] is not None else None
    out["drops"] = drops
    if acc is None or not drops or None in drops.values() or k0 is None or out["mid_3_7"] is None:
        out["reading"] = "pending (fields: acc %s, drops %s, k'0 %s)" % (acc, drops, k0); return out
    med = statistics.median(drops.values()); neg = [j for j, v in drops.items() if v < -0.05 - EPS]; out["median_drop"] = med
    tn = med >= 0.3 - EPS and not neg and k0 <= 0.1 + EPS
    if acc >= 0.9 - EPS and tn:
        rule = "HIT"
    elif acc >= 0.9 - EPS:
        rule = "PARTIAL (accurate, not THINK-NECESSARY)"
    elif acc < 0.5:
        rule = "open (under-trained)" if under else "FAIL"
    else:
        rule = "PARTIAL"
    out["rule"] = rule; out["hit"] = rule == "HIT"
    if ctrl_mid is None:
        syn = "pending (control %s)" % CTRL
    elif rule == "HIT":
        syn = "HIT"
    elif out["mid_3_7"] >= ctrl_mid + 0.2 - EPS and med >= 0.1 - EPS:
        syn = "PARTLY"
    elif rule == "open (under-trained)":                                   # a stop is not read while the loss still falls
        syn = "open (under-trained)"
    else:
        syn = "NOT: the mult interleave line closes (§186 stop)"
    out["synthesis"] = syn
    out["reading"] = "%s; compute reading %s (cols 3-7 %.3f vs control %.3f; median drop %.3f; k'0 %.3f; 4x4 %.3f)" % (rule, syn, out["mid_3_7"], ctrl_mid if ctrl_mid is not None else float("nan"), med, k0, acc)
    return out


def main():
    c = load(CTRL); ctrl_mid = mid(((c or {}).get("s181") or {}).get("acc_by_column"))
    res = {"control": CTRL, "control_mid_3_7": ctrl_mid, "runs": {s: cell(s, ctrl_mid) for s in (1, 2)}}
    r1, r2 = res["runs"][1], res["runs"][2]
    if r1.get("status") != "done" or "rule" not in r1:
        rd = "pending (s1 %s)" % r1.get("reading", r1.get("status"))
    elif r1["hit"] and r2.get("status") != "done":
        rd = "s1 HIT; s2 opens (the claim needs 2 / 2)"
    elif r1["hit"]:
        rd = "HIT (2 / 2)" if r2.get("hit") else "s1 HIT, s2 %s: not 2 / 2" % r2.get("rule")
    else:
        rd = r1["reading"]
    res["reading"] = rd; print("READING:", rd)
    if ERRORS:
        print("ERRORS:", ERRORS)
    res["errors"] = [str(e) for e in ERRORS]
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "score_s186.json"), "w", newline=chr(10)), indent=1, default=str)


if __name__ == "__main__":
    main()
