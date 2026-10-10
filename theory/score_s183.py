"""THEORY §183 scoring (rules fixed in §183; written before any §183 implementation or run; Theory is the only scorer).
INTERLEAVE-BLIND: §181 k1 exactly, with the per-token decoder blind ([EOT, " =", ans_<j] + memory; no problem token). §181's rules: HIT = 4x4 >= 0.9 AND median
drop_j (cols 3-8) >= 0.3 with none < -0.05 AND k'=0 <= 0.1; FAIL = 4x4 < 0.5 (open (under-trained) if train_loss_tail second < 0.98 x first); else PARTIAL.
Synthesis vs §181 k1 (the seeing control; its cols 3-7 mean read from its eval.json): SUPPORTED = HIT; PARTLY = cols 3-7 mean >= control + 0.2 AND median drop
>= 0.1; NOT SUPPORTED = cols 3-7 mean <= control + 0.05 AND median drop < 0.05; else mixed. A HIT opens s2 (SUPPORTED then needs 2 / 2).
Dir boxes/runs_fsq/s183_ilblind_k1_s{seed}.
    python theory/score_s183.py
"""
import json
import os
import statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "boxes", "runs_fsq")
BASE = {"steps": 48000, "pass_index": "sin_gain", "interleave": 1, "blind_decoder": True}
CTRL = "s181_interleave_k1_s1"
EPS = 1e-9
ERRORS = []


def load(d, f="eval.json"):
    p = os.path.join(R, d, f)
    return json.load(open(p)) if os.path.exists(p) else None


def mid(cols):
    return sum(float(c) for c in cols[2:7]) / 5 if cols and len(cols) >= 7 else None


def cell(s, ctrl_mid):
    d = "s183_ilblind_k1_s%d" % s; a = load(d, "args.json") or {}; r = load(d)
    if r is None:
        return {"status": "running" if a else "open"}
    bad = {k: a.get(k) for k, w in dict(BASE, seed=s).items() if str(a.get(k)) != str(w)}
    x = r.get("s181") or {}
    if str(x.get("k")) != "1":
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
        syn = "SUPPORTED"
    elif out["mid_3_7"] >= ctrl_mid + 0.2 - EPS and med >= 0.1 - EPS:
        syn = "PARTLY SUPPORTED"
    elif out["mid_3_7"] <= ctrl_mid + 0.05 + EPS and med < 0.05:
        syn = "NOT SUPPORTED"
    else:
        syn = "mixed"
    if syn == "NOT SUPPORTED" and rule == "open (under-trained)":           # a negative is not read while the loss still falls
        syn = "open (under-trained)"
    out["synthesis"] = syn
    out["reading"] = "%s; synthesis %s (cols 3-7 %.3f vs control %.3f; median drop %.3f; k'0 %.3f; 4x4 %.3f)" % (rule, syn, out["mid_3_7"], ctrl_mid if ctrl_mid is not None else float("nan"), med, k0, acc)
    return out


def main():
    c = load(CTRL); ctrl_mid = mid(((c or {}).get("s181") or {}).get("acc_by_column"))
    res = {"control": CTRL, "control_mid_3_7": ctrl_mid, "runs": {s: cell(s, ctrl_mid) for s in (1, 2)}}
    r1, r2 = res["runs"][1], res["runs"][2]
    if r1.get("status") != "done" or "rule" not in r1:
        rd = "pending (s1 %s)" % r1.get("reading", r1.get("status"))
    elif r1["hit"] and r2.get("status") != "done":
        rd = "s1 HIT; SUPPORTED needs s2 (opens)"
    elif r1["hit"]:
        rd = "SUPPORTED (2 / 2)" if r2.get("hit") else "s1 HIT, s2 %s: not 2 / 2" % r2.get("rule")
    else:
        rd = r1["reading"]
    res["reading"] = rd; print("READING:", rd)
    if ERRORS:
        print("ERRORS:", ERRORS)
    res["errors"] = [str(e) for e in ERRORS]
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "score_s183.json"), "w", newline=chr(10)), indent=1, default=str)


if __name__ == "__main__":
    main()
