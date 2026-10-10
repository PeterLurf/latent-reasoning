"""THEORY §179 scoring (rules fixed in §179; written before any §179 implementation or run; Theory is the only scorer).
READOUT: §177 (d)'s scaffold (pair inputs per pass, K = 4n, the index, p = 0 held to 24k; b240's recipe to 12k) with a BLIND end decoder ([EOT, " ="] + memory);
reference = proto177_fadetidx12k_s1 (the decoder sees the problem; 0.0 at 12k). Read on full 4x4 exact at p = 0 at 12k and 24k (s179.acc_p0_by_step).
SHORTCUT (B) = BLIND >= 0.5 at 12k or 24k while the seeing scaffold reads 0.0 at 12k; CAPACITY (A) = BLIND < 0.1 AT 24K with column 3 <= 0.4 at 24k
(the slow-learner guard); else PARTIAL. SHORTCUT opens s2. Length vs digit errors reported.
Dirs boxes/runs_fsq/s179_blindscaffold_s{seed}.
    python theory/score_s179.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "boxes", "runs_fsq")
BASE = {"steps": 48000, "stop_at": 24000, "fade": "24000,48000", "pass_index": "sin_gain", "think_k": "4n", "blind_decoder": True}
ERRORS = []


def load(d, f="eval.json"):
    p = os.path.join(R, d, f)
    return json.load(open(p)) if os.path.exists(p) else None


def scaffold(d):
    r = load(d)
    if r is None:
        return None
    x = r.get("s177") or {}; sc = x.get("acc_scaffold_12k")
    if isinstance(sc, dict):
        sc = sc.get("acc")
    cols = x.get("acc_scaffold_12k_by_column") or ((x.get("fade_evals") or {}).get("scaffold") or {}).get("acc_by_column")
    return {"acc": sc, "by_column": cols, "gpu": r.get("train_gpu")}


def blind(d):
    r = load(d)
    if r is None:
        return None
    pb = (r.get("s179") or {}).get("acc_p0_by_step") or {}
    return {k: {"acc": (v or {}).get("acc"), "by_column": (v or {}).get("acc_by_column"), "len_err": (v or {}).get("len_err"), "digit_err": (v or {}).get("digit_err")}
            for k, v in pb.items()}


def main():
    res = {"reference_sees_problem": scaffold("proto177_fadetidx12k_s1")}
    runs = {}
    for s_ in (1, 2):
        d = "s179_blindscaffold_s%d" % s_; a = load(d, "args.json") or {}; v = blind(d)
        if v is None:
            runs[s_] = {"status": "running" if a else "open"}; continue
        bad = {k: a.get(k) for k, w in BASE.items() if str(a.get(k)) != str(w)}
        if bad:
            ERRORS.append((d, bad)); runs[s_] = {"status": "error", "error": bad}; continue
        runs[s_] = {"status": "done", "by_step": v}
    res["runs"] = runs
    ref = (res["reference_sees_problem"] or {}).get("acc"); s1 = runs[1]
    if s1.get("status") != "done":
        rd = "pending (s1 %s)" % s1.get("status")
    else:
        bs = s1["by_step"]; a12 = (bs.get("12000") or {}).get("acc"); a24 = (bs.get("24000") or {}).get("acc")
        c3_24 = ((bs.get("24000") or {}).get("by_column") or [None] * 3)[2]
        if a12 is None or a24 is None:
            rd = "pending (fields missing: 12k %s, 24k %s)" % (a12, a24)
        elif max(a12, a24) >= 0.5 and ref is not None and ref < 0.1:
            rd = "SHORTCUT (B): the blind scaffold reaches %.3f / %.3f (12k / 24k) where the seeing scaffold reads %s at 12k - the decoder's view of the problem starves the memory" % (a12, a24, ref)
        elif a24 < 0.1 and c3_24 is not None and c3_24 <= 0.4:
            rd = "CAPACITY (A): the blind scaffold also fails at 24k (%.3f; column 3 %.3f) - an end readout cannot carry the product here" % (a24, c3_24)
        else:
            rd = "PARTIAL: blind %.3f / %.3f at 12k / 24k (column 3 at 24k %s), seeing %s at 12k; columns %s" % (a12, a24, c3_24, ref, (bs.get("24000") or {}).get("by_column"))
    res["reading"] = rd; print("READING:", rd)
    if ERRORS:
        print("ERRORS:", ERRORS)
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "score_s179.json"), "w", newline=chr(10)), indent=1, default=float)


if __name__ == "__main__":
    main()
