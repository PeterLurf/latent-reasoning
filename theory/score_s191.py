"""THEORY §191 scoring (PROSQA-RELABEL; rules fixed in §191; written before any §191 run; Theory is the only scorer).
A = s191_stepaux_relabel_s1 (s168s2_stepaux_s1's cell + --relabel --test500_every 1000); B = s191_textcot_relabel_s1 (TEXT-COT control, same flags).
Fields: eval.json "acc" (test accuracy on p8_na_test); "train_fit_relabel" (A: acc_by_kprime {"0","1","10",..}; B: acc_cot {"acc"}); log.jsonl "test500_acc" by step.
A: NAME-MEM = test acc >= 0.65; UNDERFIT = relabelled train fit at K' 10 < 0.90; CHANCE = test < 0.60 with fit >= 0.90 -> STILL-RISING if test500 rose
>= 0.02 per 1k steps over the last 3k ((final - final-3000) / 3), else FLAT (next: a from-scratch generator); else mixed.
Descriptive: K' 0 / 1 train fit (K' 0 >= 0.90 with test at chance = trunk structure memorisation). Control fidelity: B test acc >= 0.70, else A is flagged.
    python theory/score_s191.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_s168s2 as S168  # noqa: E402

R = os.path.join(ROOT, "boxes", "runs_fsq")
EPS = 1e-9
CELLS = {"A": ("s191_stepaux_relabel_s1", dict(S168.CELLS["stepaux"], relabel=True, test500_every=1000)),
         "B": ("s191_textcot_relabel_s1", dict(S168.CELLS["textcot"], relabel=True, test500_every=1000))}
ERRORS = []


def load(d, f):
    p = os.path.join(R, d, f)
    return json.load(open(p)) if os.path.exists(p) else None


def traj(d):
    p = os.path.join(R, d, "log.jsonl")
    if not os.path.exists(p):
        return {}
    out = {}
    for line in open(p):
        if '"test500_acc"' in line:
            r = json.loads(line); out[int(r["step"])] = float(r["test500_acc"])
    return out


def read(key):
    d, want = CELLS[key]; a = load(d, "args.json") or {}; r = load(d, "eval.json")
    if r is None:
        return {"status": "running" if a else "open"}
    if r.get("eval_partial", False) is not False:
        return {"status": "eval running (partial eval.json)"}
    bad = {k: a.get(k) for k, v in dict(want, seed=1).items() if str(a.get(k)) != str(v)}
    if key == "A" and str(a.get("aux_fade")) not in S168.FADES:
        bad["aux_fade"] = a.get("aux_fade")
    if bad:
        ERRORS.append((d, bad)); return {"status": "error", "error": bad}
    tf = r.get("train_fit_relabel") or {}; fit = tf.get("acc_by_kprime") if key == "A" else (tf.get("acc_cot") or {})
    tr = traj(d)
    return {"status": "done", "acc": r.get("acc"), "fit": fit, "traj": tr, "swap_rate": tf.get("relabel_swap_rate"), "gpu": r.get("train_gpu")}


def main():
    A, B = read("A"), read("B"); res = {"A": A, "B": B}
    if A.get("status") != "done":
        rd = "pending (A %s; B %s)" % (A.get("status"), B.get("status"))
    else:
        acc, fit = A["acc"], A["fit"] or {}; f10, f0, f1 = fit.get("10"), fit.get("0"), fit.get("1")
        tr = A["traj"]; last = max(tr) if tr else None
        rise = None if not tr or (last - 3000) not in tr else (tr[last] - tr[last - 3000]) / 3
        if acc is None or f10 is None:
            lab = "incomplete (acc or train fit missing)"
        elif acc >= 0.65 - EPS:
            lab = "NAME-MEM"
        elif f10 < 0.90 - EPS:
            lab = "UNDERFIT"
        elif acc < 0.60 - EPS:
            lab = "CHANCE / " + ("STILL-RISING" if rise is not None and rise >= 0.02 - EPS else "FLAT (next: from-scratch generator)" if rise is not None else "rise unknown")
        else:
            lab = "mixed"
        rd = "%s: A test %.3f; relabelled train fit K'0/1/10 %s / %s / %s; test500 rise %s per 1k over the last 3k; swap rate %s" % (
            lab, acc if acc is not None else float("nan"), f0, f1, f10, None if rise is None else round(rise, 4), A.get("swap_rate"))
        if f0 is not None and lab.startswith("CHANCE"):
            rd += "; K'0 fit %s -> %s" % (round(f0, 3), "trunk structure memorisation" if f0 >= 0.90 - EPS else "memory path does the item work" if f10 >= 0.90 - EPS else "neither")
        if B.get("status") == "done":
            bacc = B["acc"]; ok = bacc is not None and bacc >= 0.70 - EPS
            rd += "; control B test %s (fidelity %s)%s" % (bacc, "OK" if ok else "FAILED", "" if ok else ": A's reading is flagged")
        else:
            rd += "; control B %s" % B.get("status")
    res["reading"] = rd; print("READING:", rd)
    if ERRORS:
        print("ERRORS:", ERRORS)
    res["errors"] = [str(e) for e in ERRORS]
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "score_s191.json"), "w", newline=chr(10)), indent=1, default=str)


if __name__ == "__main__":
    main()
