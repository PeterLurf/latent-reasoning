"""THEORY §205 scoring (IIT-SERIAL; registered text docs/THEORY_IITSERIAL_DRAFT_2026-10-14.md at 13d7bb87; Theory is the only scorer).
Cells boxes/runs_fsq/s205_iit_s{1,2,3}, each paired with s185_rms_s{seed}. Inputs per cell:
  log.jsonl                 - training log (entry via theory/basin_entry.py; pairing: records 100..2000 bitwise vs the control on loss / answer_ce / aux_ce)
  eval.json                 - nested170's eval: s182.path_indep {acc_from_donor, delta_cos_final}, s170.acc_by_n_kprime["1"] (g1 = mean over n 3..8)
  eval_iit.json             - the post-landing eval-only pass (local, $0): {"iit_eval": {"tau1": {"hybrid_ccc", "own", "n_ccc"}, "tau2": {...}, "tau3": {...}},
                              "siit_eval": {"keep_ccc", "n_ccc"}, "g2_pass1_deep_probe": acc (h >= 4 values from S_1, §203's g2)}
Labels (draft §7): MET = hybrid_ccc(tau1) >= 0.8; ENTERED = entry(0.5) <= 24000 and the final 1k bin < 0.1; HARMFUL = s1 or s2 not ENTERED;
ONE-PASS-COMPUTE = g1 > 0.20 or g2 > 0.21; CLEAN-CONTRACTIVE = MET and (acc_from_donor > 0.2 or delta_cos_final > 0.65).
Arm: SUPPORTED = 3/3 ENTERED-AND-MET with no HARMFUL / ONE-PASS-COMPUTE / CLEAN-CONTRACTIVE; FAILED = any HARMFUL, any MET-NOT-ENTERED, or 0/3 MET;
else MIXED. Censoring (rule b): a cell whose log is missing, ends before 48k, or never enters is NOT ENTERED (a failure); a missing eval_iit.json
leaves MET pending (the eval is $0 and re-runnable), never assumed. Event-only mean entry step printed beside.
    python theory/score_s205.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "boxes", "runs_fsq")
sys.path.insert(0, os.path.join(ROOT, "theory"))
import basin_entry as BE  # noqa: E402

SEEDS = (1, 2, 3)
EPS = 1e-9
FULL = 48000


def jl(*p):
    f = os.path.join(R, *p)
    return json.load(open(f)) if os.path.exists(f) else None


def recs(d):
    out = {}
    f = os.path.join(R, d, "log.jsonl")
    if not os.path.exists(f):
        return out
    for line in open(f):
        r = json.loads(line)
        if "step" in r:
            out[int(r["step"])] = r                                         # last occurrence wins (resumes), as basin_entry
    return out


def paired(cell, ctrl):
    a, b = recs(cell), recs(ctrl)
    steps = [s for s in range(100, 2001, 100)]
    if not all(s in a and s in b for s in steps):
        return False
    return all(a[s].get(k) == b[s].get(k) for s in steps for k in ("loss", "answer_ce", "aux_ce"))


def seed_read(s):
    cell, ctrl = "s205_iit_s%d" % s, "s185_rms_s%d" % s
    rc = recs(cell); last = max(rc) if rc else 0; ev0 = jl(cell, "eval.json")
    o = {"cell": cell, "last_step": last, "paired": paired(cell, ctrl) if rc else None, "landed": ev0 is not None}
    if ev0 is None:                                                         # not landed (running / not started): no censoring yet
        o.update(entry_05=None, last_bin=None, entered=None, met=None, label="pending (not landed)"); return o
    if last >= FULL:
        e5 = BE.entry(cell, 0.5); lb = BE.summary(cell)["last_bin"]
        o.update(entry_05=e5, last_bin=lb, entered=bool(e5 is not None and e5 <= 24000 and lb < 0.1))
    else:
        o.update(entry_05=None, last_bin=None, entered=False, censored="log ends at %d < %d (counted NOT ENTERED)" % (last, FULL) if rc else "no log (NOT ENTERED)")
    ev, ei = jl(cell, "eval.json"), jl(cell, "eval_iit.json")
    if ei is None:
        o["met"] = None
    else:
        o["hybrid_ccc_tau1"] = ei["iit_eval"]["tau1"]["hybrid_ccc"]; o["met"] = o["hybrid_ccc_tau1"] >= 0.8 - EPS
        o["n_ccc_tau1"] = ei["iit_eval"]["tau1"].get("n_ccc"); o["g2"] = ei.get("g2_pass1_deep_probe")
        o["untrained_tau"] = {k: ei["iit_eval"][k].get("hybrid_ccc") for k in ("tau2", "tau3") if k in ei["iit_eval"]}
        o["siit_keep_ccc"] = (ei.get("siit_eval") or {}).get("keep_ccc")
    if ev is not None:
        k1 = ev["s170"]["acc_by_n_kprime"]["1"]; o["g1"] = sum(k1[str(n)] for n in range(3, 9)) / 6
        pi = ev["s182"]["path_indep"]; o["acc_from_donor"], o["delta_cos"] = pi["acc_from_donor"], pi["delta_cos_final"]
    g1, g2 = o.get("g1"), o.get("g2")
    o["one_pass_compute"] = None if g1 is None or g2 is None else bool(g1 > 0.20 + EPS or g2 > 0.21 + EPS)
    o["clean_contractive"] = None if o["met"] is None or ev is None else bool(o["met"] and (o["acc_from_donor"] > 0.2 + EPS or o["delta_cos"] > 0.65 + EPS))
    if o["met"] is None:
        o["label"] = "pending (eval_iit.json)" + (" / ENTERED" if o["entered"] else " / NOT ENTERED")
    else:
        o["label"] = {(True, True): "ENTERED-AND-MET", (False, True): "MET-NOT-ENTERED", (True, False): "ENTERED-NOT-MET", (False, False): "NEITHER"}[(o["entered"], o["met"])]
    return o


def main():
    S = {s: seed_read(s) for s in SEEDS}; out = {"seeds": S}
    harmful = any(S[s]["landed"] and not S[s]["entered"] for s in (1, 2))   # landed and never entered (censored runs count, rule b)
    out["HARMFUL"] = harmful
    mets = [S[s]["met"] for s in SEEDS]
    if None in mets:
        arm = "FAILED (HARMFUL)" if harmful else "pending"
    else:
        bad = any(S[s]["one_pass_compute"] or S[s]["clean_contractive"] for s in SEEDS)
        if harmful or any(S[s]["label"] == "MET-NOT-ENTERED" for s in SEEDS) or not any(mets):
            arm = "FAILED"
        elif all(S[s]["label"] == "ENTERED-AND-MET" for s in SEEDS) and not bad:
            arm = "SUPPORTED"
        else:
            arm = "MIXED"
    out["arm"] = arm
    out["account_falsifier"] = any(S[s].get("label") == "MET-NOT-ENTERED" for s in SEEDS)
    ent = [S[s]["entry_05"] for s in SEEDS if S[s]["entry_05"] is not None]
    out["event_only_mean_entry"] = sum(ent) / len(ent) if ent else None
    out["n_entered"] = sum(1 for s in SEEDS if S[s]["entered"]); out["n_landed"] = sum(1 for s in SEEDS if S[s]["landed"])
    lines = ["s%d %s (entry %s, last bin %s, paired %s, hyb %s)" % (s, S[s]["label"], S[s]["entry_05"], None if S[s]["last_bin"] is None else round(S[s]["last_bin"], 3),
                                                                    S[s]["paired"], S[s].get("hybrid_ccc_tau1")) for s in SEEDS]
    out["reading"] = " | ".join(lines) + " | ARM %s; HARMFUL %s; entered %d/3 (landed %d); event-only mean entry %s" % (arm, harmful, out["n_entered"], out["n_landed"], out["event_only_mean_entry"])
    print("READING:", out["reading"])
    json.dump(out, open(os.path.join(ROOT, "theory", "results", "score_s205.json"), "w", newline=chr(10)), indent=1, default=str)


if __name__ == "__main__":
    main()
