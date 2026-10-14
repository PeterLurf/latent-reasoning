"""THEORY §199-A scoring (CSR: completion-switched step-ratio target; rules fixed in amendment 199-A before any §199 run; Theory is the only scorer).
Cells s199_csr_s{1,2,3} = s185_rms_s{seed}'s args + --csr "0.1,-0.0513,-0.6931,0.05,500"; paired exact controls s185_rms_s{seed}.
Per seed:
  PAIRED   = every log record at steps 100..500 (loss, answer_ce, aux_ce) equals the control's bitwise (A4); else UNPAIRED (reads vs the 2/3 base rate).
  P-A1     = mean of log field csr_lam_pre_med over the records in the 1k bin 7001..8000 >= log 0.95 (-0.0513).
  P-A2     = pinned basin_entry: entry(0.5) <= 24000 and final 1k bin < 0.1.
  P-A3     = final eval.json s182.path_indep delta_cos_final <= 0.65 and acc_from_zero >= 0.95.
  P-A5     = FALSIFIER if the 8k AND 16k checkpoint probes (pathindep_8000.json / pathindep_16000.json, Experiments' probe_pathindep.py) read
             non-contractive (delta_cos_final <= 0.65 and answer_agree <= 0.2) AND the final 1k bin >= 0.5.
  FLOOR-GAMED = P-A1 met but a checkpoint probe (8k or 16k) contractive (delta_cos_final > 0.65 or answer_agree > 0.2) and not entered.
  INSUFFICIENT-DOSE = P-A1 not met.
Overall: SUPPORTED = P-A1 + P-A2 + P-A3 in all 3 seeds (paired status printed beside); FALSIFIED = any seed FALSIFIER; else MIXED (per-seed labels).
P-A4 (pooled K'16 >= K'8 - 0.02) and per-head-type lambda are descriptive.
    python theory/score_s199_a.py
"""
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import basin_entry as BE  # noqa: E402

R = os.path.join(ROOT, "boxes", "runs_fsq")
SEEDS, EPS = (1, 2, 3), 1e-9
LAM_MIN = math.log(0.95)


def log_records(d):
    return [json.loads(l_) for l_ in open(os.path.join(R, d, "log.jsonl")) if l_.strip()]


def paired(cell, ctrl):
    """199-A2 (Audit's ruling): (i) records 100..400 bitwise on (loss, answer_ce, aux_ce); (ii) step 500's answer_ce / aux_ce bitwise; (iii) step 500's
    loss_cell - loss_ctrl == 0.1 * csr_loss_500 / 100 within 1e-6 relative (the record's loss is the mean of steps 401..500's totals; CSR is on from
    step 500 inclusive; csr_loss is that step's own batch value). Any other residual -> UNPAIRED."""
    def early(d):
        return {r["step"]: r for r in log_records(d) if "loss" in r and 100 <= r["step"] <= 500}
    a, b = early(cell), early(ctrl)
    if not a or set(a) != set(b) or 500 not in a:
        return False, {"why": "record steps differ or step 500 missing"}
    f = ("loss", "answer_ce", "aux_ce")
    ok1 = all(a[k][x] == b[k][x] for k in a if k <= 400 for x in f)
    ok2 = a[500]["answer_ce"] == b[500]["answer_ce"] and a[500]["aux_ce"] == b[500]["aux_ce"]
    cs = a[500].get("csr_loss"); pred = None if cs is None else 0.1 * cs / 100; d500 = a[500]["loss"] - b[500]["loss"]
    ok3 = pred is not None and abs(d500 - pred) <= 1e-6 * max(abs(b[500]["loss"]), 1e-12)
    return bool(ok1 and ok2 and ok3), {"i_100_400": ok1, "ii_500_ce": ok2, "iii_500_loss": ok3, "d500": d500, "pred": pred}


def pa1(d):
    v = [r["csr_lam_pre_med"] for r in log_records(d) if 7001 <= r.get("step", 0) <= 8000 and r.get("csr_lam_pre_med") is not None]
    return (sum(v) / len(v) if v else None), len(v)


def probe(d, step):
    p = os.path.join(R, d, "pathindep_%d.json" % step)
    if not os.path.exists(p):
        return None
    j = json.load(open(p)); j = j.get("path_indep", j)
    return {"delta_cos_final": j["delta_cos_final"], "answer_agree": j["answer_agree"]}


def noncontractive(pr):
    return pr["delta_cos_final"] <= 0.65 + EPS and pr["answer_agree"] <= 0.2 + EPS


def seed_read(s):
    cell, ctrl = "s199_csr_s%d" % s, "s185_rms_s%d" % s
    if not os.path.exists(os.path.join(R, cell, "eval.json")):
        return {"status": "pending"}
    pr_ok, _ = paired(cell, ctrl); lam, nrec = pa1(cell); ent = BE.summary(cell)
    e = json.load(open(os.path.join(R, cell, "eval.json"))); pi = (e.get("s182") or {}).get("path_indep")
    a1 = lam is not None and lam >= LAM_MIN - EPS
    a2 = ent["entry_0.5"] is not None and ent["entry_0.5"] <= 24000 and ent["last_bin"] < 0.1 - EPS
    a3 = pi is not None and pi["delta_cos_final"] <= 0.65 + EPS and pi["acc_from_zero"] >= 0.95 - EPS
    p8, p16 = probe(cell, 8000), probe(cell, 16000); entered_not = ent["last_bin"] >= 0.5 - EPS
    if p8 and p16 and noncontractive(p8) and noncontractive(p16) and entered_not:
        lab = "FALSIFIER (P-A5)"
    elif not a1:
        lab = "INSUFFICIENT-DOSE"
    elif a1 and a2 and a3:
        lab = "ENTERED-NONCONTRACTIVE"
    elif a1 and not a2 and ((p8 and not noncontractive(p8)) or (p16 and not noncontractive(p16))):
        lab = "FLOOR-GAMED"
    else:
        lab = "OTHER"
    kp = (e.get("s170") or {}).get("acc_by_n_kprime", {})
    pool = lambda K: (sum(kp[K][str(n)] for n in range(3, 9)) / 6) if K in kp else None   # noqa: E731
    lam_t = {k_: [r.get(k_) for r in log_records(cell) if 7001 <= r.get("step", 0) <= 8000] for k_ in ("csr_lam_pre_med_add", "csr_lam_pre_med_del", "csr_lam_post_med")}
    return {"status": "done", "label": lab, "paired": pr_ok, "pa1_lam": lam, "pa1_n_records": nrec, "pa1": a1, "pa2": a2, "pa3": a3, "entry": ent,
            "path_indep": pi, "probe_8k": p8, "probe_16k": p16, "pa4_stable": None if pool("8") is None or pool("16") is None else pool("16") >= pool("8") - 0.02 - EPS,
            "lam_by_type_8k": {k_: (sum(v for v in vs if v is not None) / max(1, sum(v is not None for v in vs))) if any(v is not None for v in vs) else None for k_, vs in lam_t.items()}}


def main():
    res = {s: seed_read(s) for s in SEEDS}; done = [s for s in SEEDS if res[s]["status"] == "done"]
    if len(done) < 3:
        ov = "pending (%d / 3 seeds done)" % len(done)
    elif any(res[s]["label"].startswith("FALSIFIER") for s in SEEDS):
        ov = "FALSIFIED (P-A5: non-contractive at 8k and 16k yet not entered)"
    elif all(res[s]["label"] == "ENTERED-NONCONTRACTIVE" for s in SEEDS):
        ov = "SUPPORTED (P-A1 + P-A2 + P-A3 in 3 / 3)"
    else:
        ov = "MIXED"
    per = " | ".join("s%d %s (paired %s; lam8k %s over %s recs; entry %s / final %s; dcos %s)" % (
        s, r.get("label", r["status"]), r.get("paired"), None if r.get("pa1_lam") is None else round(r["pa1_lam"], 4), r.get("pa1_n_records"),
        (r.get("entry") or {}).get("entry_0.5"), None if not r.get("entry") else round(r["entry"]["last_bin"], 4),
        None if not r.get("path_indep") else round(r["path_indep"]["delta_cos_final"], 3)) for s, r in res.items())
    rd = "%s :: %s" % (ov, per); print("READING:", rd)
    unp = [s for s in done if not res[s]["paired"]]
    if unp:
        print("UNPAIRED seeds (read vs the 2/3 base rate only):", unp)
    json.dump({"reading": rd, "seeds": res}, open(os.path.join(ROOT, "theory", "results", "score_s199_a.json"), "w", newline=chr(10)), indent=1, default=str)


if __name__ == "__main__":
    main()
