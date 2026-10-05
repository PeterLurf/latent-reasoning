"""THEORY §120 scoring (rules fixed in §120; written before any §120 run; Theory is the only scorer).
Runs: boxes/runs_fsq/s120_{s5,ff}_{final,trace}_s{1,2}/eval.json with keys long_acc (S5: cumulative at 512; flip-flop:
read accuracy at 512, p_ignore 0.8), acc_by_length, code_perplexity, n_codes, codes_per_state, cert {holds, n_reachable,
n_moore, capped}, donor {pos2, pos4, pos6: {donor_consistent, recipient_consistent}}, margin_flip_rate.
PILOT-PASS (family x arm): long_acc >= 0.99 in both seeds. Pilot reading: FINAL passes in both families.
    python theory/score_s120.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "boxes", "runs_fsq")


def serial(d):
    """§104 SERIAL: donor >= 0.90 and recipient - chance_overlap < 0.10 (note 120-7: flip-flop's binary answers coincide
    by chance; for S5 chance_overlap is about 0.008, so this equals the registered recipient < 0.10)."""
    return bool(d) and all(v["donor_consistent"] >= 0.90 and v["recipient_consistent"] - v.get("chance_overlap", 0.0) < 0.10 for v in d.values())


def main():
    res = {"cells": {}}
    for fam in ("s5", "ff"):
        for arm in ("final", "trace"):
            runs = {}
            for s in (1, 2):
                p = os.path.join(R, "s120_%s_%s_s%d" % (fam, arm, s), "eval.json")
                if os.path.exists(p):
                    runs[s] = json.load(open(p))
                    pc = os.path.join(R, "s120_%s_%s_s%d" % (fam, arm, s), "eval_cert.json")      # note 120-3: post-hoc CERT
                    if os.path.exists(pc):
                        jc = json.load(open(pc)); runs[s]["cert"] = jc.get("cert", jc)
                        if fam == "ff" and jc.get("cert_task") is not None:                       # note 120-7: FF CERT over the task language
                            runs[s]["cert_registered"] = runs[s]["cert"]; runs[s]["cert"] = dict(jc["cert_task"])
                    c = runs[s].get("cert") or {}
                    if c.get("status") in ("inconclusive_cap", "inconclusive_time", "deferred") or c.get("capped"):
                        c["holds"] = False; c["certified_claim"] = False                         # no certification claim
                    runs[s]["cert"] = c
            C = {"n": len(runs), "long_acc": {s: r.get("long_acc") for s, r in runs.items()},
                 "PILOT_PASS": len(runs) == 2 and all((r.get("long_acc") or 0) >= 0.99 for r in runs.values()),
                 "cert": {s: (r.get("cert") or {}).get("holds") for s, r in runs.items()},
                 "cert_registered_ff": {s: (r.get("cert_registered") or {}).get("holds") for s, r in runs.items()} if fam == "ff" else None,
                 "n_reachable": {s: (r.get("cert") or {}).get("n_reachable") for s, r in runs.items()},
                 "n_moore": {s: (r.get("cert") or {}).get("n_moore") for s, r in runs.items()},
                 "codes_per_state": {s: r.get("codes_per_state") for s, r in runs.items()},
                 "code_perplexity": {s: r.get("code_perplexity") for s, r in runs.items()},
                 "serial": {s: serial(r.get("donor")) for s, r in runs.items()}}
            res["cells"]["%s_%s" % (fam, arm)] = C
            print("%-9s n %d PILOT_PASS %s | long %s | cert %s reach %s moore %s | serial %s | ppl %s" % (
                "%s_%s" % (fam, arm), C["n"], C["PILOT_PASS"], C["long_acc"], C["cert"], C["n_reachable"], C["n_moore"], C["serial"], C["code_perplexity"]))
    res["PILOT_FINAL_BOTH"] = res["cells"]["s5_final"]["PILOT_PASS"] and res["cells"]["ff_final"]["PILOT_PASS"]
    res["PILOT_TRACE_BOTH"] = res["cells"]["s5_trace"]["PILOT_PASS"] and res["cells"]["ff_trace"]["PILOT_PASS"]
    print("P-A pilot: FINAL both families", res["PILOT_FINAL_BOTH"], "| TRACE both", res["PILOT_TRACE_BOTH"])
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "score_s120.json"), "w", newline="\n"), indent=1, default=float)


if __name__ == "__main__":
    main()
