"""THEORY §172 scoring (rules fixed in §172; written before any §172 run; Theory is the only scorer).
PCCoT-PROBE: §149's battery on whynlp/pccot-gpt2 (GPT-2 small + LoRA, 24 Jacobi-parallel latents, T = 3), eval only, on bench/data/gsm8k_aug/test.txt (1,319).
SANITY GATE: release accuracy within +-2 points of 49.48 (else not read). LATENTS-USED: all latents zeroed <= 0.5 x release. ITERATIONS-USED: T' = 0 <= release
- 0.05. RHO: geometric mean of the Jacobi step ratio r_T over T = 3..8: > 1 EXPANDING, <= 0.28 CONTRACTIVE, else MARGINAL. SERIAL is descriptive (probe-defined
alignment; BYPASS(t) = recipient-consistent >= 0.5; n < 30 UNDERPOWERED).
Dir boxes/runs_fsq/s172_pccot.
    python theory/score_s172.py
"""
import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "boxes", "runs_fsq")
REPORTED = 0.4948


def load(d, f="eval.json"):
    p = os.path.join(R, d, f)
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    r = load("s172_pccot"); res = {}
    if r is None:
        res["reading"] = "pending"; print("READING: pending")
    else:
        a = r.get("acc_release"); bt = r.get("acc_by_T") or {}
        gate = a is not None and abs(a - REPORTED) <= 0.02
        res.update({"acc_release": a, "sanity_gate": gate, "acc_by_T": bt, "acc_latents_zeroed": r.get("acc_latents_zeroed"), "acc_c0": r.get("acc_c0")})
        if not gate:
            res["reading"] = "SANITY GATE failed (release %s vs %.4f +- 0.02): not read" % (a, REPORTED)
        else:
            z = r.get("acc_latents_zeroed"); t0 = bt.get("0"); t3 = bt.get("3")
            if t3 is not None and abs(t3 - a) > 1e-9:
                print("WARNING: acc_by_T['3'] %s != acc_release %s (gate: T' = 3 reproduces release)" % (t3, a))
            used = None if z is None else bool(z <= 0.5 * a)
            iters = None if t0 is None else bool(t0 <= a - 0.05)
            jr = r.get("jacobi_ratio_by_T") or {}
            rs = [float(jr[str(T)]) for T in range(3, 9) if jr.get(str(T)) is not None and float(jr[str(T)]) > 0]
            gm = math.exp(sum(math.log(x) for x in rs) / len(rs)) if len(rs) == 6 else None
            rho = None if gm is None else ("EXPANDING" if gm > 1 else ("CONTRACTIVE" if gm <= 0.28 else "MARGINAL"))
            ser = {}
            for t, v in (r.get("serial") or {}).items():
                n = v.get("n") or 0
                ser[t] = dict(v, cell="UNDERPOWERED" if n < 30 else ("BYPASS" if (v.get("recipient") or 0) >= 0.5 else "not BYPASS"))
            res.update({"latents_used": used, "iterations_used": iters, "jacobi_gm_3_8": gm, "rho": rho, "rho_perturb_by_iter": r.get("rho_perturb_by_iter"),
                        "serial_descriptive": ser})
            if used and iters:
                rd = "PCCoT computes in its latents and across its Jacobi iterations (zeroed %.3f, T'=0 %.3f vs %.3f)" % (z, t0, a)
            elif used and iters is False:
                rd = "latents used but iterations not (T'=0 %.3f vs %.3f): pause-like width, no serial latent use" % (t0, a)
            elif used is False:
                rd = "latents not used (zeroed %.3f vs %.3f)" % (z, a)
            else:
                rd = "incomplete fields"
            byp = [t for t, v in ser.items() if v["cell"] == "BYPASS"]
            rd += "; RHO %s (gm %s)" % (rho, None if gm is None else round(gm, 3)) + ("; BYPASS at steps %s" % byp if byp else "")
            res["reading"] = rd
        print("READING:", res["reading"])
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "score_s172.json"), "w", newline=chr(10)), indent=1, default=float)


if __name__ == "__main__":
    main()
