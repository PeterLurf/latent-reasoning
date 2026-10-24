"""REVIEW_HD: far-loss trajectories from committed log.jsonl (read-only): 2k-bin means of far_ce, fh_gate_share, answer_ce, lr; hold-signal proxies.
    python -B docs/research_holddepth_2026-10-23/review_scripts/rv_logs.py"""
import json, os
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
out = {}
for rn in ["s224_rd_s1", "s224_rd_s2", "s224_rd_s3", "s224_rdft_s1", "s224_rdft_s2", "s224_fhft_s1", "s224_fhft_s2"]:
    L = [json.loads(l) for l in open(os.path.join(ROOT, "boxes/runs_fsq", rn, "log.jsonl")) if l.strip()]
    L = [r for r in L if "far_ce" in r and r.get("far_ce") is not None]
    S = max(r["step"] for r in L); bw = 2000 if S > 20000 else 1000
    bins = {}
    for r in L:
        b = (r["step"] - 1) // bw
        bins.setdefault(b, []).append(r)
    rows = []
    for b in sorted(bins):
        rs = bins[b]
        f = lambda k: float(np.mean([r[k] for r in rs if r.get(k) is not None])) if any(r.get(k) is not None for r in rs) else None   # noqa: E731
        rows.append({"bin_end": (b + 1) * bw, "far_ce": f("far_ce"), "gate_Tm1": f("fh_gate_share"), "answer_ce": f("answer_ce"), "lr": f("lr"), "gn": f("gn_preclip")})
    # lr-integral (per 100-step record) and lr-weighted far_ce on gated (g = 1 at T-1) vs not: proxy only (far_ce is not split by gate in the log)
    lr_int = sum(r["lr"] * 100 for r in L)
    lr_int_after16k = sum(r["lr"] * 100 for r in L if r["step"] > 16000)
    rep_dose = sum((1 - r["fh_gate_share"]) * 100 for r in L)
    rep_lr_dose = sum((1 - r["fh_gate_share"]) * r["lr"] * 100 for r in L)
    farce_lr = sum(r["far_ce"] * r["lr"] * 100 for r in L)
    out[rn] = {"bins": [{k: (round(v, 5) if isinstance(v, float) else v) for k, v in x.items()} for x in rows],
               "lr_integral": lr_int, "lr_integral_after_16k": lr_int_after16k, "repair_dose_steps": rep_dose, "repair_lr_dose": rep_lr_dose, "int_far_ce_x_lr": farce_lr,
               "first_step_gate_ge_0.99": next((r["step"] for r in L if r["fh_gate_share"] >= 0.99), None),
               "final_far_ce_last1k": float(np.mean([r["far_ce"] for r in L if r["step"] > S - 1000]))}
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rv_logs.out.json"), "w", newline="\n"), indent=1)
for rn, d in out.items():
    print(rn, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items() if k != "bins"})
    for x in d["bins"][:12]:
        print("   ", x)
