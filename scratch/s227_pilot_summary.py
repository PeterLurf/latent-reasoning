"""§227 TEXT-SERIAL pilot summary (registered f69fd658; the pass rule is read by Theory) -> scratch/gate_s227/pilot_text.json, from
scratch/_tmp/s227_text/TEXT: decode (decode_evals.jsonl, last record per step: free-decode acc / terminated / parsed / per-step value acc / wrap split by d),
stages (stages.jsonl), entry_step (on step_value_ce), the teacher-forced 1k bins (step_value_ce, step_value_acc, num_ce), LP0 (gn_preclip rows, the first
window flagged; the max after it; non-finite rows), per-d exposure, args, card_meta.   python -B scratch/s227_pilot_summary.py"""
import json
import math
import os

import numpy as np

D = "scratch/_tmp/s227_text/TEXT"


def main():
    f = lambda n: os.path.join(D, n)  # noqa: E731
    rd = lambda n: [json.loads(l) for l in open(f(n)) if l.strip()] if os.path.exists(f(n)) else []  # noqa: E731
    rows = sorted({r["step"]: r for r in rd("log.jsonl")}.values(), key=lambda r: r["step"])
    dec = sorted({r["step"]: r for r in rd("decode_evals.jsonl")}.values(), key=lambda r: r["step"])
    args = json.load(open(f("args.json"))) if os.path.exists(f("args.json")) else {}
    bins = {}
    for r in rows:
        bins.setdefault(str((r["step"] - 1) // 1000), []).append(r)
    mb = lambda rs, k: float(np.mean([r[k] for r in rs if r.get(k) is not None])) if any(r.get(k) is not None for r in rs) else None  # noqa: E731
    nf = [r["step"] for r in rows if any(isinstance(r.get(k), float) and not math.isfinite(r[k]) for k in ("loss", "gn_preclip", "step_value_ce"))]
    gn = [[r["step"], r.get("gn_preclip")] for r in rows]; after = [g for _, g in gn[1:] if isinstance(g, float) and math.isfinite(g)]
    dh = np.zeros(13)
    for r in rows:
        dh[:len(r.get("d_hist", []))] += np.array(r.get("d_hist", []))
    out = {"note": "§227 TEXT-SERIAL (registered f69fd658): pass = free-decode d 2 >= 0.80 AND d 3 >= 0.70 at stop + LP0; read by Theory",
           "decode": dec, "decode_at_stop": dec[-1] if dec else None, "stages": rd("stages.jsonl"),
           "entry_step": next((r.get("entry_step") for r in reversed(rows) if r.get("entry_step") is not None), None),
           "tf_1k_bins": {k: {"step_value_ce": mb(v, "step_value_ce"), "step_value_acc": mb(v, "step_value_acc"), "num_ce": mb(v, "num_ce")} for k, v in bins.items()},
           "lp0": {"gn_preclip_rows": gn, "first_window_flag": "row 1 (steps 1..100) includes the warm-up; LP0 reads the rows after it",
                   "gn_preclip_max_after_first_window": max(after) if after else None, "nonfinite_rows": nf},
           "exposure_by_d": {"steps": {str(i): int(x) for i, x in enumerate(dh) if x}, "items": {str(i): int(x * args.get("eff_bs", 64)) for i, x in enumerate(dh) if x}},
           "last_step": rows[-1]["step"] if rows else None, "sec": rows[-1]["sec"] if rows else None, "micro": args.get("micro"),
           "card_meta": json.load(open(f("cell_meta.json"))) if os.path.exists(f("cell_meta.json")) else None,
           "args": {k: args.get(k) for k in ("task", "order_mode", "full_ft", "lr", "text_cot", "cg_signal", "cg_thr", "cg_cap", "steps", "text_eval_sha", "module_sha256", "train_gpu")}}
    os.makedirs("scratch/gate_s227", exist_ok=True); json.dump(out, open("scratch/gate_s227/pilot_text.json", "w", newline="\n"), indent=1)
    print("PILOT-TEXT", out["last_step"], len(dec), "decode records", len(out["stages"]), "stages")


if __name__ == "__main__":
    main()
