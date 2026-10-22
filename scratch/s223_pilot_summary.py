"""§223 Note 223-4 pilot summary (descriptive; LP1-LP3 are read by Theory): per cell (DIRECT, PS on the argv card; m1_4060 / direct_4060 when present,
their card in card_meta from the cell's cell_meta.json) from scratch/_tmp/s223_pilot/<cell>/ ->
stages.jsonl lines (each advance: step, from, to, reason), entry_step (the trainer's first 100-step mean num_ce < 0.5), num_ce per 1k-step bin (mean of
the 100-step rows in the bin), the gn_* rows (step, gn_ans_shared, gn_rec_shared, cos_ans_rec), the last logged step, sec, micro; + card and kernel.
    python -B scratch/s223_pilot_summary.py <cuda dev> <kernel>  -> scratch/gate_s223/pilot.json"""
import json
import math
import os
import sys

import numpy as np

P2 = "--p2" in sys.argv                                                   # Note 223-8 pilot P2: its dir, cells and output
T = "scratch/_tmp/s223_p2" if P2 else "scratch/_tmp/s223_pilot"


def cell(c):
    d = os.path.join(T, c); rows = [json.loads(l) for l in open(os.path.join(d, "log.jsonl")) if l.strip()] if os.path.exists(os.path.join(d, "log.jsonl")) else []
    st = [json.loads(l) for l in open(os.path.join(d, "stages.jsonl"))] if os.path.exists(os.path.join(d, "stages.jsonl")) else []
    args = json.load(open(os.path.join(d, "args.json"))) if os.path.exists(os.path.join(d, "args.json")) else {}
    rows = sorted({r["step"]: r for r in rows}.values(), key=lambda r: r["step"])     # a --resume re-runs steps after the ckpt: last record per step
    st = sorted({(r.get("step"), r.get("from"), r.get("to")): r for r in st}.values(), key=lambda r: r.get("step") or 0)
    bins = {}; abins = {}
    for r in rows:
        bins.setdefault(str((r["step"] - 1) // 1000), []).append(r.get("num_ce")); abins.setdefault(str((r["step"] - 1) // 1000), []).append(r.get("num_acc"))
    last = rows[-1]["step"] if rows else 0; l5 = [r for r in rows if r["step"] > last - 500]
    w = [r for r in rows if 7100 <= r["step"] <= 7600]                  # Note 223-8 amendment 2: LP4 window (100-step log rows 7,100..7,600)
    nf = [r["step"] for r in rows if any(isinstance(r.get(k), float) and not math.isfinite(r[k]) for k in ("loss", "gn_preclip", "num_ce", "num_acc"))]
    return {"stages": st, "first_advance": st[0] if st else None, "entry_step": next((r.get("entry_step") for r in reversed(rows) if r.get("entry_step") is not None), None),
            "num_ce_1k_bins": {k: float(np.mean([v for v in vs if v is not None])) for k, vs in bins.items()},
            "gn_rows": [[r["step"], r.get("gn_ans_shared"), r.get("gn_rec_shared"), r.get("cos_ans_rec")] for r in rows if r.get("gn_ans_shared") is not None],
            "last_step": rows[-1]["step"] if rows else None, "sec": rows[-1]["sec"] if rows else None, "micro": args.get("micro"), "cell": args.get("cell"), "train_gpu": args.get("train_gpu"), "transformers": args.get("transformers_version"),
            "num_acc_1k_bins": {k: float(np.mean([v for v in vs if v is not None])) for k, vs in abins.items() if any(v is not None for v in vs)},
            "last500": {"num_acc": float(np.mean([r["num_acc"] for r in l5])) if l5 else None, "num_ce": float(np.mean([r["num_ce"] for r in l5])) if l5 else None,
                        "d_max": l5[-1].get("d_max") if l5 else None},
            "m2_rows": [[r["step"], r.get("m2_m"), r.get("m2_start")] for r in rows if "m2_m" in r],
            "lp4_window_7100_7600": {"n_rows": len(w), "num_acc": float(np.mean([r["num_acc"] for r in w])) if w else None,
                                     "num_ce": float(np.mean([r["num_ce"] for r in w])) if w else None, "d_max": [r.get("d_max") for r in w]},
            "lp0": {"gn_preclip_max_of_100step_means": max((r["gn_preclip"] for r in rows if isinstance(r.get("gn_preclip"), float) and math.isfinite(r["gn_preclip"])), default=None),
                    "nonfinite_rows": nf, "note": "the trainer logs the 100-step MEAN gn_preclip only (no per-step max): a single step > 1e3 gives a mean > 10"},
            "module_sha256": args.get("module_sha256"), "card_meta": json.load(open(os.path.join(d, "cell_meta.json"))) if os.path.exists(os.path.join(d, "cell_meta.json")) else None}


def main():
    out = {"lp5_mass_share_gate": json.load(open("scratch/gate_s223p2/iii.json")) if P2 and os.path.exists("scratch/gate_s223p2/iii.json") else None,
           "note": "§223 Note 223-8 pilot P2 (descriptive; LP1-LP3 read by Theory)" if P2 else "§223 Note 223-4 learning pilot (descriptive; LP1-LP3 read by Theory)", "card_device": sys.argv[1], "kernel": sys.argv[2],
           "cells": ({c: cell(c) for c in ("direct", "m1", "ps")} if P2 else
                     {c: cell(c) for c in ("direct", "ps", "m1_4060", "direct_4060") if c in ("direct", "ps") or os.path.isdir(os.path.join(T, c))})}
    os.makedirs("scratch/gate_s223", exist_ok=True); json.dump(out, open("scratch/gate_s223/pilot2.json" if P2 else "scratch/gate_s223/pilot.json", "w", newline="\n"), indent=1)
    print("PILOT-SUMMARY", {c: {k: v[k] for k in ("first_advance", "entry_step", "last_step")} for c, v in out["cells"].items()})


if __name__ == "__main__":
    main()
