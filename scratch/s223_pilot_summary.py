"""§223 Note 223-4 pilot summary (descriptive; LP1-LP3 are read by Theory): per cell (DIRECT, PS) from scratch/_tmp/s223_pilot/<cell>/ ->
stages.jsonl lines (each advance: step, from, to, reason), entry_step (the trainer's first 100-step mean num_ce < 0.5), num_ce per 1k-step bin (mean of
the 100-step rows in the bin), the gn_* rows (step, gn_ans_shared, gn_rec_shared, cos_ans_rec), the last logged step, sec, micro; + card and kernel.
    python -B scratch/s223_pilot_summary.py <cuda dev> <kernel>  -> scratch/gate_s223/pilot.json"""
import json
import os
import sys

import numpy as np

T = "scratch/_tmp/s223_pilot"


def cell(c):
    d = os.path.join(T, c); rows = [json.loads(l) for l in open(os.path.join(d, "log.jsonl")) if l.strip()] if os.path.exists(os.path.join(d, "log.jsonl")) else []
    st = [json.loads(l) for l in open(os.path.join(d, "stages.jsonl"))] if os.path.exists(os.path.join(d, "stages.jsonl")) else []
    args = json.load(open(os.path.join(d, "args.json"))) if os.path.exists(os.path.join(d, "args.json")) else {}
    bins = {}
    for r in rows:
        bins.setdefault(str((r["step"] - 1) // 1000), []).append(r.get("num_ce"))
    return {"stages": st, "first_advance": st[0] if st else None, "entry_step": next((r.get("entry_step") for r in reversed(rows) if r.get("entry_step") is not None), None),
            "num_ce_1k_bins": {k: float(np.mean([v for v in vs if v is not None])) for k, vs in bins.items()},
            "gn_rows": [[r["step"], r.get("gn_ans_shared"), r.get("gn_rec_shared"), r.get("cos_ans_rec")] for r in rows if r.get("gn_ans_shared") is not None],
            "last_step": rows[-1]["step"] if rows else None, "sec": rows[-1]["sec"] if rows else None, "micro": args.get("micro"), "cell": args.get("cell"), "train_gpu": args.get("train_gpu"), "transformers": args.get("transformers_version"),
            "module_sha256": args.get("module_sha256")}


def main():
    out = {"note": "§223 Note 223-4 learning pilot (descriptive; LP1-LP3 read by Theory)", "card_device": sys.argv[1], "kernel": sys.argv[2],
           "cells": {c: cell(c) for c in ("direct", "ps")}}
    os.makedirs("scratch/gate_s223", exist_ok=True); json.dump(out, open("scratch/gate_s223/pilot.json", "w", newline="\n"), indent=1)
    print("PILOT-SUMMARY", {c: {k: v[k] for k in ("first_advance", "entry_step", "last_step")} for c, v in out["cells"].items()})


if __name__ == "__main__":
    main()
