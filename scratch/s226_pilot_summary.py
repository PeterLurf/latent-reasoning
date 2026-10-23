"""§226 ORDERED-WORDS stage-0 pilot summary (registered 09368b14; descriptive fields, gates read by Theory) -> scratch/gate_s226/pilot_ow.json.
Per cell (scratch/_tmp/s226_p0/<cell>): stages (stages.jsonl), entry_step, the num_acc / num_ce 1k bins (mean of the 100-step rows in each bin), LP0
(gn_preclip rows, the first window flagged; non-finite rows; max of the 100-step means after the first window), the wrap split (pooled number-token
accuracy on items whose final step wrapped mod 11 vs not: sum of the logged per-window means of num_ok_* / n_* over all rows and per 1k bin), and
per-d training exposure (the summed d_hist over all rows = steps per d; items = steps x eff_bs), + card_meta.   python -B scratch/s226_pilot_summary.py"""
import json
import math
import os

import numpy as np

T = "scratch/_tmp/s226_p0"; CELLS = ("P0-OW", "P0b-TOPO", "P0c-CHAIN")


def cell(c):
    d = os.path.join(T, c); f = lambda n: os.path.join(d, n)  # noqa: E731
    rows = [json.loads(l) for l in open(f("log.jsonl")) if l.strip()] if os.path.exists(f("log.jsonl")) else []
    rows = sorted({r["step"]: r for r in rows}.values(), key=lambda r: r["step"])                    # a --resume re-runs steps: last record per step
    st = [json.loads(l) for l in open(f("stages.jsonl"))] if os.path.exists(f("stages.jsonl")) else []
    args = json.load(open(f("args.json"))) if os.path.exists(f("args.json")) else {}
    meta = json.load(open(f("cell_meta.json"))) if os.path.exists(f("cell_meta.json")) else None
    bins = {}
    for r in rows:
        bins.setdefault(str((r["step"] - 1) // 1000), []).append(r)
    mb = lambda rs, k: float(np.mean([r[k] for r in rs if r.get(k) is not None])) if any(r.get(k) is not None for r in rs) else None  # noqa: E731
    pooled = lambda rs, ok, n: (sum(r.get(ok, 0.0) for r in rs) / sum(r.get(n, 0.0) for r in rs)) if sum(r.get(n, 0.0) for r in rs) > 0 else None  # noqa: E731
    nf = [r["step"] for r in rows if any(isinstance(r.get(k), float) and not math.isfinite(r[k]) for k in ("loss", "gn_preclip", "num_ce", "num_acc"))]
    gn = [[r["step"], r.get("gn_preclip")] for r in rows]; after = [g for s_, g in gn[1:] if isinstance(g, float) and math.isfinite(g)]
    dh = np.zeros(13)
    for r in rows:
        dh[:len(r.get("d_hist", []))] += np.array(r.get("d_hist", []))
    eff = args.get("eff_bs", 64)
    return {"stages": st, "entry_step": next((r.get("entry_step") for r in reversed(rows) if r.get("entry_step") is not None), None),
            "num_acc_1k_bins": {k: mb(v, "num_acc") for k, v in bins.items()}, "num_ce_1k_bins": {k: mb(v, "num_ce") for k, v in bins.items()},
            "lp0": {"gn_preclip_rows": gn, "first_window_flag": "row 1 (steps 1..100) includes the warm-up; LP0 reads the rows after it",
                    "gn_preclip_max_after_first_window": max(after) if after else None, "nonfinite_rows": nf},
            "wrap_split": {"num_acc_wrap": pooled(rows, "num_ok_wrap", "n_wrap"), "num_acc_nowrap": pooled(rows, "num_ok_nowrap", "n_nowrap"),
                           "share_wrap": (sum(r.get("n_wrap", 0) for r in rows) / max(1e-9, sum(r.get("n_wrap", 0) + r.get("n_nowrap", 0) for r in rows))) if rows else None,
                           "by_1k_bin": {k: {"wrap": pooled(v, "num_ok_wrap", "n_wrap"), "nowrap": pooled(v, "num_ok_nowrap", "n_nowrap")} for k, v in bins.items()}},
            "exposure_by_d": {"steps": {str(i): int(x) for i, x in enumerate(dh) if x}, "items": {str(i): int(x * eff) for i, x in enumerate(dh) if x}},
            "last_step": rows[-1]["step"] if rows else None, "sec": rows[-1]["sec"] if rows else None, "micro": args.get("micro"), "card_meta": meta,
            "args": {k: args.get(k) for k in ("task", "order_mode", "dag", "stream_seed", "lr", "cg_thr", "cg_cap", "steps", "direct", "train_gpu", "module_sha256", "ow_words_sha256")}}


def main():
    out = {"note": "§226 ORDERED-WORDS stage 0 (registered 09368b14): one-pass learnability attribution panel; gates LP0 / G1 / G2 / G3 / THRESHOLD-MISSET read by Theory",
           "cells": {c: cell(c) for c in CELLS if os.path.isdir(os.path.join(T, c))}}
    os.makedirs("scratch/gate_s226", exist_ok=True); json.dump(out, open("scratch/gate_s226/pilot_ow.json", "w", newline="\n"), indent=1)
    print("PILOT-OW", {c: (v["last_step"], len(v["stages"])) for c, v in out["cells"].items()})


if __name__ == "__main__":
    main()
