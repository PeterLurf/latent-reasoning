"""§241 LEN-STEP driver (LS_v1 + Note 241-1; $0 local): load <run>/model.pt as inject_aw_eval.load(), run part_lenstep, write ONLY <out>/eval_lenstep.json (default
out = the run dir; eval.json and the other part files are not touched). fp32, one card, one eval_bs for every cell.
    python -B scratch/s241_lenstep.py --run lh_fh_s1 [--device cuda:0 --eval_bs 250 --src_dir DIR --out DIR]"""
import argparse
import json
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from boxes import inject_aw_eval as E  # noqa: E402
from boxes import inject_ow as IO  # noqa: E402


def structure(a):
    """Audit G-D §241 (descriptive): per cell the op mix per step, the mul_s share (items with >= 1 mul_s) and its step positions, the distractor count and the
    token length - D is a length-SELECTED subset of C's stream, so this shows whether its programs also differ in kind. Rebuilds the cells (deterministic, G0)."""
    from collections import Counter
    import numpy as np
    cells, _ = E.lenstep_cells(a); o = {}
    for c_, (its, _ps) in cells.items():
        n = len(its); d = its[0]["d"]
        o[c_] = {"n": n, "d": d, "op_mix_by_step": {str(t): {k: v / n for k, v in sorted(Counter(it["ops"][t - 1] for it in its).items())} for t in range(1, d + 1)},
                 "op_mix_all": {k: v / (n * d) for k, v in sorted(Counter(op for it in its for op in it["ops"]).items())},
                 "mul_s_share": sum("mul_s" in it["ops"] for it in its) / n, "mul_s_per_item_mean": float(np.mean([it["ops"].count("mul_s") for it in its])),
                 "mul_s_steps": {str(t): v / n for t, v in sorted(Counter(t for it in its for t, op in enumerate(it["ops"], 1) if op == "mul_s").items())},
                 "n_givens_mean": float(np.mean([len(it["givens"]) for it in its])), "n_distractors_mean": float(np.mean([it["n_distractors"] for it in its])),
                 "tok_mean": float(np.mean([E.ls_len(a, it) for it in its]))}
    return o


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--eval_bs", type=int, default=250)
    ap.add_argument("--src_dir", default="", help="gates only: the model dir (default boxes/runs_fsq/<run>)"); ap.add_argument("--out", default="", help="gates only: the output dir")
    a_ = ap.parse_args(); src = a_.src_dir or os.path.join(ROOT, "boxes", "runs_fsq", a_.run); out = a_.out or src; os.makedirs(out, exist_ok=True)
    t1 = time.time(); dev = a_.device; a, m, g = E.load(src, dev); assert m is not None, "§241: inject runs only"
    if dev.startswith("cuda"):
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(dev)
    r_ = E.part_lenstep(m, g, None, a, dev, a_.eval_bs); r_["structure"] = structure(a)   # Audit G-D §241: C vs D (and A / B) program structure, descriptive
    r_.update({"run": a_.run, "eval_dtype": "fp32", "eval_gpu": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu", "eval_bs": a_.eval_bs, "law": E.law(a),
               "think_k": a["think_k"], "module_sha256": IO.sha_file(E.__file__), "driver_sha256": IO.sha_file(os.path.abspath(__file__)),
               "peak_gib": torch.cuda.max_memory_allocated(dev) / 2 ** 30 if dev.startswith("cuda") else None, "sec": round(time.time() - t1)})
    json.dump(r_, open(os.path.join(out, "eval_lenstep.json"), "w", newline="\n")); print("LENSTEP done", out, r_["sec"], "s", flush=True)


if __name__ == "__main__":
    main()
