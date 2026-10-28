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


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--eval_bs", type=int, default=250)
    ap.add_argument("--src_dir", default="", help="gates only: the model dir (default boxes/runs_fsq/<run>)"); ap.add_argument("--out", default="", help="gates only: the output dir")
    a_ = ap.parse_args(); src = a_.src_dir or os.path.join(ROOT, "boxes", "runs_fsq", a_.run); out = a_.out or src; os.makedirs(out, exist_ok=True)
    t1 = time.time(); dev = a_.device; a, m, g = E.load(src, dev); assert m is not None, "§241: inject runs only"
    if dev.startswith("cuda"):
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(dev)
    r_ = E.part_lenstep(m, g, None, a, dev, a_.eval_bs)
    r_.update({"run": a_.run, "eval_dtype": "fp32", "eval_gpu": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu", "eval_bs": a_.eval_bs, "law": E.law(a),
               "think_k": a["think_k"], "module_sha256": IO.sha_file(E.__file__), "driver_sha256": IO.sha_file(os.path.abspath(__file__)),
               "peak_gib": torch.cuda.max_memory_allocated(dev) / 2 ** 30 if dev.startswith("cuda") else None, "sec": round(time.time() - t1)})
    json.dump(r_, open(os.path.join(out, "eval_lenstep.json"), "w", newline="\n")); print("LENSTEP done", out, r_["sec"], "s", flush=True)


if __name__ == "__main__":
    main()
