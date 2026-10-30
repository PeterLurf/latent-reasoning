"""§249 FRAME-TRANSFER driver (FT_v1 §2 + Note 249-1; $0 local): load <run>/model.pt as inject_aw_eval.load(), build the run law's sets (set shas asserted vs args.json, as
run_long), run part_transfer (LAT or CD-AN), write ONLY <out>/eval_transfer.json (default out = the run dir). fp32, one card, one eval_bs.
    python -B scratch/s249_ft.py --run lh_fh_s1 [--device cuda:0 --eval_bs 250 --src_dir DIR --out DIR]"""
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
    t1 = time.time(); dev = a_.device; a, m, g = E.load(src, dev); sets = IO.build_sets(mul_s_p=E.law(a))
    for k_, v_ in sets["shas"].items():
        assert a.get(k_) in (None, v_), ("§249: set sha mismatch vs args.json", k_)
    if dev.startswith("cuda"):
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(dev)
    for part, fn, name in (("transfer", E.part_transfer, "eval_transfer.json"),):
        t2 = time.time(); r_ = fn(m, g, sets, a, dev, a_.eval_bs)
        r_.update({"run": a_.run, "eval_dtype": "fp32", "eval_gpu": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu", "eval_bs": a_.eval_bs, "law": E.law(a),
                   "set_law": sets["shas"].get("set_law", "default"), "think_k": a.get("think_k"), "arm": a.get("arm"), "module_sha256": IO.sha_file(E.__file__),
                   "driver_sha256": IO.sha_file(os.path.abspath(__file__)), "peak_gib": torch.cuda.max_memory_allocated(dev) / 2 ** 30 if dev.startswith("cuda") else None,
                   "sec": round(time.time() - t2)})
        json.dump(r_, open(os.path.join(out, name), "w", newline="\n")); print("PART", part, "done", out, r_["sec"], "s", flush=True)
    print("FT done", out, round(time.time() - t1), "s", flush=True)


if __name__ == "__main__":
    main()
