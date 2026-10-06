"""THEORY 128-4 on one Vast 3090 (Theory 2026-10-06): queue-runner wrapper for scratch/cert_cover128.py.
    python -m scratch.cc_job --run boxes/runs_fsq/s128_<fam>_s<k> --device cuda:0 --out boxes/runs_fsq/s128cc3090_<fam>_s<k>
Writes eval_cert_cover.json + eval_cert_fp32.json into --out (not the run dir, so the 4060 files stay distinguishable), copies the
run's args.json there as source_args.json, and writes eval.json (the cover summary + gpu / driver) so gen_vast_job counts it done."""
import argparse
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scratch"))
import cert_cover128  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--out", required=True)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
shutil.copy(os.path.join(a.run, "args.json"), os.path.join(a.out, "source_args.json"))
out = cert_cover128.main(a.run, a.device, None, a.out)
out["gpu_driver"] = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
json.dump(out, open(os.path.join(a.out, "eval.json"), "w", newline="\n"), indent=1)
