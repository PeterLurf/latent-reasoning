"""Queue wrapper for a post-hoc fsq120 CERT on a Vast host (Theory 2026-10-06: §128 S5 CERTs, fp32 only, --cert_time 2700, descriptive).
Runs fsq120 --from_args <run>/args.json --cert_only --fp32 on the shipped <run>/model.pt, then copies eval_cert_fp32.json into --out
(a separate dir, so the drain fetches only the small result, not the model) and writes --out/eval.json (the CERT summary) for the queue.
    python -m scratch.cert_job --run boxes/runs_fsq/s128_s5_s1 --device cuda:0 --out boxes/runs_fsq/s128cert_s5_s1"""
import argparse
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--out", required=True)
ap.add_argument("--cert_time", type=int, default=2700)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
rc = subprocess.call([sys.executable, "-u", "-m", "boxes.fsq120", "--from_args", os.path.join(a.run, "args.json"), "--cert_only", "--fp32",
                      "--cert_time", str(a.cert_time), "--device", a.device, "--out", a.run], cwd=ROOT)
src = os.path.join(a.run, "eval_cert_fp32.json")
if rc == 0 and os.path.exists(src):
    shutil.copy(src, os.path.join(a.out, "eval_cert_fp32.json")); c = json.load(open(src))
    json.dump({"run": a.run, "rc": rc, **{k: c.get(k) for k in ("holds", "n_moore", "status", "n_reachable", "eval_dtype", "device", "driver", "cert_cap")}},
              open(os.path.join(a.out, "eval.json"), "w", newline="\n"), indent=1)
sys.exit(rc)
