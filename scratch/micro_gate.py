"""Audit 2026-10-06 (note 131-4 reruns): pick the smallest --micro in (1, 2, 4) whose training step fits this card at L 31 and L 32
(fsq120 applies --micro at L > 16; L 31 at micro 1 is the largest micro-1 shape; at micro > 1 also L 16 at micro 1), with >= MARGIN GiB left, then run fsq120 with it.
Measurements -> <out>/micro_gate.json; fsq120 records the chosen micro in args.json.
    python -m scratch.micro_gate --device cuda:0 --out <run dir> [--gate_only] [--Ls 31,32] <fsq120 args without --micro / --device / --out>"""
import argparse
import json
import os
import re
import subprocess
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MARGIN = 1.5

ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--out", required=True)
ap.add_argument("--gate_only", action="store_true"); ap.add_argument("--Ls", default="31,32")
a, rest = ap.parse_known_args(); assert "--micro" not in rest, "micro_gate chooses --micro"
os.makedirs(a.out, exist_ok=True); total = torch.cuda.get_device_properties(a.device).total_memory / 2 ** 30
rec = {"device": torch.cuda.get_device_name(a.device), "total_gib": round(total, 2), "margin_gib": MARGIN, "checks": []}
chosen = None
for mc in (1, 2, 4):
    ok = True
    shapes = [(mc, int(x)) for x in a.Ls.split(",")] + ([(1, 16)] if mc > 1 else [])   # Audit: at micro > 1 the L 16 stage still runs micro 1
    for mc_, L in shapes:
        p = subprocess.run([sys.executable, os.path.join(ROOT, "scratch", "memcheck_fsq120.py"), a.device, str(mc_), "64", str(L)], capture_output=True, text=True, cwd=ROOT)
        mm = re.search(r"peak_alloc ([0-9.]+) GiB reserved ([0-9.]+) GiB", p.stdout)
        r = {"micro": mc_, "for_micro": mc, "L": L, "rc": p.returncode, "peak_alloc": float(mm.group(1)) if mm else None, "reserved": float(mm.group(2)) if mm else None,
             "oom": "OutOfMemoryError" in p.stderr}
        rec["checks"].append(r); print("GATE", json.dumps(r), flush=True)
        if not mm or r["reserved"] > total - MARGIN:
            ok = False; break
    if ok:
        chosen = mc; break
rec["chosen_micro"] = chosen
json.dump(rec, open(os.path.join(a.out, "micro_gate.json"), "w", newline="\n"), indent=1)
assert chosen is not None, "no micro in (1, 2, 4) fits"
if not a.gate_only:
    sys.exit(subprocess.call([sys.executable, "-u", "-m", "boxes.fsq120", *rest, "--micro", str(chosen), "--device", a.device, "--out", a.out], cwd=ROOT))
