"""Recompute the §67b mechanism tests (with the reachable-code irrelevant substitution) on fetched D1b models and merge
them into eval.json["mechanism"].   python -m c1.remech c1/runs/p67b_d1b_s<k> [...] [--device cuda:0]"""
import argparse
import json
import os

import torch

import c1.core as cc
from c1.train import build, mechanism

ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+"); ap.add_argument("--device", default="cuda:0")
a = ap.parse_args()
for r in a.runs:
    args = json.load(open(os.path.join(r, "args.json"))); cc.M = args["M"]
    m = build(args["arm"], a.device, args["seed"])
    if args.get("head_norm") == "ln":
        m.ln_heads()
    if args.get("readout") == "bil":
        m.bil_readout()
    m.load_state_dict(torch.load(os.path.join(r, "model.pt"), map_location=a.device)); m.eval()
    mech = mechanism(m, a.device)
    e = json.load(open(os.path.join(r, "eval.json"))); e["mechanism"] = mech; e["mechanism_recomputed_locally"] = True
    json.dump(e, open(os.path.join(r, "eval.json"), "w", newline="\n"), indent=1)
    print(r, mech, flush=True)
