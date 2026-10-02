"""Re-run c1.train's evaluation on a finished run's saved model.pt (for runs whose evaluation crashed after training,
e.g. a CUDA OOM on a shared card). Same evaluate() as c1.train; the train record comes from log.jsonl.
    python -m c1.reeval c1/runs/<run> [...] [--device cuda:0]"""
import argparse
import json
import os

import torch

import c1.core as cc
import c1.train as ct

ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+"); ap.add_argument("--device", default="cuda:0")
a = ap.parse_args()
for r in a.runs:
    args = json.load(open(os.path.join(r, "args.json"))); cc.M = args.get("M", 6); ct.GPT2_INIT = args.get("gpt2_init", "pretrained")
    m = ct.build(args["arm"], a.device, args["seed"])
    if args.get("head_norm") == "ln" and hasattr(m, "ln_heads"):
        m.ln_heads()
    if args.get("readout") == "bil" and hasattr(m, "bil_readout"):
        m.bil_readout()
    if args.get("feat_norm") == "tablestd" and hasattr(m, "feat_norm"):
        m.feat_norm = "tablestd"
    m.load_state_dict(torch.load(os.path.join(r, "model.pt"), map_location=a.device)); m.eval()
    if hasattr(m, "_stats"):
        m._stats = None
    res = ct.evaluate(m, args["arm"], a.device, args["n_test"], args["n_null"], r)
    last = [json.loads(l) for l in open(os.path.join(r, "log.jsonl"))][-1]
    res["train"] = {"updates": last.get("update"), "examples": last.get("examples"), "gpt2_tokens": last.get("gpt2_tokens"),
                    "sec": last.get("sec")}
    res["reevaluated"] = "evaluation re-run from model.pt by c1/reeval.py (the original evaluation crashed after training)"
    json.dump(res, open(os.path.join(r, "eval.json"), "w", newline="\n"), indent=1)
    print(r, res["acc"], flush=True)
