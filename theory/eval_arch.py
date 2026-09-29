"""Evaluate a theory-stream run with the project's own measures (loaders swapped for the run's architecture).

Writes <run>/eval.json with the PROTOCOL5/6 definitions:
  final_stage, delayed (bal-acc at K, confirm split, n 4096), matched5/6 (competence_matched), gate
  (delayed >= 0.85 and matched5 >= 0.60 and final stage full), length_gen rows (K' 0..16), d7_K12 with CI and
  extrap_K12 (d7 >= 0.60 and CI lower bound > 0.55), plus far8_10_K12 (mean of distances 8-10 at K' = 12).
    python theory/eval_arch.py theory/runs/mk_s0 [--device cuda:0]
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import latent.competence as C  # noqa: E402
import latent.competence_matched as CM  # noqa: E402
import latent.length_gen as LG  # noqa: E402
from latent.graphs import set_shapes  # noqa: E402
from arch import build  # noqa: E402


def load_arch(run, device):
    ck = torch.load(os.path.join(run, "model.pt"), map_location=device, weights_only=False)
    c = ck["config"]
    ap = os.path.join(run, "arch.json")
    A = json.load(open(ap)) if os.path.exists(ap) else {"arch": "coconut", "overrun": 0}
    m = build(A["arch"], d=c["d"], n_layers=c["layers"], n_heads=c["heads"], k_max=max(c["K"] + A.get("overrun", 0), 8),
              mode=c["mode"], emb=c.get("emb", "slots"), d_id=c.get("d_id", 64), init=c.get("init", "torch"),
              feedback=c.get("feedback", "post_ln"),
              step_emb=(not c.get("no_step_emb", False)) or bool(A.get("periodic_step"))).to(device)
    set_shapes(c.get("shapes", "default"), c.get("p_fork", 0.5))
    m.load_state_dict(ck["model"])
    if A.get("periodic_step"):
        from arch import add_periodic_step
        add_periodic_step(m, A["periodic_step"])
    return m.eval(), c


C.load = CM.load = LG.load = load_arch


def final_stage(run):
    f = os.path.join(run, "log.jsonl")
    if not os.path.exists(f):
        return None
    adv = [json.loads(l) for l in open(f) if "curriculum_advance_to" in l]
    return "full" if adv and adv[-1]["curriculum_advance_to"] is None else (adv[-1]["curriculum_advance_to"] if adv else 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    out = {"final_stage": final_stage(a.run)}
    comp = C.evaluate_run(a.run, 4096, device=a.device)
    K = comp["config"]["K"]
    out["delayed"] = comp["budgets"][K]["delayed"]["bal_acc"][0]
    out["upfront"] = comp["budgets"][K]["upfront"]["bal_acc"][0]
    mt = CM.matched(a.run, a.device)
    out["matched"] = {str(k): v["bal_acc"] for k, v in mt.items()}
    out["matched5"], out["matched6"] = mt[5]["bal_acc"], mt[6]["bal_acc"]
    out["gate"] = bool(out["delayed"] >= 0.85 and out["matched5"] >= 0.60 and out["final_stage"] == "full")
    L = LG.evaluate(a.run, a.device)
    out["length_gen"] = L
    row = next(x for x in L["rows"] if x["K'"] == 12)
    ci = row["far_ci"]["7"]
    out["d7_K12"], out["d7_K12_ci"] = row["far"]["7"], ci
    out["extrap_K12"] = bool(row["far"]["7"] >= 0.60 and ci[0] > 0.55)
    out["far8_10_K12"] = sum(row["far"][str(d)] for d in (8, 9, 10)) / 3
    json.dump(out, open(os.path.join(a.run, "eval.json"), "w"), indent=1)
    print("EVAL", a.run, "stage", out["final_stage"], "delayed %.3f m5 %.3f gate %s d7 %.3f extrap %s far8-10 %.3f" % (
        out["delayed"], out["matched5"], out["gate"], out["d7_K12"], out["extrap_K12"], out["far8_10_K12"]), flush=True)


if __name__ == "__main__":
    main()
