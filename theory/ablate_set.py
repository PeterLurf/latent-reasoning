"""THEORY §24 manipulation check: evaluate set-memory (A6) runs with the set write ablated (s_gate -> 0).

Same evaluators as theory/eval_arch.py (competence + length_gen), with the loader patched to zero the gate after
loading. Writes theory/results/setmem_ablation.json: final |gate|, delayed bal-acc and d8 (far 8 at K' = 12) ablated.
    python theory/ablate_set.py theory/runs/pg_g_s900 ... --device cuda:0
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_arch as EA  # noqa: E402  (patches the latent evaluators' loaders to load_arch)

OUT = "theory/results/setmem_ablation.json"


def ablated_loader(run, device):
    m, c = EA.load_arch(run, device)
    with torch.no_grad():
        m.s_gate.zero_()
    return m, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for run in a.runs:
        name = os.path.basename(run.rstrip("/\\"))
        if name in res or not os.path.exists(os.path.join(run, "model.pt")):
            continue
        m, _ = EA.load_arch(run, "cpu")
        gate = float(m.s_gate)
        EA.C.load = EA.CM.load = EA.LG.load = ablated_loader
        comp = EA.C.evaluate_run(run, 4096, device=a.device)
        K = comp["config"]["K"]
        L = EA.LG.evaluate(run, a.device)
        EA.C.load = EA.CM.load = EA.LG.load = EA.load_arch
        row = next(x for x in L["rows"] if x["K'"] == 12)
        res[name] = {"gate": gate, "delayed_ablated": comp["budgets"][K]["delayed"]["bal_acc"][0],
                     "d8_ablated": row["far"]["8"]}
        json.dump(res, open(a.out, "w", newline="\n"), indent=1)
        print(name, res[name], flush=True)


if __name__ == "__main__":
    main()
