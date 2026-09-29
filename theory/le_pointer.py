"""THEORY §25 (Prop. 9, LE1–LE3, LE-R): measurement only — pointer trajectory of each listed run, no labels read.

Frozen measurement: analyse_pilotA.ptr_traj (unchanged) on rollout.probe(512); p[i] = pointer depth of thought h_i,
i = 0..11 (logit >= 1, else 0). Markov-architecture runs use the rollout block on their Markov layers, as in
analyse_pilotA. Models load from model.pt only (theory/runs via eval_arch.load_arch; runs/ via rollout.load).
Labels (competence, extrapolation) are joined later by the scoring step, after the owning session has scored the runs.
    python theory/le_pointer.py theory/runs/pc_k5_s790 runs/r3_s403 ...
"""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.set_num_threads(int(os.environ.get("THEORY_THREADS", "3")))
from analyse_pilotA import ptr_traj  # noqa: E402
from eval_arch import load_arch  # noqa: E402
from rollout import load, probe  # noqa: E402

OUT = "theory/results/le_pointer.json"


def main(paths):
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    g, br, E = probe(512)
    for path in paths:
        path = path.rstrip("/\\")
        name = os.path.basename(path)
        if name in res or not os.path.exists(os.path.join(path, "model.pt")):
            continue
        if path.replace("\\", "/").startswith("theory/runs"):
            m, c = load_arch(path, "cpu")
        else:
            m, c = load(path, "cpu")
        if hasattr(m, "s_gate"):            # set-memory models: the rollout harness does not reproduce them
            print(name, "skipped (setmem)", flush=True)
            continue
        ml = getattr(m, "markov_layers", ())
        kw = {"block_from": 1, "block_layers": ml} if ml else {}
        with torch.no_grad():
            p, a, head = ptr_traj(m, c["K"], g, br, E, kw)
        res[name] = {"path": path, "K": c["K"], "depth_cap": c.get("depth_cap"), "ptr": p, "amp": a, "head": head,
                     "pK1": p[c["K"] - 1]}
        json.dump(res, open(OUT, "w", newline="\n"), indent=1)
        print(name, "K", c["K"], "cap", c.get("depth_cap"), "p[K-1]", p[c["K"] - 1], "ptr", " ".join(map(str, p)), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
