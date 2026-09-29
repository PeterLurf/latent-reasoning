"""THEORY §23 PF1 and §25 LE4: transfer-curve restoring index for pilot runs (measurement only; labels joined later).

Frozen pieces, unchanged: sweep head = exp2's heads[0] (argmax of the attention excess over steps 1..K on probe(512),
computed by analyse_pilotA.ptr_traj), transfer curve = exp9_transfer.measure on probe(256), as in exp9's main().
Index z = log2(a_out(s=2) / a_out(s=1)) = log2(exp9 "sat"): 1 for a linear map, <= 0 for a clamped (restoring) one.
    python theory/transfer_pairs.py theory/runs/pd_c_s900 theory/runs/pf_j_s900 ...
"""
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.set_num_threads(int(os.environ.get("THEORY_THREADS", "3")))
from analyse_pilotA import ptr_traj  # noqa: E402
from eval_arch import load_arch  # noqa: E402
from exp9_transfer import measure  # noqa: E402
from rollout import probe  # noqa: E402

OUT = "theory/results/transfer_pilots.json"


def main(paths):
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    g5, br5, E5 = probe(512)
    g2, br2, E2 = probe(256)
    for path in paths:
        path = path.rstrip("/\\"); name = os.path.basename(path)
        if name in res or not os.path.exists(os.path.join(path, "model.pt")):
            continue
        m, c = load_arch(path, "cpu")
        if hasattr(m, "s_gate") or getattr(m, "markov_layers", ()):
            print(name, "skipped (architecture not reproduced by the rollout harness)", flush=True)
            continue
        K = c["K"]
        with torch.no_grad():
            _, _, head = ptr_traj(m, K, g5, br5, E5, {})
            o = measure(m, K, g2, br2, E2, head)
        o["head"] = head
        o["z"] = math.log2(o["sat"]) if o["sat"] > 0 else None
        res[name] = o
        json.dump(res, open(OUT, "w", newline="\n"), indent=1)
        print("%-14s head %d gain %.2f slope %.2f sat %.2f z %s | a_out %s" % (name, head, o["gain"], o["slope"], o["sat"],
              "%.2f" % o["z"] if o["z"] is not None else "-", " ".join("%.1f" % x for x in o["a_out"])), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
