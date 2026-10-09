"""THEORY §177 (a) post hoc (descriptive; Theory 2026-10-09 ~18:20): do the learned pass-index vectors keep passes distinct in DIRECTION?
(the norm of g * s~(t) is near-constant by construction - Audit). Uses inject169's own Inject.pass_vec on the saved pgain / pscale.
  cos(pass_vec(1), pass_vec(t)) and the raw-sinusoid reference cos(s~(1), s~(t)) (gain = 1), t = 2, 4, 8, 16 (+ 32, 64 for reference)
    python scratch/s177_index_dir.py boxes/runs_fsq/s177_tidx_s1      -> <run>/eval_s177_index_dir.json"""
import json
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import torch  # noqa: E402

from boxes.inject169 import Inject  # noqa: E402


def main():
    run = sys.argv[1]; sd = torch.load(os.path.join(run, "model.pt"), map_location="cpu", weights_only=False)
    g = sd["pgain"].float(); sc = sd["pscale"].float()
    learned = types.SimpleNamespace(pgain=g, pscale=sc); raw = types.SimpleNamespace(pgain=torch.ones_like(g), pscale=sc)
    cos = lambda u, v: float(torch.nn.functional.cosine_similarity(u, v, dim=0))  # noqa: E731
    T = (2, 4, 8, 16, 32, 64); v1, r1 = Inject.pass_vec(learned, 1), Inject.pass_vec(raw, 1)
    out = {"run": run, "t": list(T), "cos_learned_1_t": {str(t): cos(v1, Inject.pass_vec(learned, t)) for t in T},
           "cos_raw_sinusoid_1_t": {str(t): cos(r1, Inject.pass_vec(raw, t)) for t in T},
           "norm_learned_by_t": {str(t): float(Inject.pass_vec(learned, t).norm()) for t in (1,) + T},
           "gain": {"norm": float(g.norm()), "mean": float(g.mean()), "abs_mean": float(g.abs().mean()), "frac_neg": float((g < 0).float().mean()),
                    "max_abs": float(g.abs().max())},
           "pscale": float(sc), "note": "descriptive (Theory ~18:20 2026-10-09); pass_vec = g * s~(t) / ||s~(t)|| * pscale (inject169); raw = gain 1"}
    json.dump(out, open(os.path.join(run, "eval_s177_index_dir.json"), "w", newline="\n"), indent=1); print(json.dumps(out))


if __name__ == "__main__":
    main()
