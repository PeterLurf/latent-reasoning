"""THEORY §155 / §159 compute (C3): Meter FLOPs and sequential passes per item for multpair's DeltaMem-R digit-pair chain, in the §98 / §152 convention
(boxes.core.Meter: each segment a fresh full 12-block pass, KV within the segment, full-vocabulary LM-head rows), plus the DeltaMem branch:
2 x branch params per processed token, and per head-token the read on the pass-start state (2 DH^2, all 12 heads) and the writes (additive 2 DH^2, delta 8 DH^2).
Chain for an n x n problem: n^2 pair segments [EOT, a_i, b_j, flag] (4 tokens) + the flush [EOT, " ="] (2 tokens); 2n emissions (LM-head rows).
Branch params are counted from the run's own model.pt (keys under "mem."). Writes theory/results/compute_s155.json.
    python theory/compute_s155.py s155_r_4x4_s1 [s159_r_5x5_s1 ...]
"""
import os
os.environ["BOXES_NB"] = "5"
import json  # noqa: E402
import sys  # noqa: E402

import torch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from boxes import s5  # noqa: E402
from boxes import core as bc  # noqa: E402
import boxes.deltamem as DM  # noqa: E402

TCM = {"4x4": (15459259392.0, 57), "5x5": (23314827264.0, 87)}                # §98 m98_tc_{4x4,5x5}_s1
ICOT = {"4x4": (4926282240.0, 13), "5x5": (5764879872.0, 15)}                 # §98 icotsi_{4x4,5x5}


def main():
    g = s5.load_gpt2("cpu"); res = {}
    for d in sys.argv[1:]:
        run = os.path.join(ROOT, "boxes", "runs_fsq", d); a = json.load(open(os.path.join(run, "args.json"))); task = a["task"]; n = int(task[0])
        sd = torch.load(os.path.join(run, "model.pt"), map_location="cpu")
        nb = sum(v.numel() for k, v in sd.items() if k.startswith("mem.") and not k.split(".")[-1].startswith("s_") and v.dtype.is_floating_point and v.dim() > 0)
        mt = bc.Meter(g); rows = 0
        for _ in range(n * n):                                               # pair segments: 4 tokens each; an emission row only at a column's last pair
            mt.step(4, 0, 0)
        mt.step(2, 0, 0)                                                     # the flush
        rows = 2 * n; mt.flops += 2 * g.D * 50257 * rows                     # 2n emissions, full-vocabulary rows (counted once each)
        toks = 4 * n * n + 2; D2 = DM.DH * DM.DH
        fb = 2 * nb * toks + ((DM.H_ADD + DM.H_DEL) * 2 * D2 + DM.H_ADD * 2 * D2 + DM.H_DEL * 8 * D2) * toks
        tot = mt.flops + fb; passes = n * n + 1
        res[d] = {"task": task, "flops": tot, "flops_trunk": mt.flops, "flops_branch": float(fb), "sequential_passes": passes, "lm_head_rows": rows,
                  "tokens_processed": toks, "branch_params": nb, "flops_vs_tcm": tot / TCM[task][0], "passes_vs_tcm": passes / TCM[task][1],
                  "flops_vs_icotsi": tot / ICOT[task][0], "passes_vs_icotsi": passes / ICOT[task][1],
                  "convention": "§98 / §152: boxes.core.Meter, each segment a fresh full pass, full-vocabulary LM-head rows; branch 2 x params per token + head-token ops"}
        print(d, json.dumps({k: res[d][k] for k in ("flops", "sequential_passes", "flops_vs_tcm", "passes_vs_tcm", "flops_vs_icotsi")}))
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "compute_s155.json"), "w", newline=chr(10)), indent=1)


if __name__ == "__main__":
    main()
