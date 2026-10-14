"""§201 clarification 201-3 (Theory, 0d4646b7): the resolution test for a model that failed 201-1's bf16 re-check. halt173.halt_fields itself in FP32
(amp None; halt173.load_model, DM flags as halt173 leaves them; the first 1,000 test items per n in halt173.NEST, T = 2K) on the SAME card class as
§201 T1's fp32 CLEAN (the 2070), compared with that CLEAN's acc_by_n_t over every cell: gates.fp32_crosscode_max_items = max |k1 - k2| (an integer
item count) written into <run>/eval_s201_freeze.json; <= 2 -> RESOLVED-NUMERICAL (readings released, flagged), > 2 -> CODE-MISMATCH (withheld).
    python scratch/s201_fp32cross.py <run> [device, default cuda:2]   -> <run>/eval_s201_fp32cross.json"""
import json
import os
import sys

ROOT = "D:/Latent"; sys.path.insert(0, ROOT); os.chdir(ROOT)
os.environ.setdefault("BOXES_NB", "5")
import torch  # noqa: E402

import boxes.deltamem as DM  # noqa: E402
import boxes.halt173 as H  # noqa: E402


def main():
    run = sys.argv[1]; dev = sys.argv[2] if len(sys.argv) > 2 else "cuda:2"; rd = os.path.join(ROOT, "boxes", "runs_fsq", run)
    kind, m, a = H.load_model(rd, dev); flags = {"FAST_ADD": bool(DM.FAST_ADD), "CHUNK_DELTA": int(DM.CHUNK_DELTA)}
    _, _, by_t, nn = H.halt_fields(kind, m, a, dev, None, 1000, 250)
    fp_ = os.path.join(rd, "eval_s201_freeze.json"); fj = json.load(open(fp_)); cl = fj["arms"]["CLEAN"]["acc_by_n_t"]
    cells = [(n, t, abs(round(by_t[n][t] * nn[n]) - round(cl[n][t] * nn[n]))) for n in by_t for t in range(len(by_t[n])) if n in cl]
    mx = max(c[2] for c in cells); verdict = "RESOLVED-NUMERICAL" if mx <= 2 else "CODE-MISMATCH"
    out = {"halt173_fp32_acc_by_t": by_t, "fp32_crosscode_max_items": int(mx), "verdict": verdict, "cells_gt_0": [c for c in cells if c[2] > 0],
           "card": torch.cuda.get_device_name(dev), "dm_flags_halt173": flags, "bs": 250, "s201_t1_card": fj.get("card"),
           "def": "clarification 201-3: halt173.halt_fields in fp32 vs §201 T1 fp32 CLEAN acc_by_n_t, same items, max |k1 - k2| over all cells (n in NEST, t 0..2K)"}
    json.dump(out, open(os.path.join(rd, "eval_s201_fp32cross.json"), "w", newline="\n"), indent=1)
    fj["gates"].update({"fp32_crosscode_max_items": int(mx), "fp32_crosscode_verdict": verdict}); json.dump(fj, open(fp_, "w", newline="\n"), indent=1)
    print(run, "fp32 cross-code: max %d items -> %s" % (mx, verdict), flags, out["card"], flush=True)


if __name__ == "__main__":
    main()
