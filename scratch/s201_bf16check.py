"""§201 clarification 201-1 (Theory): when a model's fp32 T1 CLEAN acc_by_t differs from the bf16 eval_s173_halt by > 0.005, a bf16 CLEAN re-check on
the same items: halt173.halt_fields itself (the s173 computation: halt173.load_model, DM flags as halt173 leaves them, bf16 autocast, T = 2K, the
first 1,000 test items per n in halt173.NEST) on an Ampere card (eval_s173_halt ran on the 3070). Passes if its acc_by_t matches eval_s173_halt
within 0.005 (then the fp32 - bf16 differences are descriptive). -> <run>/eval_s201_bf16check.json
    python scratch/s201_bf16check.py <run> [device, default cuda:1 = the 3070]"""
import json
import os
import sys

ROOT = "D:/Latent"; sys.path.insert(0, ROOT); os.chdir(ROOT)
os.environ.setdefault("BOXES_NB", "5")
import torch  # noqa: E402

import boxes.deltamem as DM  # noqa: E402
import boxes.halt173 as H  # noqa: E402


def main():
    run = sys.argv[1]; dev = sys.argv[2] if len(sys.argv) > 2 else "cuda:1"; rd = os.path.join(ROOT, "boxes", "runs_fsq", run)
    kind, m, a = H.load_model(rd, dev); flags = {"FAST_ADD": bool(DM.FAST_ADD), "CHUNK_DELTA": int(DM.CHUNK_DELTA)}
    _, _, by_t, nn = H.halt_fields(kind, m, a, dev, torch.bfloat16, 1000, 250)
    s173 = json.load(open(os.path.join(rd, "eval_s173_halt.json")))["acc_by_t"]; fz = json.load(open(os.path.join(rd, "eval_s201_freeze.json")))["arms"]["CLEAN"]["acc_by_n_t"]
    d_s173 = max(abs(by_t[n][t] - s173[n][t]) for n in s173 for t in range(len(s173[n])))
    d_fp32 = {n: [round(fz[n][t] - by_t[n][t], 4) for t in range(len(by_t[n]))] for n in by_t if n in fz}
    out = {"bf16_acc_by_t": by_t, "maxdiff_bf16_vs_s173": d_s173, "pass": d_s173 <= 0.005, "fp32_minus_bf16_by_t": d_fp32,
           "maxabs_fp32_minus_bf16": max(abs(v) for r in d_fp32.values() for v in r), "card": torch.cuda.get_device_name(dev), "dm_flags": flags,
           "def": "clarification 201-1: halt173.halt_fields at bf16 on the same items; pass iff within 0.005 of eval_s173_halt acc_by_t"}
    json.dump(out, open(os.path.join(rd, "eval_s201_bf16check.json"), "w", newline="\n"), indent=1)
    print(run, "bf16 check: maxdiff vs s173 %.4f pass %s; max |fp32 - bf16| %.4f" % (d_s173, out["pass"], out["maxabs_fp32_minus_bf16"]), flags, flush=True)


if __name__ == "__main__":
    main()
