"""§198 amendment 198-2a (Theory / Audit): eval-time cost of a trained nested170 run - timed seconds per item at K' = K (think_k) with NO aux decoder,
on the n 3..8 test items, at the run's eval batch (args eval_bs), the run's eval dtype (bf16 autocast on Ampere+), after one warm-up batch; cuda-synced
wall time. Works for --arm inject (incl. --cache_lower / --mem_layer: eval is uncached as in nested170's eval) and --arm readslots.
    python -m scratch.eval_timing198 --run <run dir> [--device cuda:0] [--out <ignored marker dir>]   -> <run>/eval_timing.json"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
os.environ.setdefault("BOXES_NB", "5")
import torch  # noqa: E402

import boxes.deltamem as DM  # noqa: E402
import boxes.nested170 as N  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--out", default=""); ap.add_argument("--fname", default="eval_timing.json", help="output file name in the run dir (note 201-6: eval_timing_4060.json for the local same-card timing)")
    ap.add_argument("--no_load", action="store_true", help="smoke only: skip model.pt (random init)"); o = ap.parse_args(); dev = o.device
    a = json.load(open(os.path.join(o.run, "args.json"))); DM.FAST_ADD = bool(a.get("fast_add")); K = int(a["think_k"]); bs = int(a.get("eval_bs") or 250)
    g = N.make_trunk(a, dev); ML, HA, HD = N.mem_geometry(a)
    m = (N.ReadSlotsN(g, ML, "top", HA, HD, a.get("add_norm") or None, int(a.get("n_slots") or 8)) if a["arm"] == "readslots"
         else N.InjectN(g, ML, "top", HA, HD, a.get("add_norm") or None)).to(dev)
    m.nopass = bool(a.get("aux_nopass"))
    if not o.no_load:
        m.load_state_dict(torch.load(os.path.join(o.run, "model.pt"), map_location=dev, weights_only=True))
    m.eval(); amp = None if (not dev.startswith("cuda") or torch.cuda.get_device_capability(dev)[0] < 8) else torch.bfloat16
    TE = N.test_set(int(a.get("per_n") or 1000)); its = [it for it in TE if 3 <= it["n"] <= 8]

    @torch.no_grad()
    def answers(ch):
        x, msk = m.think_inputs(ch, dev)
        with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
            S = m.mem.zero_state(len(ch), dev)
            for _ in range(K):
                S = m.write(S, x, msk)
            lg = m.dec_logits(S, ch, dev)
        return lg[:, N.NUM_IDS].argmax(-1)
    answers(its[:bs])                                                       # warm-up
    if dev.startswith("cuda"):
        torch.cuda.synchronize(dev)
    t0 = time.time(); n_ = 0
    for i in range(0, len(its), bs):
        answers(its[i:i + bs]); n_ += len(its[i:i + bs])
    if dev.startswith("cuda"):
        torch.cuda.synchronize(dev)
    sec = time.time() - t0
    out = {"eval_timing": {"s_per_item": sec / n_, "batch": bs, "gpu": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu", "n_items": n_, "kprime": K,
                           "arm": a["arm"], "cache_lower_train": bool(a.get("cache_lower")), "mem_layer": int(a.get("mem_layer") or 6), "eval_dtype": str(amp), "total_sec": sec,
                           "def": "198-2a: wall seconds per item for the K' = K answer path (think K passes + the main decoder; no aux decoder), n 3..8 test items, the "
                                  "run's eval batch, after one warm-up batch, cuda-synced", "no_load": bool(o.no_load)}}
    json.dump(out, open(os.path.join(o.run, o.fname), "w", newline="\n"), indent=1); print("EVAL_TIMING", o.run, out["eval_timing"]["s_per_item"], flush=True)


if __name__ == "__main__":
    main()
