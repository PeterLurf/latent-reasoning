"""THEORY (proposal #4 gate, descriptive): memory norms vs passes on s175_auxk8_s1. 256 test items at n 3..8 (nested170.test_set, rng 17501),
passes t = 1..16 from M_0 = 0; per t the median ||S_add|| (S[:, :h_add]) and ||S_delta|| (S[:, h_add:]) over items and the main decoder's pooled accuracy.
    python scratch/s175_memnorm.py [--device cpu]   -> boxes/runs_fsq/s175_auxk8_s1/eval_memnorm.json"""
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np  # noqa: E402
import torch  # noqa: E402

from boxes.halt173 import load_model  # noqa: E402
from boxes.nested170 import test_set, NUM_IDS  # noqa: E402


@torch.no_grad()
def main():
    dev = sys.argv[sys.argv.index("--device") + 1] if "--device" in sys.argv else "cpu"; torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "3"))); t0 = time.time()
    run = "boxes/runs_fsq/s175_auxk8_s1"; kind, m, a = load_model(run, dev); assert kind == "nested"
    pool = [it for it in test_set(a.get("per_n", 1000)) if 3 <= it["n"] <= 8]; its = [pool[i] for i in np.random.default_rng(17501).permutation(len(pool))[:256]]
    tgt = np.array([it["t"] for it in its]); ha = m.mem.h_add; out = {}
    for i in range(0, len(its), 64):
        ch = its[i:i + 64]; x, msk = m.think_inputs(ch, dev); S = m.mem.zero_state(len(ch), dev)
        for t in range(1, 17):
            S = m.write(S, x, msk); lg = m.dec_logits(S, ch, dev); pr = lg[:, NUM_IDS].argmax(-1).cpu().numpy()
            o_ = out.setdefault(t, {"add": [], "delta": [], "ok": []})
            o_["add"].append(S[:, :ha].float().flatten(1).norm(dim=1).cpu().numpy()); o_["delta"].append(S[:, ha:].float().flatten(1).norm(dim=1).cpu().numpy())
            o_["ok"].append(pr == tgt[i:i + len(ch)])
    res = {"by_t": {str(t): {"add_norm_median": float(np.median(np.concatenate(v["add"]))), "delta_norm_median": float(np.median(np.concatenate(v["delta"]))),
                             "acc_pooled": float(np.concatenate(v["ok"]).mean())} for t, v in sorted(out.items())},
           "run": run, "n_items": len(its), "items": "nested170.test_set n 3..8, rng 17501 permutation[:256]", "device": dev, "sec": round(time.time() - t0),
           "note": "Theory proposal #4 gate (descriptive): additive magnitude growth vs the K' accuracy drift"}
    json.dump(res, open(os.path.join(run, "eval_memnorm.json"), "w", newline="\n"), indent=1)
    for t, v in res["by_t"].items():
        print(t, round(v["add_norm_median"], 3), round(v["delta_norm_median"], 3), round(v["acc_pooled"], 3), flush=True)


if __name__ == "__main__":
    main()
