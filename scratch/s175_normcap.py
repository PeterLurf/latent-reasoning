"""THEORY proposal #4, NORM-CAP-GLOBAL (descriptive; the deployable version of NORM-CLAMP): s175_auxk8_s1,
M_0 = 0, t = 1..16, on the FULL test pool at n 3..8 (1,000 items per n; Audit's amendment). Arms: unclamped; perclamp (NORM-CLAMP's oracle: t > 8 rescales S_add to the item's own t = 8 norm); cap1 / cap1.5: at EVERY pass,
after the write, an item whose ||S_add|| exceeds C is rescaled to norm C (direction kept; delta untouched), C = 1.0 / 1.5 x the median t = 8 additive
norm from eval_memnorm.json. -> pooled accuracy per t per arm, accuracy by n at t = 8 and t = 16 per arm.
    python scratch/s175_normcap.py [--device cpu]   -> boxes/runs_fsq/s175_auxk8_s1/eval_normcap.json"""
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
    C8 = json.load(open(os.path.join(run, "eval_memnorm.json")))["by_t"]["8"]["add_norm_median"]; caps = {"cap1": C8, "cap1.5": 1.5 * C8}
    its = [it for it in test_set(a.get("per_n", 1000)) if 3 <= it["n"] <= 8]   # Audit / Theory amendment: the FULL test pool at n 3..8 (1,000 per n), not 256
    tgt = np.array([it["t"] for it in its]); nn = np.array([it["n"] for it in its]); ha = m.mem.h_add; arms = ("unclamped", "perclamp", "cap1", "cap1.5"); ok = {r: {} for r in arms}
    frac_capped = {r: {} for r in ("cap1", "cap1.5")}
    bs = int(os.environ.get("NORMCAP_BS", "64"))
    for i in range(0, len(its), bs):
        ch = its[i:i + bs]; x, msk = m.think_inputs(ch, dev)
        for arm in arms:
            S = m.mem.zero_state(len(ch), dev); n8 = None
            for t in range(1, 17):
                S = m.write(S, x, msk); cur = S[:, :ha].float().flatten(1).norm(dim=1)
                if t == 8:
                    n8 = cur.clone()
                f = None
                if arm == "perclamp" and t > 8:
                    f = n8 / cur.clamp_min(1e-12)
                elif arm in caps:
                    over = cur > caps[arm]; f = torch.where(over, caps[arm] / cur.clamp_min(1e-12), torch.ones_like(cur))
                    frac_capped[arm].setdefault(t, []).append(over.cpu().numpy())
                if f is not None:
                    S = torch.cat([S[:, :ha] * f.to(S.dtype)[:, None, None, None], S[:, ha:]], 1)
                pr = m.dec_logits(S, ch, dev)[:, NUM_IDS].argmax(-1).cpu().numpy(); ok[arm].setdefault(t, []).append(pr == tgt[i:i + len(ch)])
    acc = lambda r, t: float(np.concatenate(ok[r][t]).mean())   # noqa: E731
    res = {"by_t": {str(t): {r: acc(r, t) for r in arms} for t in range(1, 17)},
           "frac_capped_by_t": {r: {str(t): float(np.concatenate(v).mean()) for t, v in sorted(d.items())} for r, d in frac_capped.items()},
           "acc_by_n": {str(tt): {r: {str(n): float(np.concatenate(ok[r][tt])[nn == n].mean()) for n in range(3, 9)} for r in arms} for tt in (8, 16)},
           "caps": caps, "C8_source": "eval_memnorm.json by_t.8.add_norm_median", "run": run, "n_items": len(its), "items": "nested170.test_set n 3..8, all (1,000 per n)", "device": dev, "sec": round(time.time() - t0),
           "def": "cap: at every pass after the write, ||S_add|| > C -> S_add * C / ||S_add|| (per item; delta untouched); perclamp = NORM-CLAMP's per-item t = 8 oracle"}
    json.dump(res, open(os.path.join(run, "eval_normcap.json"), "w", newline="\n"), indent=1)
    for t, v in res["by_t"].items():
        print(t, {r: round(x, 3) for r, x in v.items()}, flush=True)
    print("by n", json.dumps(res["acc_by_n"]), flush=True)


if __name__ == "__main__":
    main()
