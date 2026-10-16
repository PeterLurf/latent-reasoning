"""Raw (VB slot) span needed to hold the handle's radius-2 COMPACTED neighbourhood (5 alive slots centred on the operator); share of test states
(n 9..16) whose span exceeds the training maximum / training p99 (n 1..8)."""
import sys, json
sys.path.insert(0, "D:/Latent"); sys.path.insert(0, __import__("os").path.dirname(__file__))
import numpy as np
import boxes.nested170 as N
from local_rule_cov import raw_states, handle_index


def spans(root):
    out = []
    for z in raw_states(root)[:-1]:
        h = handle_index(z)
        alive = [i for i, s in enumerate(z) if s != "V"]; j = alive.index(h)
        lo, hi = alive[max(0, j - 2)], alive[min(len(alive) - 1, j + 2)]
        out.append(hi - lo + 1)
    return out


rng = np.random.default_rng(31400)
tr = [s for _ in range(40000) for s in spans(N.generate(rng, int(rng.integers(1, 9)))[0])]
mx, p99 = max(tr), float(np.percentile(tr, 99))
res = {"train": {"mean": round(float(np.mean(tr)), 2), "p99": p99, "max": mx}}
for n in (9, 10, 12, 14, 16):
    te = [s for _ in range(1000) for s in spans(N.generate(rng, n)[0])]
    res[n] = {"mean": round(float(np.mean(te)), 2), "p99": float(np.percentile(te, 99)), "max": int(max(te)),
              "share_gt_train_max": round(float(np.mean(np.array(te) > mx)), 4), "share_gt_train_p99": round(float(np.mean(np.array(te) > p99)), 4)}
print(json.dumps(res)); json.dump(res, open("raw_span.json", "w"), indent=1)
