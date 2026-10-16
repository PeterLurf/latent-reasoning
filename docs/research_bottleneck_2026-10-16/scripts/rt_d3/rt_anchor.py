"""Red-team D3 COPY-ANCHOR (CPU, 1 thread, no model): under exact decodes, which stored slots does the anchor rule ĉ_t(j) == ĉ_0(j) send back to V_0?
On the §170 test set (n 1..16), per (item, t): (a) the reduction's value slot (vbank_eval.value_slots) whose class equals its INPUT class
(a computed value anchored to the input token's V_0 - stale context by construction); (b) the share of real slots anchored; (c) the number of
DISTINCT stored classes per slot over the trajectory (how much version history the bank can hold at a slot under the anchor).
Usage: python rt_anchor.py <out.json>"""
import json, sys
from collections import defaultdict
sys.path.insert(0, "D:/Latent")
import numpy as np
import boxes.nested170 as N
import boxes.vbank as VB
import boxes.vbank_eval as VE

TE = N.test_set(1000)
by_n = defaultdict(lambda: {"red": 0, "val_anch": 0, "items_any": 0, "items": 0, "slots": 0, "anch_slots": 0, "distinct": []})
for it in TE:
    n = it["n"]; E = VB.recode_targets(it["root"]); E0 = E[0]; vs = VE.value_slots(it["root"]); b = by_n[n]; b["items"] += 1; anyv = False
    for t in range(1, n + 1):
        v = vs[t - 1]; b["red"] += 1
        if E[t][v] == E0[v]:
            b["val_anch"] += 1; anyv = True
        real = range(1, len(E0) - 1)
        b["slots"] += len(real); b["anch_slots"] += sum(E[t][j] == E0[j] for j in real)
    b["items_any"] += int(anyv)
    for j in range(1, len(E0) - 1):
        b["distinct"].append(len({tuple([E[t][j]]) for t in range(n + 1)}))
out = {str(n): {"value_slot_anchored_per_reduction": round(b["val_anch"] / b["red"], 4), "items_with_any": round(b["items_any"] / b["items"], 4),
               "share_slots_anchored": round(b["anch_slots"] / b["slots"], 4), "distinct_classes_per_slot_mean": round(float(np.mean(b["distinct"])), 3),
               "distinct_max": int(max(b["distinct"]))} for n, b in sorted(by_n.items())}
json.dump(out, open(sys.argv[1], "w", newline="\n"), indent=1); print(json.dumps(out, indent=0))
