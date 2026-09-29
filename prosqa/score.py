"""Score PROTOCOL6 addendum D (ProsQA transfer) from the pod's result files.

    python prosqa/score.py prosqa/res      -> prosqa/res/protocol6_prosqa.json (+ printed summary)
"""
import json
import os
import sys

import numpy as np

R = sys.argv[1] if len(sys.argv) > 1 else "prosqa/res"
J = lambda f: json.load(open(os.path.join(R, f))) if os.path.exists(os.path.join(R, f)) else None
HOPS = (1, 2, 3)


def boot_ci(x, n=1000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


tv, tt, bl, kp = J("trace_valid_e50.json"), J("trace_test_e50.json"), J("block_test_e50.json"), J("kprime_test_e50.json")
out = {}
# T0 competence
acc6 = kp["6"]["acc"] if kp else (bl["base_acc"] if bl else None)
out["T0"] = {"acc_K6_test": acc6, "competent": bool(acc6 is not None and acc6 >= 0.90)}
# frontier layer and clock (valid)
L = int(np.argmax(tv["frontier_score"]))
kv = [tv["path_clock"][L][h - 1] for h in HOPS]
out["layer"] = {"L_star": L, "frontier_score": tv["frontier_score"], "valid_clock": kv}
# T1 sweep (test)
kt = [tt["path_clock"][L][h - 1] for h in HOPS]
ex = [tt["excess"][L][kv[i] - 1][h - 1] for i, h in enumerate(HOPS)]
inc = kv[0] < kv[1] < kv[2]
out["T1"] = {"test_clock": kt, "valid_clock_increasing": inc, "test_equals_valid": kt == kv, "excess_at_clock": ex,
             "reading": "sweep present" if inc and kt == kv and all(e > 0 for e in ex) else "absent"}
# T2 step-locked use (test)
rows = bl["per_example"]
per_h = {}
ok_all = True
for i, h in enumerate(HOPS):
    k = kv[i]
    diag = [d for r in rows for kind, kk, hh, d in r["d"] if kind == "path" and kk == k and hh == h]
    off = {kk: float(np.mean([d for r in rows for kind, k2, hh, d in r["d"] if kind == "path" and k2 == kk and hh == h]))
           for kk in range(1, bl["K"] + 1) if abs(kk - k) >= 2}
    ctrl = [d for r in rows for kind, kk, hh, d in r["d"] if kind == "ctrl" and kk == k and hh == h]
    dm, ci = float(np.mean(diag)), boot_ci(diag)
    a = dm < 0 and ci[1] < 0
    b = all(abs(v) <= 0.25 * abs(dm) for v in off.values())
    c = bool(ctrl) and abs(float(np.mean(ctrl))) <= 0.25 * abs(dm)
    per_h[h] = {"k_star": k, "diag_mean": dm, "diag_ci": ci, "n": len(diag), "off_diag": off,
                "ctrl_mean": float(np.mean(ctrl)) if ctrl else None, "n_ctrl": len(ctrl), "a": a, "b": b, "c": c,
                "delta_all": bl["delta_all"][h - 1]}
    ok_all &= a and b and c
not_needed = all(abs(bl["delta_all"][h - 1]) < 0.1 for h in HOPS)
out["T2"] = {"per_hop": per_h, "delta_path_matrix": bl["delta_path"], "delta_all": bl["delta_all"],
             "reading": "step-locked" if ok_all else "thoughts not needed" if not_needed else "not step-locked"}
# T3 descriptive
out["T3"] = {"end_activity_L_star": {f"e{e}": (J(f"trace_test_e{e}.json") or {}).get("end_activity", [None] * 12)[L]
                                     for e in (30, 35, 40, 45, 50)},
             "end_activity_all_layers_e50": tt["end_activity"],
             "kprime": kp}
json.dump(out, open(os.path.join(R, "protocol6_prosqa.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "T3"} | {"T3_EA": out["T3"]["end_activity_L_star"]},
                 indent=1, default=float)[:6000])
