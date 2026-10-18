"""§218 G-L9 (loader half): the A1 trainer's plain draw law over the full 24,000 steps x bs 64 at the final stage's D_max 8 (the costliest stage),
without the model: langvb.main's draw() order (rng = seed 1: integers(3, 9, size=64); stream prosqa_sym.Stream(22100) with the test + GATE signatures
excluded; lvb_item; rng_k = seed + 21100 for K). Logs draws/s every 1,000 steps, collisions, max T / max |S| -> scratch/gate_s218/g_l9_loader.json."""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "D:/Latent"); os.chdir("D:/Latent")
import boxes.langvb as LV  # noqa: E402
import boxes.prosqa_sym as PS  # noqa: E402

t0 = time.time(); EX = PS.exclusion_sigs(); t_ex = time.time() - t0
s = PS.Stream(22100, EX); rng = np.random.default_rng(1); rng_k = np.random.default_rng(1 + 21100); tmax = smax = 0; marks = []; t1 = time.time()
for step in range(1, 24001):
    its = [LV.lvb_item(s.draw(int(D))) for D in rng.integers(3, 9, size=64)]; Ks = [min(it["D"] + int(rng_k.integers(0, 5)), 12) for it in its]
    tmax = max(tmax, max(len(it["ids"]) for it in its)); smax = max(smax, max(len(it["slot_idx"]) for it in its))
    if step % 1000 == 0:
        marks.append({"step": step, "sec": round(time.time() - t1, 1), "sec_per_step": round((time.time() - t1) / step, 4), "collisions": len(s.collisions)})
        print(marks[-1], flush=True)
res = {"steps": 24000, "bs": 64, "D_law": "U{3..8} (final stage)", "exclusion_sec": round(t_ex, 1), "sec_total": round(time.time() - t1, 1),
       "sec_per_step": (time.time() - t1) / 24000, "collisions": len(s.collisions), "draws_by_D": dict(s.n), "t_max": tmax, "s_max": smax, "marks": marks,
       "pass": True, "note": "completes; the per-step loader cost is paid inside the trainer's loop (no prefetch)"}
json.dump(res, open("scratch/gate_s218/g_l9_loader.json", "w", newline="\n"), indent=1); print("DONE", res["sec_per_step"], res["collisions"], tmax, smax)
