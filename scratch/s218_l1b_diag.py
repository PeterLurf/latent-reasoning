"""§218 G-L1b diagnostic (descriptive, $0, CPU): on the 2k-step local model (scratch/gate_s218/l1b_run/model.pt), 64 fresh D 3-4 items (Stream 21150 / rng
21151, test + GATE excluded) at K = D + 2: SLVA's newest-version mass (head mean, real slots, passes t >= 2) and candidate accuracy / mean cand_ce
under c = 1e4 (CARRY), c = 19.86 (step 1050) and c = the run's final c. Asks whether the step-1000 jump (plain_cand_ce 0.82 -> 1.87 while answer_ce
fell) coincides with SLVA reading older versions once c leaves 1e4. -> scratch/gate_s218/g_l1b_diag.json"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "D:/Latent"); os.chdir("D:/Latent")
import boxes.langvb as LV  # noqa: E402
import boxes.prosqa_sym as PS  # noqa: E402
from boxes import s5  # noqa: E402

R = "scratch/gate_s218/l1b_run"; dev = "cpu"
m = LV.LangSLVA(s5.load_gpt2(dev)); m.load_state_dict(torch.load(os.path.join(R, "model.pt"), map_location=dev, weights_only=True)); m.eval()
c_final = [json.loads(l)["c"] for l in open(os.path.join(R, "log.jsonl")) if '"c"' in l][-1]
s = PS.Stream(21150, PS.exclusion_sigs()); rng = np.random.default_rng(21151); its = [LV.lvb_item(s.draw(int(D))) for D in rng.integers(3, 5, size=64)]
out = {"c_final": c_final, "items": 64}
for name, c in (("c_1e4", 1e4), ("c_19.86", 19.86), ("c_final", c_final)):
    m.c_anneal.fill_(c); ok = []; cc = []; mass = []
    for i in range(0, 64, 8):
        sub = its[i:i + 8]; b = LV.lvb_batch(sub, dev); x, pad, lens, si, sp, dR, dP = b
        for D in (3, 4):
            rows = [q for q, it in enumerate(sub) if it["D"] == D]
            if not rows:
                continue
            bb = LV.lvb_batch([sub[q] for q in rows], dev); m.slva_log = []
            with torch.no_grad():
                sR, sP = LV.cand_scores(m, [sub[q] for q in rows], D + 2, dev, lambda: torch.autocast("cuda", enabled=False))
            ok += (sR > sP).long().tolist(); cc += (-torch.log_softmax(torch.stack([sR, sP], 1), 1)[:, 0]).tolist()
            spb = bb[4]
            for t, nw in m.slva_log:
                if t >= 2:
                    mass.append(float((nw * spb).sum() / spb.sum()))
            m.slva_log = None
    out[name] = {"c": c, "acc": float(np.mean(ok)), "cand_ce": float(np.mean(cc)), "slva_newest_mass": float(np.mean(mass)) if mass else None,
                 "slva_offnewest_mass": (1 - float(np.mean(mass))) if mass else None}
    print(name, out[name], flush=True)
json.dump(out, open("scratch/gate_s218/g_l1b_diag.json", "w", newline="\n"), indent=1)
