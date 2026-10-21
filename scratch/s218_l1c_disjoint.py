"""Note 218-22 (ii): the re-run G-L1c reference model's 256 training items must be disjoint from A1's eval items (the SYM test set) - signature check
against prosqa_sym.test_set (and the GATE set, i.e. excl_sigs), recorded -> scratch/gate_s218/g_l1c_model/disjoint.json. Regenerates the items exactly as
scratch/gate_s218_l1c.py / s218_l1c_save.py do (Stream 21160 with exclusion_sigs, D from np rng 21161 U{3..4}, 256 items)."""
import json
import sys

sys.path.insert(0, "D:/Latent")
import numpy as np  # noqa: E402

import boxes.prosqa_sym as PS  # noqa: E402
import boxes.langvb as LV  # noqa: E402

EX = PS.exclusion_sigs(); s = PS.Stream(21160, EX); rD = np.random.default_rng(21161)
raw = [s.draw(int(D)) for D in rD.integers(3, 5, size=256)]
items = [LV.lvb_item(r) for r in raw]
sha = PS.set_sha([{"text": str(it["ids"])} for it in items])
sig = [PS.sig_of(r) for r in raw]
test = PS.fixed_sigs(PS.SEED_TEST, 1000, range(3, 17)); gate = PS.fixed_sigs(PS.SEED_GATE, 2000, range(9, 13))
res = {"items_sha": sha, "items_sha_matches_7973da5b": sha.startswith("7973da5b"), "n_items": len(items), "n_distinct_sigs": len(set(sig)),
       "test_sigs": len(test), "gate_sigs": len(gate), "hits_test": sum(x in test for x in sig), "hits_gate": sum(x in gate for x in sig),
       "def": __doc__}
res["disjoint"] = bool(res["hits_test"] == 0 and res["hits_gate"] == 0)
import os
os.makedirs("D:/Latent/scratch/gate_s218/g_l1c_model", exist_ok=True)
json.dump(res, open("D:/Latent/scratch/gate_s218/g_l1c_model/disjoint.json", "w", newline="\n"), indent=1); print(res)
