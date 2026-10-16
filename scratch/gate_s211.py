"""§211 VERSION-BANK Stage 0 gates (Theory pin 6: diff E_t against the red team's reference). -> scratch/gate_s211/<gate>.json
    python scratch/gate_s211.py et      # boxes.vbank.recode_targets vs scratch/s211_ref_slot_targets.slot_targets on 100k items, n 1-16"""
import json
import os
import sys

sys.path.insert(0, "D:/Latent"); sys.path.insert(0, "D:/Latent/scratch"); os.chdir("D:/Latent")
import numpy as np  # noqa: E402

import boxes.nested170 as N  # noqa: E402
import boxes.vbank as VB  # noqa: E402


def gate_et(n_items=100_000, seed=20902, n_max=16):
    import s211_ref_slot_targets as R
    rng = np.random.default_rng(seed); bad = 0; vers = 0; first = None
    for i in range(n_items):
        root, _ = N.generate(rng, int(rng.integers(1, n_max + 1))); ref = R.slot_targets(root)
        mine = [["V" if c == VB.VOID else VB.CLS[c] for c in e] for e in VB.recode_targets(root)]
        vers += len(ref)
        if mine != ref:
            bad += 1; first = first if first is not None else i
    return {"items": n_items, "seed": seed, "n_range": [1, n_max], "versions": vers, "diffs": bad, "first_diff_item": first, "pass": bad == 0,
            "reference": "scratch/s211_ref_slot_targets.py (red team, sha256 ebf25d3b... of the source file)"}


if __name__ == "__main__":
    g = sys.argv[1]; r = {"et": gate_et}[g](); json.dump(r, open("scratch/gate_s211/%s.json" % g, "w", newline="\n"), indent=1); print(json.dumps(r))
