"""§262 local gate: boxes/ask_render.py's s1_sentence / ask_of are the ST_v1 originals (docs/research_steptold_2026-10-24/st_ask_name_local.py) byte for byte, and give
the same (class, X, v, text, fallback) on 4,000 training-stream items (d 1-4, half of d >= 2 permuted by perm_render.permuted) -> scratch/gate_s269/ask_equiv.json.
    python -B scratch/s269_build/gate_ask_equiv.py"""
import importlib.util
import json
import os
import sys

import numpy as np

ROOT = "D:/Latent"; sys.path.insert(0, ROOT); os.chdir(ROOT)
import boxes.ask_render as AR  # noqa: E402
import boxes.ow_words as OW  # noqa: E402
import boxes.perm_render as PR  # noqa: E402
from boxes.inject_ow import OWKW  # noqa: E402

P = "docs/research_steptold_2026-10-24/st_ask_name_local.py"; src = open(P, encoding="utf-8").read(); new = open("boxes/ask_render.py", encoding="utf-8").read()
seg = src[src.index("def s1_sentence(it):"):src.index("def tok_askname(")].rstrip() + "\n"
sp = importlib.util.spec_from_file_location("st_ask_name_local", P); SA = importlib.util.module_from_spec(sp); sp.loader.exec_module(SA)
s = OW.Stream(OW.SEED_TRAIN + 7, set(), frames=OW.FRAME_TRAIN, mul_s_p=0, **OWKW); n = same = 0; cls = {}
for k in range(4000):
    d = 1 + k % 4; it = s.draw(d)
    if d >= 2 and k % 2:
        it, _, _ = PR.permuted(it, np.random.default_rng((1, 99, k)))
    a, b = AR.ask_of(it), SA.ask_of(it); n += 1; same += a == b; cls[a[0]] = cls.get(a[0], 0) + 1
o = {"source_identical": new.endswith(seg) and seg in new, "n": n, "same": same, "classes": dict(sorted(cls.items()))}
o["pass"] = o["source_identical"] and same == n and set(cls) == {"i", "ii", "iii", "iv"}
os.makedirs("scratch/gate_s269", exist_ok=True); json.dump(o, open("scratch/gate_s269/ask_equiv.json", "w", newline="\n"), indent=1); print("ask_equiv", o)
