"""§271 Note 271-5 gate: on SO training batches (so_words.Stream SEED_TRAIN, d 1-4, 64 items each, tok_so cue index), for every (item, t <= m) the DIRECT-SI weight
inject_so.si_weight(it, t) times the loop's n_tok for that row - (y_a != -100).sum(1), with y_a built exactly as aux_logits does (nested170.pad(tg, -100), tg = aux[t - 1]
per item at pass t) - equals 1, i.e. the name token gets the loop's per-token share of CE_t. Also reports the n_tok range. -> scratch/gate_s271n1/g_siweight.json
    python -B scratch/n1_build/g_siweight.py"""
import json
import os
import sys

ROOT = "D:/Latent"; sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import boxes.so_words as SO  # noqa: E402
import boxes.inject_so as IS  # noqa: E402
from boxes.nested170 import pad  # noqa: E402
from boxes.so_tok import tok_so  # noqa: E402

st = SO.Stream(SO.SEED_TRAIN); bad = 0; n = 0; ntoks = []
for d in (1, 2, 3, 4):
    for _ in range(4):
        items = [tok_so(st.draw(d), "index") for _ in range(64)]
        for t in range(1, d + 1):
            y_, _m = pad([it["aux"][t - 1] for it in items], -100); y = torch.tensor(y_); nt = (y != -100).sum(1).tolist()
            for it, k in zip(items, nt):
                n += 1; ntoks.append(k); bad += abs(IS.si_weight(it, t) * k - 1.0) > 1e-12
out = {"pairs": n, "bad": bad, "n_tok_min": int(min(ntoks)), "n_tok_max": int(max(ntoks)), "n_tok_mean": float(np.mean(ntoks)), "pass": bad == 0 and n > 0}
os.makedirs("scratch/gate_s271n1", exist_ok=True); json.dump(out, open("scratch/gate_s271n1/g_siweight.json", "w", newline="\n"), indent=1); print(json.dumps(out))
sys.exit(0 if out["pass"] else 1)
