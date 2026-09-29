"""Re-run the curriculum gate evaluation on a run's latest checkpoint, with both eval chunk sizes."""
import sys

import numpy as np
import torch

import latent.train as T
from latent.competence import load

run, dev = sys.argv[1], sys.argv[2]
m, c = load(run, dev)
ck = torch.load(f"{run}/model.pt", map_location="cpu", weights_only=False)
print(run, "checkpoint step", ck["step"])
for d in (1, 2):
    ev = T.evaluate(m, c["K"], n=512, device=dev, max_bdepth=d, seed=int(ck["step"]))
    e = ev["delayed"]
    print(f"  gate eval max_bdepth={d}: delayed bal {e['bal_acc']:.3f} acc {e['acc']:.3f} by_depth", {k: round(v, 3) for k, v in e["by_depth"].items()},
          "| upfront bal", round(ev["upfront"]["bal_acc"], 3))
