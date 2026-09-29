import json
import time

import latent.study as S

t = time.time()
P = {"seed": 5, "k_star_default": 1, "k_star_by_run": {}, "n_decode": 300, "n_per_width": 60,
     "widths": [2, 8],
     "conditions": [{"kind": "full", "direction": "remove"}, {"kind": "probe", "direction": "remove"},
                    {"kind": "full", "direction": "remove", "steps": [0, 3], "widths": [2]}],
     "controls": ["identity", "same_status"], "upfront_kinds": ["full"],
     "trace_widths": [4], "n_trace": 50}
json.dump(P, open("scratch/proto_smoke.json", "w"))
r = S.confirm("scratch", "scratch/proto_smoke.json", "cuda:1")
print({k: v for k, v in r["effects"].items() if "control" in k})
print(json.dumps(r["downstream"], indent=0)[:1500])
print("total", time.time() - t)
