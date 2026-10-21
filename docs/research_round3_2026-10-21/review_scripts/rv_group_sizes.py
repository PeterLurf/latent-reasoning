"""REVIEW_R3 check 3 ($0, CPU, read-only on s213_d3t_s1/model.pt): parameter count per group, with r3_param_drift.py's group() map
(imported from the draft's script), to size a matched random-freeze control. Writes rv_group_sizes.json beside this script."""
import json, os, sys
import torch
torch.set_num_threads(2)
sys.path.insert(0, "D:/Latent/docs/research_round3_2026-10-21")
import importlib.util
src = open("D:/Latent/docs/research_round3_2026-10-21/r3_param_drift.py").read()
g = {}; exec(src[:src.index("cache, out")], g)          # group() and constants only; no model loads beyond ours
sd = torch.load("D:/Latent/boxes/runs_fsq/s213_d3t_s1/model.pt", map_location="cpu", weights_only=True)
cnt = {}
for k, v in sd.items():
    if k == "c_anneal":
        continue
    G = g["group"](k); G2 = G.split("_")[0] if G[:4] in ("core", "coda", "prel") else G
    cnt[G2] = cnt.get(G2, 0) + v.numel()
tot = sum(cnt.values())
out = {"def": __doc__, "total": tot, "groups": {k: {"n": n, "share": round(n / tot, 5)} for k, n in sorted(cnt.items(), key=lambda kv: -kv[1])}}
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rv_group_sizes.json"), "w", newline="\n"), indent=1)
print(json.dumps(out["groups"], indent=0), tot)
