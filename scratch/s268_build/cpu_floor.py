"""Local CPU only: the run-to-run floor of gate_s268's offbit on this box = the NEW tree twice (pilot flags, bun's cue, det wrap, fp32 micro 1, 2 steps) ->
scratch/_tmp/s268g_local/cpu_floor.json {n_mismatch, max_abs_model_delta}. If the new tree against itself already differs at ~1e-8, the local offbit_aw failure
(70 / 162 model tensors, max 6.7e-8, logs equal) is the CPU reduction-order floor, not a code change; the host gate (GPU det) is the binding one.
    G268_DEV=cpu G268_STOP=2 python -B scratch/s268_build/cpu_floor.py"""
import json
import os
import shutil
import sys

sys.path.insert(0, "D:/Latent/scratch"); os.chdir("D:/Latent")
import gate_s268 as G  # noqa: E402
import torch  # noqa: E402

assert G.ST == 2 and G.DEV == "cpu"
fl = G.flags(None) + G.COM + ["--ckpt_every", "2", "--log_every", "10", "--stop_at", "2"]; da, db = G.TMP + "/floor_a", G.TMP + "/floor_b"
rc = [G.det_in(G.ROOT, "boxes/inject_aw.py", fl, da), G.det_in(G.ROOT, "boxes/inject_aw.py", fl, db)]
a = torch.load(da + "/ckpt_last.pt", map_location="cpu", weights_only=False)["model"]; b = torch.load(db + "/ckpt_last.pt", map_location="cpu", weights_only=False)["model"]
dl = [float((a[k].float() - b[k].float()).abs().max()) for k in a if not torch.equal(a[k], b[k])]
o = {"rc": rc, "n_model_tensors": len(a), "n_mismatch": len(dl), "max_abs_model_delta": max(dl) if dl else 0.0, "log_equal": G.strip(da + "/log.jsonl") == G.strip(db + "/log.jsonl")}
json.dump(o, open(G.OUT + "/cpu_floor.json", "w"), indent=1); print("cpu_floor", o, flush=True); shutil.rmtree(da); shutil.rmtree(db)
