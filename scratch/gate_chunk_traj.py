"""CHUNK_DELTA gate 3 trajectory (Theory 2026-10-10, fixed pre-data): b259's traj_chunk / traj_loop (micro 2) / traj_loop_m4 (micro 4), 200 steps, loss logged
every 10 steps (20 points). d_chunk = max |loss(chunk) - loss(loop)|, d_reorder = max |loss(loop) - loss(loop_m4)| over the logged points.
PASS iff d_chunk <= max(2 x d_reorder, 0.02) and no non-finite record / loss in any run. -> gate_chunk_delta.json g5_traj
    python scratch/gate_chunk_traj.py"""
import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT)
F = "boxes/runs_fsq/chunk_delta_gates/gate_chunk_delta.json"; R = "boxes/runs_fsq/s168s2_traj_%s/log.jsonl"


def curve(nm):
    rs = [json.loads(l) for l in open(R % nm) if l.strip()]
    bad = sum(1 for r in rs if r.get("nonfinite") or (r.get("loss") is not None and not math.isfinite(r["loss"])))
    return {r["step"]: r["loss"] for r in rs if r.get("loss") is not None and "step" in r}, bad


c, bc = curve("chunk"); l2, bl = curve("loop"); l4, b4 = curve("loop_m4")
steps = sorted(set(c) & set(l2) & set(l4)); assert len(steps) == 20, steps
d_chunk = max(abs(c[s] - l2[s]) for s in steps); d_reorder = max(abs(l2[s] - l4[s]) for s in steps); bound = max(2 * d_reorder, 0.02)
G = json.load(open(F))
G["g5_traj"] = {"d_chunk": d_chunk, "d_reorder": d_reorder, "bound": bound, "n_points": len(steps), "nonfinite": {"chunk": bc, "loop": bl, "loop_m4": b4},
                "curves": {"chunk": [c[s] for s in steps], "loop": [l2[s] for s in steps], "loop_m4": [l4[s] for s in steps], "steps": steps},
                "def": "Theory (pre-data): PASS iff d_chunk <= max(2 x d_reorder, 0.02) and no non-finite in any run"}
G["g5_pass"] = bool(d_chunk <= bound and bc == bl == b4 == 0)
json.dump(G, open(F, "w", newline="\n"), indent=1); print("G5", d_chunk, d_reorder, bound, G["g5_pass"])
