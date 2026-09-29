"""PROTOCOL6 addenda A/B: copy runs to runs/<run>_ext for continuation and write the two local queues.

Run as  PYTHONPATH=. python scratch/p6_setup.py

Every continuation starts from the run's final weights (model.pt) with a fresh AdamW state (zero moments, step count 0) and a fresh data
stream (seed + 100003): 7 of the 17 runs trained on pods whose state.pt was never fetched, so the resume is uniform.
"""
import json, os, shutil

import numpy as np
import torch

COMP = ["rel_base_s20", "rel_base_s24", "rel_base_s25", "rel_base_s26", "rel_base_s27",
        "rel_lr5_s20", "rel_lr5_s22", "rel_lr5_s24", "rel_lr5_s26", "rel_lr5_s27", "rel_lr5_s21", "rel_lr5_s23"]
STALL = ["rel_base_s21", "rel_base_s22", "rel_base_s23", "rel_lr5_s25", "nostep_s0"]


def params(ck):
    """Parameter tensors in model.parameters() order (the order train.py saves and restores optimizer state in)."""
    from latent.model import LatentReasoner
    c = ck["config"]
    m = LatentReasoner(c["d"], c["layers"], c["heads"], k_max=max(c["K"], 8), mode=c["mode"], emb=c.get("emb", "slots"),
                       d_id=c.get("d_id", 64), init=c.get("init", "torch"), feedback=c.get("feedback", "post_ln"),
                       step_emb=not c.get("no_step_emb", False))
    m.load_state_dict(ck["model"])
    return [p.detach() for p in m.parameters()]


def ext_line(r):
    c = json.load(open(f"runs/{r}/config.json"))
    d = f"runs/{r}_ext"
    if not os.path.exists(d):
        os.makedirs(d)
        for f in ("config.json", "log.jsonl", "model.pt"):
            shutil.copy2(f"runs/{r}/{f}", f"{d}/{f}")
        ck = torch.load(f"{d}/model.pt", map_location="cpu", weights_only=False)
        adv = [json.loads(l) for l in open(f"{d}/log.jsonl") if "curriculum_advance_to" in l]
        hist, stage = [], 1
        for e in adv:
            hist.append((e["step"], stage)); stage = e["curriculum_advance_to"]
        seed = c["seed"] + 100003
        torch.cuda.manual_seed(seed)
        torch.save({"model": ck["model"], "step": ck["step"], "sec": 0.0, "cur": {"d": stage, "hist": hist},
                    "opt": [{"step": torch.zeros(()), "exp_avg": torch.zeros_like(v, dtype=torch.float32),
                             "exp_avg_sq": torch.zeros_like(v, dtype=torch.float32)} for v in params(ck)], "np_rng": np.random.default_rng(seed).bit_generator.state,
                    "torch_rng": torch.Generator().manual_seed(seed).get_state(),
                    "cuda_rng": torch.cuda.get_rng_state("cuda:0")}, f"{d}/state.pt")
    args = f"--K {c['K']} --seed {c['seed']} --lr {c['lr']}" + (" --no_step_emb" if c.get("no_step_emb") else "")
    return f"{r}_ext {args} --amp {c.get('amp') or 'bf16'} --steps 45000 --save_every 2500"


dev = lambda s, amp: f"dev6_s{s} --K 6 --seed {s} --lr 5e-4 --save_every 1000 --amp {amp}"
A = ["# PROTOCOL6 local queue for the RTX 4060 / 3070 (bf16): addendum A continuation, A3 stalled runs, addendum B."]
A += [ext_line(r) for r in COMP[:10]] + [ext_line(r) for r in STALL] + [dev(s, "bf16") for s in range(30, 39)]
B = ["# PROTOCOL6 local queue for the RTX 2070 (fp32): the two fp32 pilot runs' continuation, then addendum B."]
B += [ext_line(r) for r in COMP[10:]] + [dev(s, "fp32") for s in range(39, 42)]
open("scratch/p6_queue.txt", "w").write("\n".join(A) + "\n")
open("scratch/p6_queue_2070.txt", "w").write("\n".join(B) + "\n")
print("\n".join(A + B))
