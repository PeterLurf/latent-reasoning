"""PROTOCOL6 addendum E: build branch runs from the dev6 checkpoints saved at curriculum advances, and the queue files.

    PYTHONPATH=. python scratch/p6_branch.py            # after every dev6 run is trained and measured

Selection (frozen in addendum E): among competent dev6 runs in seed order, the first two with final end activity
>= 0.25 and the first two below it. Branch points: the checkpoints at the advances to stage 2, stage 4 and full depth.
Two branches per (run, point): checkpoint weights, fresh AdamW state (zero moments, step count 0), fresh data stream
(seed + 200003 + 7j), curriculum state rebuilt from the log up to that step; then the original schedule to 30 000.
"""
import json
import os

import numpy as np
import torch

from latent import protocol5_confirm as P5
from latent.protocol6 import DEV, ENDACT, T

EA = json.load(open(ENDACT))


def params(sd, c):
    from latent.model import LatentReasoner
    m = LatentReasoner(c["d"], c["layers"], c["heads"], k_max=max(c["K"], 8), mode=c["mode"], emb=c.get("emb", "slots"),
                       d_id=c.get("d_id", 64), init=c.get("init", "torch"), feedback=c.get("feedback", "post_ln"),
                       step_emb=not c.get("no_step_emb", False))
    m.load_state_dict(sd)
    return [p.detach() for p in m.parameters()]


def select():
    comp = [r for r in DEV if P5.pilot_run(r).get("gate") and f"runs/{r}" in EA]
    hi = [r for r in comp if EA[f"runs/{r}"]["end_activity"] >= T][:2]
    lo = [r for r in comp if EA[f"runs/{r}"]["end_activity"] < T][:2]
    return (hi + lo) if hi and lo else comp[:4]


def branch(r, j, adv_to):
    c = json.load(open(f"runs/{r}/config.json"))
    lines = [json.loads(l) for l in open(f"runs/{r}/log.jsonl")]
    adv = [d for d in lines if "curriculum_advance_to" in d]
    at = next(d["step"] for d in adv if d["curriculum_advance_to"] == adv_to)
    ck = torch.load(f"runs/{r}/ckpts/step_{at:05d}.pt", map_location="cpu", weights_only=False)
    assert ck["step"] == at
    tag = "full" if adv_to is None else f"st{adv_to}"
    name = f"{r}_br{tag}_{j}"
    d = f"runs/{name}"
    if not os.path.exists(f"{d}/state.pt"):
        os.makedirs(d, exist_ok=True)
        json.dump(c, open(f"{d}/config.json", "w"), indent=2)
        with open(f"{d}/log.jsonl", "w") as f:
            f.write("".join(json.dumps(x) + "\n" for x in lines if x.get("step", 0) <= at))
        hist, stage = [], 1
        for e in adv:
            if e["step"] > at:
                break
            hist.append((e["step"], stage)); stage = e["curriculum_advance_to"]
        seed = c["seed"] + 200003 + 7 * j
        torch.cuda.manual_seed(seed)
        torch.save({"model": ck["model"], "step": at, "sec": 0.0, "cur": {"d": stage, "hist": hist},
                    "opt": [{"step": torch.zeros(()), "exp_avg": torch.zeros_like(v), "exp_avg_sq": torch.zeros_like(v)}
                            for v in params(ck["model"], c)],
                    "np_rng": np.random.default_rng(seed).bit_generator.state,
                    "torch_rng": torch.Generator().manual_seed(seed).get_state(),
                    "cuda_rng": torch.cuda.get_rng_state("cuda:0")}, f"{d}/state.pt")
    return name, at, f"{name} --K {c['K']} --seed {c['seed']} --lr {c['lr']} --amp {c['amp']}", c["amp"]


if __name__ == "__main__":
    runs = select()
    print("selected", runs, {r: round(EA[f"runs/{r}"]["end_activity"], 3) for r in runs})
    bf, fp = [], []
    for r in runs:
        for adv_to in (2, 4, None):
            for j in (1, 2):
                name, at, line, amp = branch(r, j, adv_to)
                (fp if amp == "fp32" else bf).append((30000 - at, line))
                print(name, "from step", at)
    # longest first so the three GPUs finish together
    open("scratch/p6_branch_queue.txt", "w").write(
        "# PROTOCOL6 addendum E branches (bf16; 4060/3070)\n" + "\n".join(l for _, l in sorted(bf, reverse=True)) + "\n")
    open("scratch/p6_branch_queue_2070.txt", "w").write(
        "# PROTOCOL6 addendum E branches (fp32; 2070)\n" + "\n".join(l for _, l in sorted(fp, reverse=True)) + "\n")
