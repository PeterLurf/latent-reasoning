"""TRAINING_TODO Phase 0 stall pilot: 4 configs x 3 seeds of the base recipe, each stopped at stage-1 advance
or at step 4000 (the restart rule's cutoff), whichever comes first. --steps stays 30000 so the LR schedule is
the real one. Writes results/stall_pilot.json (advance step or null per run) and logs/stall_pilot.log.

    python scratch/stall_pilot.py <device> [wait-for-file|-] [tag seeds configs extra-args...]
e.g. python scratch/stall_pilot.py cuda:0 - nostep 10,11,12,13,14,15 base,wu1000 --no_step_emb
     -> results/stall_pilot_nostep.json, runs/pilot/nostep_<cfg>_s<seed>, shared logs/stall_pilot.log
"""
import json
import os
import subprocess
import sys
import time

dev = sys.argv[1]
wait_for = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "-" else None
tag = sys.argv[3] if len(sys.argv) > 3 else ""
CONFIGS = {"base": [], "wu1000": ["--warmup", "1000"], "lr5e-4": ["--lr", "5e-4"],
           "lr5e-4_wu1000": ["--lr", "5e-4", "--warmup", "1000"]}
SEEDS = [int(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else [0, 1, 2]
if len(sys.argv) > 5:
    CONFIGS = {c: CONFIGS[c] for c in sys.argv[5].split(",")}
EXTRA = sys.argv[6:]
BASE = ["-m", "latent.train", "--mode", "coconut", "--bs", "512", "--steps", "30000", "--warmup", "300",
        "--eval_every", "1000", "--aux_steps", "10000", "--aux_anneal", "5000", "--children_weight", "1",
        "--curriculum", "gated", "--init", "gpt2", "--device", dev]
OUT = f"results/stall_pilot{'_' + tag if tag else ''}.json"
log = open("logs/stall_pilot.log", "a")


def say(*a):
    print(time.strftime("%H:%M:%S"), *a, file=log, flush=True)


while wait_for and "QUEUE_DONE" not in (open(wait_for).read() if os.path.exists(wait_for) else ""):
    time.sleep(60)
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
for cfg, extra in CONFIGS.items():
    for s in SEEDS:
        key = f"{tag + '_' if tag else ''}{cfg}_s{s}"
        if key in res:
            continue
        out = f"runs/pilot/{key}"
        os.makedirs(out, exist_ok=True)
        cmd = [sys.executable] + BASE + EXTRA + extra + ["--seed", str(s), "--out", out]   # later flags override --warmup
        p = subprocess.Popen(cmd, stdout=open(f"{out}/train.out", "a"), stderr=open(f"{out}/train.err", "a"))
        say("start", key, "pid", p.pid)
        adv, step = None, 0
        while p.poll() is None:
            time.sleep(20)
            f = os.path.join(out, "log.jsonl")
            if not os.path.exists(f):
                continue
            for line in open(f):
                x = json.loads(line)
                step = max(step, x.get("step", 0))
                if adv is None and "curriculum_advance_to" in x:
                    adv = x.get("step", step)
            if adv is not None or step >= 4000:
                p.kill(); p.wait()
        if p.returncode not in (None, 0) and adv is None and step < 4000:
            say("CRASH", key, "rc", p.returncode, "at step", step)
            continue
        res[key] = {"config": cfg, "seed": s, "advance_step": adv, "stalled": adv is None, "last_step": step}
        if os.path.exists(OUT):                      # another GPU may share this file: merge before writing
            res = {**json.load(open(OUT)), **res}
        json.dump(res, open(OUT, "w"), indent=1)
        say("done", key, "advance", adv, "last step", step)
res = {**(json.load(open(OUT)) if os.path.exists(OUT) else {}), **res}
pre = tag + "_" if tag else ""
rates = {c: sum(res[f"{pre}{c}_s{s}"]["stalled"] for s in SEEDS if f"{pre}{c}_s{s}" in res) for c in CONFIGS}
say(tag, f"stall counts (of {len(SEEDS)}):", rates)
say("PILOT_DONE", tag)
