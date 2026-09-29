"""PROTOCOL5 plan step 4a: set a late-stage (stages 2-4) restart rule from the training logs already on disk.

    python scratch/stall_rule.py        -> results/stall_rule.json

Every gated-curriculum attempt of the protocol runs (final runs and archived stage-1 failures; hardware-crash folders
and pre-protocol pilots excluded) is one trajectory: the step of each curriculum advance, censored at 30k. A candidate
rule "relaunch if the run spends more than T steps in stage 2, 3 or 4" is replayed on each trajectory together with
the existing stage-1 rule (relaunch at step 4000 with no advance). With relaunches drawn from the same empirical
distribution, the expected training steps per attempt that reaches full depth = E[steps per attempt] / P(full depth).
"""
import glob
import json
import os

import numpy as np

EXCLUDE = ("crashed", "stalled", "moved", "r3_p_gated", "r3_pc_gated_torch", "rel_")
STEPS = 30000


def trajectories():
    out = {}
    for f in sorted(glob.glob("runs/*/log.jsonl")):
        run = os.path.basename(os.path.dirname(f))
        cfg = os.path.join(os.path.dirname(f), "config.json")
        if any(e in run for e in EXCLUDE) or not os.path.exists(cfg) or json.load(open(cfg)).get("curriculum") != "gated":
            continue
        adv, last = [], 0
        for line in open(f):
            x = json.loads(line)
            last = max(last, x.get("step", 0))
            if "curriculum_advance_to" in x:
                adv.append(x["step"])
        if last < 4000 and not adv:
            continue                                   # killed early for non-protocol reasons
        out[run] = {"adv": adv, "last": last, "full": len(adv) == 4,
                    "no_step_emb": bool(json.load(open(cfg)).get("no_step_emb"))}
    return out


def replay(t, T):
    """Steps used and outcome of one attempt under stage-1 rule + dwell limit T for stages 2-4 (None = no limit)."""
    adv = t["adv"]
    if not adv:
        return 4000, False                             # stage-1 rule
    bounds = adv + [None] * (4 - len(adv))             # advance steps into stages 2, 3, 4, full
    for s in range(3):                                 # dwell in stage s+2 = bounds[s+1] - bounds[s]
        start, end = bounds[s], bounds[s + 1]
        if T is not None and (end if end is not None else STEPS) - start > T:
            return min(start + T, STEPS), False
        if end is None:
            return STEPS, False
    return STEPS, True


def main():
    tr = trajectories()
    dwell = {2: [], 3: [], 4: []}
    for t in tr.values():
        a = t["adv"]
        for s in range(min(len(a), 4) - 1):
            dwell[s + 2].append(a[s + 1] - a[s])
    res = {"n_attempts": len(tr), "n_full": sum(t["full"] for t in tr.values()),
           "dwell_successful": {s: sorted(v) for s, v in dwell.items()}, "rules": {}}
    for group, sel in (("all", lambda t: True), ("no_step_emb", lambda t: t["no_step_emb"]),
                       ("step_emb", lambda t: not t["no_step_emb"])):
        g = [t for t in tr.values() if sel(t)]
        res["rules"][group] = {}
        for T in (None, 6000, 8000, 10000, 12000, 14000, 16000):
            r = [replay(t, T) for t in g]
            steps = np.array([x for x, _ in r]); ok = np.array([y for _, y in r])
            p = ok.mean()
            res["rules"][group][str(T)] = {"n": len(g), "p_full": float(p), "mean_steps": float(steps.mean()),
                                           "steps_per_full": float(steps.mean() / p) if p else None,
                                           "successes_killed": int(sum(t["full"] for t in g) - ok.sum())}
    json.dump(res, open("results/stall_rule.json", "w"), indent=1)
    print("attempts", res["n_attempts"], "reached full depth", res["n_full"])
    for s, v in dwell.items():
        print(f"stage {s} dwell (successful): n={len(v)} median={np.median(v):.0f} p90={np.percentile(v, 90):.0f} max={max(v)}")
    for group, rules in res["rules"].items():
        print(group)
        for T, x in rules.items():
            print(f"  T={T:>6s} n={x['n']} P(full)={x['p_full']:.2f} steps/attempt={x['mean_steps']:.0f} "
                  f"steps per full-depth run={x['steps_per_full'] or float('nan'):.0f} successes killed={x['successes_killed']}")


if __name__ == "__main__":
    main()
