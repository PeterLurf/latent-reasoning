"""Score PROTOCOL6 addendum O (longer lookup scaffold, paired). Written 2026-09-29 ~00:50, before any O outcome was read
(queue logs were only checked for progress counts).

Pairs: (o_s<seed>, p30_s<seed>) for seeds 100-123 (local, same card), (o_s<seed>, ob_s<seed>) for seeds 124-147 (Vast,
same instance). A run succeeds if competent and extrapolating (PROTOCOL5 rules, latent.protocol5_confirm.pilot_run); a
run that never produced a model (stage-1 failure after its relaunch budget) counts as not successful. Cut-offs:
local runs not *started* by Tue 2026-09-29 12:00 and Vast runs not *finished* by then drop with their pair.
O1: l = long-only successes, b = base-only successes; supported if l - b >= 4 and one-sided exact sign test p <= 0.05;
not supported if l <= b; else inconclusive. Secondary: per set; competence and full-depth step by arm; P7 by arm
(frozen far_sweep, seed 2033); dev6 pairs (o_dev6_s51-56 vs dev6_s51-56), descriptive.
    PYTHONPATH=. python scratch/score_o.py --device cuda:0 [--final]
"""
import argparse
import glob
import json
import os
import subprocess
import sys
from math import comb

import numpy as np

from latent import protocol5_confirm as P5

FAR = "results/far_sweep_o_s2033.json"


EVALUATED = None


def evaluated():
    """Runs whose queue worker logged 'evaluated <run>' (local logs and logs fetched from Vast)."""
    global EVALUATED
    if EVALUATED is None:
        EVALUATED = set()
        for f in glob.glob("logs/queue_*.out") + glob.glob("logs/vast_*/queue_*.out"):
            for line in open(f, errors="ignore"):
                if " evaluated " in line:
                    EVALUATED.add(line.split(" evaluated ")[1].split()[0])
    return EVALUATED


def status(r):
    """'done' (evaluated, with results) | 'no_model' (queue finished it without a model: stage-1 failure) | 'pending'."""
    if os.path.exists(f"results/competence_matched_{r}.json") and os.path.exists(f"results/length_gen_{r}.json"):
        return "done"
    if r in evaluated() and not os.path.exists(f"runs/{r}/model.pt"):
        return "no_model"
    return "pending"


def success(r):
    p = P5.pilot_run(r)
    return bool(p.get("gate") and p.get("extrap_K12")), p


def sign_p(l, b):
    n = l + b
    return None if n == 0 else sum(comb(n, k) for k in range(l, n + 1)) / 2 ** n


def full_step(r):
    f = f"runs/{r}/log.jsonl"
    if not os.path.exists(f):
        return None
    adv = [json.loads(x) for x in open(f) if "curriculum_advance_to" in x]
    return next((a["step"] for a in adv if a["curriculum_advance_to"] is None), None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--final", action="store_true", help="after the cut-off: pending pairs are dropped")
    ap.add_argument("--dry", action="store_true", help="only report which pairs are scorable (no outcomes, no GPU)")
    a = ap.parse_args()
    if a.dry:
        for name, pairs in (("local", [(f"o_s{s}", f"p30_s{s}") for s in range(100, 124)]),
                            ("vast", [(f"o_s{s}", f"ob_s{s}") for s in range(124, 148)])):
            st = [(status(lo), status(ba)) for lo, ba in pairs]
            print(name, "scorable", sum("pending" not in x for x in st), "of", len(pairs),
                  "| no_model runs", sum(x.count("no_model") for x in st))
        return
    sets = {"local": [(f"o_s{s}", f"p30_s{s}") for s in range(100, 124)],
            "vast": [(f"o_s{s}", f"ob_s{s}") for s in range(124, 148)]}
    rows, pending = {}, []
    for name, pairs in sets.items():
        for lo, ba in pairs:
            st = (status(lo), status(ba))
            if "pending" in st:
                pending.append((lo, ba))
                continue
            (sl, pl), (sb, pb) = (success(lo) if st[0] == "done" else (False, {})), (success(ba) if st[1] == "done" else (False, {}))
            rows[lo] = {"set": name, "base": ba, "long_success": sl, "base_success": sb,
                        "long_competent": bool(pl.get("gate")), "base_competent": bool(pb.get("gate")),
                        "long_full_step": full_step(lo), "base_full_step": full_step(ba),
                        "long_d7": pl.get("d7_K12"), "base_d7": pb.get("d7_K12")}
    # P7 by arm for competent runs (frozen code)
    comp = sorted([lo for lo, v in rows.items() if v["long_competent"]] +
                  [v["base"] for v in rows.values() if v["base_competent"]])
    F = json.load(open(FAR)) if os.path.exists(FAR) else {}
    need = [f"runs/{r}" for r in comp if f"runs/{r}" not in F and not r.startswith("p30_")]
    if need:
        subprocess.run([sys.executable, "scratch/far_sweep_frozen_I.py", *need, "--seed", "2033", "--out", FAR,
                        "--device", a.device], check=True, env=dict(os.environ, PYTHONPATH="."))
        F = json.load(open(FAR))
    P30 = json.load(open("results/p30.json"))["rows"]
    p7 = lambda r: (P30[r]["P7"] if r in P30 else None) if r.startswith("p30_") else F.get(f"runs/{r}", {}).get("P7")
    for lo, v in rows.items():
        v["long_P7"] = p7(lo) if v["long_competent"] else None
        v["base_P7"] = p7(v["base"]) if v["base_competent"] else None
    out = {"n_pairs_scored": len(rows), "pending": pending, "final": a.final, "rows": rows}

    def o1(sel):
        l = sum(v["long_success"] and not v["base_success"] for v in sel)
        b = sum(v["base_success"] and not v["long_success"] for v in sel)
        p = sign_p(l, b)
        dec = "supported" if (l - b >= 4 and p is not None and p <= 0.05) else "not supported" if l <= b else "inconclusive"
        return {"n": len(sel), "l": l, "b": b, "sign_p": p, "decision": dec,
                "success_long": sum(v["long_success"] for v in sel), "success_base": sum(v["base_success"] for v in sel),
                "competent_long": sum(v["long_competent"] for v in sel), "competent_base": sum(v["base_competent"] for v in sel),
                "median_full_step_long": float(np.median([v["long_full_step"] for v in sel if v["long_full_step"]] or [np.nan])),
                "median_full_step_base": float(np.median([v["base_full_step"] for v in sel if v["base_full_step"]] or [np.nan])),
                "share_P7_ge_0.2_long": float(np.mean([v["long_P7"] >= 0.2 for v in sel if v["long_P7"] is not None] or [np.nan])),
                "share_P7_ge_0.2_base": float(np.mean([v["base_P7"] >= 0.2 for v in sel if v["base_P7"] is not None] or [np.nan]))}
    out["O1_all"] = o1(list(rows.values()))
    out["O1_local"] = o1([v for v in rows.values() if v["set"] == "local"])
    out["O1_vast"] = o1([v for v in rows.values() if v["set"] == "vast"])
    if pending and not a.final:
        out["O1_all"]["decision"] = "incomplete (" + str(len(pending)) + " pairs pending)"
    dev = {}
    for s in range(51, 57):
        lo, ba = f"o_dev6_s{s}", f"dev6_s{s}"
        if status(lo) == "done" and status(ba) == "done":
            dev[s] = {"long": success(lo)[0], "base": success(ba)[0]}
    out["dev6_pairs"] = dev
    json.dump(out, open("results/p_o.json", "w"), indent=1)
    print({k: v for k, v in out.items() if k not in ("rows",)})


if __name__ == "__main__":
    main()
