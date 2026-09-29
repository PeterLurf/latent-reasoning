"""Measure and score docs/PROTOCOL6.md (addenda A, B, E, F, H).

    python -m latent.protocol6 measure <runs...> --device cuda:0   # end activity + strict clock -> results/p6_endact.json, p6_strict.json
    python -m latent.protocol6 traj <runs...> --device cuda:0      # EA and d7 at K'=12 per checkpoint -> results/p6_traj.json
    python -m latent.protocol6 score                               # verdicts -> results/protocol6_confirm.json

Definitions are those of PROTOCOL5 (end activity, P26, strict clock) and PROTOCOL6; nothing is tuned here.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from . import length_gen as LG
from . import protocol5_confirm as P5
from .confirm_sweep import trace
from .development import load_ckpt
from .protocol3_confirm import jload
from .protocol4_confirm import final_stage

ENDACT, STRICT, TRAJ, OUT = ("results/p6_endact.json", "results/p6_strict.json", "results/p6_traj.json",
                             "results/protocol6_confirm.json")
COMP = ["rel_base_s20", "rel_base_s24", "rel_base_s25", "rel_base_s26", "rel_base_s27", "rel_lr5_s20", "rel_lr5_s21",
        "rel_lr5_s22", "rel_lr5_s23", "rel_lr5_s24", "rel_lr5_s26", "rel_lr5_s27"]      # addendum A: competent pilot runs
STALL = ["rel_base_s21", "rel_base_s22", "rel_base_s23", "rel_lr5_s25", "nostep_s0"]   # addendum A3
DEV = [f"dev6_s{s}" for s in range(30, 42)]                                              # addendum B
T = P5.END_T
PREC = list(range(50, 58))                                                               # addendum F (same-card pairs)
PREC2070 = list(range(50, 54))                                                           # addendum F, 2070 arm
STALLH = list(range(70, 78))                                                             # addendum H


def measure(runs, device):
    P5.ENDACT, P5.STRICT = ENDACT, STRICT          # same measurement code, Part VI output files
    P5.measure_endact(runs, device)
    P5.measure_strict(runs, device)


def traj(runs, device, n_len=512):
    """Addendum B1: EA (W = 4 trace, seed 2026 + 4) and distance-7 accuracy at K' = 12 at every checkpoint."""
    done = jload(TRAJ) or {}
    for run in runs:
        files = sorted(glob.glob(f"{run}/ckpts/step_*.pt")) + [f"{run}/model.pt"]
        res = done.setdefault(run, {})
        for f in files:
            m, c, step = load_ckpt(f, device)
            if str(step) in res:
                continue
            K = c["K"]
            tr = trace(m, K, max(8, K), 4, device)
            LG.load = lambda _r, _d, m=m, c=c: (m, c)
            row = LG.evaluate(run, device, n=n_len, budgets=(12,))["rows"][0]
            res[str(step)] = {"end_activity": P5.end_activity(tr, K), "d7_K12": row["far"]["7"],
                              "d7_ci": row["far_ci"]["7"]}
            json.dump(done, open(TRAJ, "w"), indent=1)
            print(f"{run} step {step}: EA {res[str(step)]['end_activity']:.3f} d7 {row['far']['7']:.3f}", flush=True)


def p26(rows):
    """PROTOCOL5 addendum B rule on (run, end activity, extrapolates) rows."""
    lo = [e for _, v, e in rows if v < T]; hi = [e for _, v, e in rows if v >= T]
    a_ok = sum(lo) <= 1
    rate = lambda xs: (sum(xs) / len(xs)) if xs else None
    b_dir = None if not lo or not hi else (rate(hi) > rate(lo)) if rate(hi) != rate(lo) else None
    c_auc = (P5.auc([v for _, v, _ in rows], [e for _, _, e in rows])
             if len(lo) >= 2 and len(hi) >= 2 and 0 < sum(e for *_, e in rows) < len(rows) else None)
    rd = ("fails" if not a_ok or b_dir is False else
          "confirmed" if b_dir and (c_auc is None or c_auc >= 0.70) else "unresolved")
    return {"runs": {r: {"end_activity": v, "extrapolates": e} for r, v, e in rows},
            "low": {"n": len(lo), "n_extrap": sum(lo)}, "high": {"n": len(hi), "n_extrap": sum(hi)},
            "a_necessity": a_ok, "b_direction": b_dir, "c_auc": c_auc, "reading": rd}


def full_step(r):
    f = f"runs/{r}/log.jsonl"
    adv = [json.loads(l) for l in open(f) if "curriculum_advance_to" in l] if os.path.exists(f) else []
    return next((a["step"] for a in adv if a["curriculum_advance_to"] is None), None)


def sign_p(b, f):
    """Exact one-sided sign test: P(X >= b) for X ~ Binomial(b + f, 1/2)."""
    from math import comb
    n = b + f
    return None if n == 0 else sum(comb(n, k) for k in range(b, n + 1)) / 2 ** n


def stage_steps(r):
    """Steps spent before each curriculum advance (from log.jsonl), and the last logged step."""
    f = f"runs/{r}/log.jsonl"
    if not os.path.exists(f):
        return None
    rows = [json.loads(l) for l in open(f)]
    adv = [(x["step"], x["curriculum_advance_to"]) for x in rows if "curriculum_advance_to" in x]
    return {"advances": adv, "last_step": max((x.get("step", 0) for x in rows), default=None)}


def score_prec(ea6):
    """Addendum F (P27) and addendum H (P28)."""
    run = lambda r: dict(P5.pilot_run(r), end_activity=ea6(r))
    out = {}
    # F: pairs where both same-card arms are competent; bf16-only vs fp32-only extrapolators
    F = {s: {"bf16": run(f"prec_bf16_s{s}"), "fp32": run(f"prec_fp32_s{s}")} for s in PREC}
    both = [s for s in PREC if F[s]["bf16"].get("gate") and F[s]["fp32"].get("gate")]
    ev = [s for s in both if "extrap_K12" in F[s]["bf16"] and "extrap_K12" in F[s]["fp32"]]
    b = sum(F[s]["bf16"]["extrap_K12"] and not F[s]["fp32"]["extrap_K12"] for s in ev)
    f = sum(F[s]["fp32"]["extrap_K12"] and not F[s]["bf16"]["extrap_K12"] for s in ev)
    done = all(F[s][a]["trained"] and ("gate" in F[s][a]) for s in PREC for a in ("bf16", "fp32"))
    arms = {}
    for a in ("bf16", "fp32"):
        rs = [F[s][a] for s in PREC]
        comp = [x for x in rs if x.get("gate")]
        arms[a] = {"full_depth": sum(x["final_stage"] == "full" for x in rs), "gate": len(comp),
                   "extrap": sum(bool(x.get("extrap_K12")) for x in comp),
                   "EA_high": sum(x["end_activity"] is not None and x["end_activity"] >= T for x in comp),
                   "extrap_given_EA_high": f"{sum(bool(x.get('extrap_K12')) for x in comp if x['end_activity'] is not None and x['end_activity'] >= T)}"
                                           f"/{sum(x['end_activity'] is not None and x['end_activity'] >= T for x in comp)}"}
    A2070 = {s: run(f"prec2070_fp32_s{s}") for s in PREC2070}
    arms["fp32_2070"] = {"full_depth": sum(x["final_stage"] == "full" for x in A2070.values()),
                         "gate": sum(bool(x.get("gate")) for x in A2070.values()),
                         "extrap": sum(bool(x.get("gate") and x.get("extrap_K12")) for x in A2070.values())}
    out["F"] = {"pairs": F, "arm_2070": A2070, "both_competent": both, "b": b, "f": f, "sign_p": sign_p(b, f),
                "complete": done, "arms": arms,
                "P26_bf16": p26([(f"prec_bf16_s{s}", F[s]["bf16"]["end_activity"], F[s]["bf16"]["extrap_K12"]) for s in PREC
                                 if F[s]["bf16"].get("gate") and F[s]["bf16"]["end_activity"] is not None and "extrap_K12" in F[s]["bf16"]]),
                "P26_fp32": p26([(f"prec_fp32_s{s}", F[s]["fp32"]["end_activity"], F[s]["fp32"]["extrap_K12"]) for s in PREC
                                 if F[s]["fp32"].get("gate") and F[s]["fp32"]["end_activity"] is not None and "extrap_K12" in F[s]["fp32"]]),
                "P27": (None if not done else "supported" if b - f >= 3 and f <= 1 else "not supported" if f >= b
                        else "inconclusive")}
    # H: pairs where only one arm reaches full curriculum depth
    def pairs(seeds, fmt):
        rows = {s: {a: {"final_stage": final_stage(fmt.format(a=a, s=s)), "steps": stage_steps(fmt.format(a=a, s=s)),
                        # finished = the eval battery has run (model.pt also appears mid-run with --save_every)
                        "trained": os.path.exists(f"results/competence_matched_{fmt.format(a=a, s=s)}.json")}
                    for a in ("bf16", "fp32")}
                for s in seeds}
        fin = [s for s in seeds if rows[s]["bf16"]["trained"] and rows[s]["fp32"]["trained"]]
        bb = sum(rows[s]["bf16"]["final_stage"] == "full" and rows[s]["fp32"]["final_stage"] != "full" for s in fin)
        ff = sum(rows[s]["fp32"]["final_stage"] == "full" and rows[s]["bf16"]["final_stage"] != "full" for s in fin)
        return rows, fin, bb, ff
    rows, fin, bb, ff = pairs(STALLH, "stall_{a}_s{s}")
    for s in STALLH:
        for a in ("bf16", "fp32"):
            if rows[s][a]["trained"]:
                rows[s][a].update({k: v for k, v in run(f"stall_{a}_s{s}").items() if k in ("gate", "extrap_K12", "end_activity")})
    _, finF, bF, fF = pairs(PREC, "prec_{a}_s{s}")
    out["H"] = {"pairs": rows, "finished": fin, "b": bb, "f": ff, "sign_p": sign_p(bb, ff),
                "P28": (None if len(fin) < len(STALLH) else "supported" if bb - ff >= 2 and ff <= 1
                        else "not supported" if ff >= bb else "inconclusive"),
                "pooled_with_F_descriptive": {"n_pairs": len(fin) + len(finF), "b": bb + bF, "f": ff + fF,
                                              "sign_p": sign_p(bb + bF, ff + fF)}}
    return out


def score():
    E5, E6 = jload(P5.ENDACT) or {}, jload(ENDACT) or {}
    ea5 = lambda r: E5.get(f"runs/{r}", {}).get("end_activity")
    ea6 = lambda r: E6.get(f"runs/{r}", {}).get("end_activity")
    out = {}
    # A1: EA class before vs after continuation
    rows = {r: {"before": ea5(r), "after": ea6(f"{r}_ext"), "extrap_before": P5.pilot_run(r).get("extrap_K12"),
                "ext": P5.pilot_run(f"{r}_ext")} for r in COMP}
    ok = [r for r in COMP if rows[r]["after"] is not None]
    hi0 = [r for r in ok if rows[r]["before"] >= T]; lo0 = [r for r in ok if rows[r]["before"] < T]
    same = sum((rows[r]["before"] >= T) == (rows[r]["after"] >= T) for r in ok)
    closed = sum(rows[r]["after"] < T for r in hi0); opened = sum(rows[r]["after"] >= T for r in lo0)
    rd = ("incomplete" if len(ok) < len(COMP) else "stable" if same >= 10 else
          "closure" if closed >= 4 and opened <= 1 else "reopening" if opened >= 4 and closed <= 1 else "unstable")
    out["A1"] = {"runs": rows, "unchanged": f"{same}/{len(ok)}", "high_to_low": f"{closed}/{len(hi0)}",
                 "low_to_high": f"{opened}/{len(lo0)}", "reading": rd}
    # A2: P26 on the continued models; does extrapolation follow a change of EA class?
    comp = [(f"{r}_ext", rows[r]["after"], rows[r]["ext"]["extrap_K12"]) for r in ok
            if rows[r]["ext"].get("gate") and "extrap_K12" in rows[r]["ext"]]
    ch = [r for r in ok if (rows[r]["before"] >= T) != (rows[r]["after"] >= T) and "extrap_K12" in rows[r]["ext"]]
    track = [int(rows[r]["ext"]["extrap_K12"]) - int(bool(rows[r]["extrap_before"])) == (1 if rows[r]["after"] >= T else -1)
             for r in ch]
    out["A2"] = {"P26_continued": p26(comp), "changed_class": ch, "n_tracking": sum(track),
                 "tracking_reading": (None if len(ch) < 3 else "tracks" if sum(track) >= 2 / 3 * len(ch) else "does not track")}
    # A3: stalled runs (exploratory)
    out["A3"] = {r: {"stage_before": final_stage(r), "stage_after": final_stage(f"{r}_ext"),
                     "full_depth_step_after": full_step(f"{r}_ext"), "EA_before": ea5(r), "EA_after": ea6(f"{r}_ext"),
                     "extrap_before": P5.pilot_run(r).get("extrap_K12"), "after": P5.pilot_run(f"{r}_ext")}
                 for r in STALL}
    # addendum B: development runs
    D = {r: P5.pilot_run(r) for r in DEV}
    for r in DEV:
        D[r]["end_activity"] = ea6(r)
        D[r]["amp"] = (jload(f"runs/{r}/config.json") or {}).get("amp")
        D[r]["full_depth_step"] = full_step(r)
    out["dev"] = D
    rows_b = [(r, D[r]["end_activity"], D[r]["extrap_K12"]) for r in DEV
              if D[r].get("gate") and D[r]["end_activity"] is not None and "extrap_K12" in D[r]]
    out["P26b"] = p26(rows_b)
    out["P26b_bf16_only"] = p26([x for x in rows_b if D[x[0]]["amp"] != "fp32"])
    out["gate"] = {"n": sum("gate" in D[r] for r in DEV), "pass": sum(bool(D[r].get("gate")) for r in DEV),
                   "final_stages": {r: D[r]["final_stage"] for r in DEV}}
    # P23'' / P24'': strict clock on competent dev runs, pooled with the PROTOCOL5 pilot set
    S = jload(STRICT) or {}
    per = {r: dict(zip(("P23", "P24"), P5.p23_p24(o)), single_step=o["single_step"], clock=o["clock"])
           for r, o in S.items() if r[len("runs/"):] in DEV and D[r[len("runs/"):]].get("gate")}
    ss = [r for r in per if per[r]["single_step"]]
    p5 = (jload(P5.OUT) or {}).get("strict_clock", {})
    p5ss = p5.get("pilot_single_step_models", [])
    p5per = p5.get("per_model", {})
    out["strict_clock"] = {
        "per_model": per, "single_step_models": ss,
        "dev_P23": P5.reading([per[r]["P23"]["pass"] for r in ss]),
        "dev_P24": P5.reading([per[r]["P24"]["pass"] for r in ss]),
        "dev_other_P23": P5.reading([per[r]["P23"]["pass"] for r in per if r not in ss]),
        "dev_other_P24": P5.reading([per[r]["P24"]["pass"] for r in per if r not in ss]),
        "pooled_P23": P5.reading([per[r]["P23"]["pass"] for r in ss] + [p5per[r]["P23"]["pass"] for r in p5ss]),
        "pooled_P24": P5.reading([per[r]["P24"]["pass"] for r in ss] + [p5per[r]["P24"]["pass"] for r in p5ss])}
    # B1 (exploratory): is the final EA class set 1000 steps after full depth?
    TR = jload(TRAJ) or {}
    b1 = {}
    for r in DEV:
        tr, fs = TR.get(f"runs/{r}"), D[r]["full_depth_step"]
        if not tr or fs is None:
            continue
        st = sorted(map(int, tr))
        after = [s for s in st if s >= fs]
        early = next((s for s in after if s >= fs + 1000), None)
        fin = tr[str(st[-1])]["end_activity"]
        ea, d7 = [tr[str(s)]["end_activity"] for s in after], [tr[str(s)]["d7_K12"] for s in after]
        b1[r] = {"full_depth_step": fs, "EA_at_full": tr[str(after[0])]["end_activity"] if after else None,
                 "EA_full_plus_1000": tr[str(early)]["end_activity"] if early else None, "EA_final": fin,
                 "class_set_by_full_plus_1000": (early is not None and (tr[str(early)]["end_activity"] >= T) == (fin >= T)),
                 "spearman_EA_d7_after_full": (float(np.corrcoef(np.argsort(np.argsort(ea)), np.argsort(np.argsort(d7)))[0, 1])
                                               if len(after) >= 4 else None)}
    out["B1"] = b1
    # addendum E: branching (branch runs are named <run>_br{st2,st4,full}_{1,2})
    E = {}
    for tag in ("st2", "st4", "full"):
        rows = {}
        for d in sorted(glob.glob(f"runs/dev6_s*_br{tag}_*")):
            b = os.path.basename(d); orig = b.split("_br")[0]
            o_ea, b_ea = ea6(orig), ea6(b)
            pr = P5.pilot_run(b)
            rows[b] = {"orig_EA": o_ea, "EA": b_ea, "final_stage": final_stage(b), "gate": pr.get("gate"),
                       "extrap": pr.get("extrap_K12"), "orig_extrap": D.get(orig, {}).get("extrap_K12"),
                       "same_class": (None if b_ea is None or o_ea is None else (b_ea >= T) == (o_ea >= T))}
        comp = [r for r in rows.values() if r["gate"] and r["same_class"] is not None]
        k = sum(r["same_class"] for r in comp)
        n = len(comp)
        E[tag] = {"runs": rows, "agree": f"{k}/{n}",
                  "reading": None if n == 0 else "set" if k / n >= 7 / 8 else "open" if k / n <= 5 / 8 else "partial"}
    out["E1"] = E
    out.update(score_prec(ea6))
    json.dump(out, open(OUT, "w"), indent=1, default=float)
    brief = {k: out[k] for k in ("P26b", "P26b_bf16_only", "gate")}
    brief["A1"] = {k: v for k, v in out["A1"].items() if k != "runs"}
    brief["A2"] = {k: v for k, v in out["A2"].items()}
    brief["strict_clock"] = {k: v for k, v in out["strict_clock"].items() if k != "per_model"}
    brief["F"] = {k: v for k, v in out["F"].items() if k not in ("pairs", "arm_2070")}
    brief["H"] = {k: v for k, v in out["H"].items() if k != "pairs"}
    print(json.dumps(brief, indent=1, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["measure", "traj", "score"])
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    runs = [r if r.startswith("runs/") else f"runs/{r}" for r in a.runs]
    if a.cmd == "measure":
        measure(runs, a.device)
    elif a.cmd == "traj":
        traj(runs, a.device)
    else:
        score()
