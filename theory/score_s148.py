"""THEORY §148 scoring (rules fixed in §148; written before any §148 implementation or run; Theory is the only scorer).
LOOP-RETRO (the research program's B3): direct-answer 4x4 multiplication ("a * b =" -> 8 product digits LSB first, no CoT) on GPT-2 small retrofitted as
prelude (blocks 0-3) / weight-tied loop (blocks 4-7, r iterations, input injection P(e) zero-init, damped update eta 0.25 for r > 1) / coda (8-11);
r = 1 to 2k then r ~ U{1..4}; deep supervision (CE at the final r plus one uniform earlier r). Arms (dirs boxes/runs_fsq/s148_{arm}_4x4_s{seed}):
loop4 (the above) and plain (r = 1 always: the direct-answer baseline, same data and budget). One run per seed (no restart slots).
PASS(loop4): >= 3/4 seeds with test_acc >= 0.99 at r = 4. IMPROVES (descriptive): loop4 >= plain + 0.2 on the proof (s1). C2 (descriptive at the proof):
acc at test r = 8 and 16 >= acc at r = 4. Per-loop monotone progress: acc_by_r over r = 1..16 (s148.acc_by_r). C5 (descriptive): convergence halting
(s148.halt: acc, mean r). C1 (descriptive): loop-state swaps (s148.loop_swap).
    python theory/score_s148.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "boxes", "runs_fsq")
ARMS = {"loop4": {"family": "mult_direct", "task": "4x4", "loop": "4-7", "r_max": 4, "eta": 0.25, "deep_sup": True},
        "plain": {"family": "mult_direct", "task": "4x4", "r_max": 1}}
SKIPPED, ERRORS = [], []


def load(d, f="eval.json"):
    p = os.path.join(R, d, f)
    return json.load(open(p)) if os.path.exists(p) else None


def run(arm, seed):
    d = "s148_%s_4x4_s%d" % (arm, seed); r = load(d)
    if r is None:
        return None
    a = load(d, "args.json") or {}
    if int(a.get("steps", 24000)) != 24000 or float(a.get("sched_scale", 1.0)) != 1.0 or str(a.get("device", "")).startswith("cpu") or a.get("mem_check"):
        SKIPPED.append(d); return None
    bad = {k: a.get(k) for k, v in ARMS[arm].items() if a.get(k) != v}
    if bad:
        ERRORS.append((d, bad)); return {"error": bad}
    s8 = r.get("s148") or {}; abr = s8.get("acc_by_r") or {}
    out = {"test_acc": r.get("test_acc"), "acc_by_r": abr, "halt": s8.get("halt"), "loop_swap": s8.get("loop_swap"), "rho_loop": s8.get("rho_loop"),
           "gpu": r.get("train_gpu", r.get("gpu")), "driver": r.get("train_driver")}
    if arm == "loop4" and abr:
        a4, a8, a16 = abr.get("4"), abr.get("8"), abr.get("16")
        out["c2"] = None if None in (a4, a8, a16) else bool(a8 >= a4 and a16 >= a4)
        rs = sorted(int(k) for k in abr if int(k) <= 4)
        out["monotone_1_4"] = all(abr[str(rs[i + 1])] >= abr[str(rs[i])] - 1e-9 for i in range(len(rs) - 1)) if len(rs) > 1 else None
    out["pass"] = bool((out["test_acc"] or 0) >= 0.99) if arm == "loop4" else None
    return out


def main():
    res = {}
    for arm in ARMS:
        runs = {s: run(arm, s) for s in (1, 2, 3, 4)}; res[arm] = {"runs": runs}
        for s, x in runs.items():
            print(arm, "s%d" % s, x)
    lp = {s: x for s, x in res["loop4"]["runs"].items() if x and "error" not in x}
    n = sum(1 for x in lp.values() if x["pass"])
    v = "PASS" if n >= 3 else ("FAIL (%d / 4)" % n if len(lp) - n >= 2 else "incomplete (%d / 4)" % len(lp))
    res["verdict_loop4"] = v
    p1 = (res["plain"]["runs"].get(1) or {}).get("test_acc"); l1 = (res["loop4"]["runs"].get(1) or {}).get("test_acc")
    imp = None if p1 is None or l1 is None else bool(l1 >= p1 + 0.2)
    res["improves_proof"] = imp
    rd = ("looped depth buys implicit computation at this budget (stage 2: removal coupled to loops, a new registration)" if imp
          else "depth alone does not buy direct multiplication at this budget: B3 stops" if imp is False else "pending")
    res["reading"] = rd
    print("VERDICT loop4:", v, "| IMPROVES (proof, loop4 >= plain + 0.2):", imp, "(loop4 %s, plain %s)" % (l1, p1)); print("READING:", rd)
    if SKIPPED:
        print("SKIPPED (smoke):", SKIPPED)
    if ERRORS:
        print("ERRORS:", ERRORS)
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "score_s148.json"), "w", newline=chr(10)), indent=1, default=float)


if __name__ == "__main__":
    main()
