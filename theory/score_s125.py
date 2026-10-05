"""THEORY §125 scoring (rules fixed in §125; written before any §125 run; Theory is the only scorer).
Runs: boxes/runs_fsq/s125_s5_std_s{1..4}, s125_ff_std_s{1,2} (STD) and s125_s5_ctrl_s{3,4} (concurrent §120 FINAL control);
§120 FINAL S5 s1/s2 (boxes/runs_fsq/s120_s5_final_s{1,2}) are the existing control seeds.
Plateau flag (note 123-3, label-free, from log.jsonl): loss(2k) >= 0.9 loss(1k) and loss(2k) >= 0.25 loss(100).
PLATEAU-FIX: STD S5 plateau count <= 1/4 AND seed 2 not plateaued.  ACC: STD S5 long_acc >= 0.99 in >= 3/4.
FF guard: FF long_acc >= 0.99 in 2/2 and SERIAL (donor >= 0.90, recipient - chance_overlap < 0.10).
    python theory/score_s125.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "boxes", "runs_fsq")


def load(d, f="eval.json"):
    p = os.path.join(R, d, f)
    return json.load(open(p)) if os.path.exists(p) else None


def plateau(d):
    p = os.path.join(R, d, "log.jsonl")
    if not os.path.exists(p):
        return None
    L = {}
    for line in open(p):
        x = json.loads(line)
        if "loss" in x and "step" in x:
            L[x["step"]] = x["loss"]
    if not L:
        return None
    g = lambda t: L[min(L, key=lambda s: abs(s - t))]  # noqa: E731
    return bool(g(2000) >= 0.9 * g(1000) and g(2000) >= 0.25 * g(100))


def serial(d):
    return bool(d) and all(v["donor_consistent"] >= 0.90 and v["recipient_consistent"] - v.get("chance_overlap", 0.0) < 0.10 for v in d.values())


def cell(d):
    r = load(d)
    if r is None:
        return None
    return {"long_acc": r.get("long_acc"), "plateau": plateau(d), "serial": serial(r.get("donor")), "carried_norm": r.get("carried_norm"),
            "codes_per_state": r.get("codes_per_state"), "final_loss": r.get("final_loss")}


def main():
    std = {s: cell("s125_s5_std_s%d" % s) for s in (1, 2, 3, 4)}; std = {k: v for k, v in std.items() if v}
    ctrl = {s: cell("s125_s5_ctrl_s%d" % s) for s in (3, 4)}; ctrl.update({s: cell("s120_s5_final_s%d" % s) for s in (1, 2)}); ctrl = {k: v for k, v in ctrl.items() if v}
    ff = {s: cell("s125_ff_std_s%d" % s) for s in (1, 2)}; ff = {k: v for k, v in ff.items() if v}
    n_pl = sum(bool(v["plateau"]) for v in std.values())
    res = {"STD_S5": std, "CONTROL_S5": ctrl, "FF_STD": ff,
           "PLATEAU_FIX": (len(std) == 4 and n_pl <= 1 and 2 in std and not std[2]["plateau"]) if len(std) == 4 else "incomplete (%d / 4)" % len(std),
           "ACC": (sum((v["long_acc"] or 0) >= 0.99 for v in std.values()) >= 3) if len(std) == 4 else "incomplete",
           "FF_GUARD": (len(ff) == 2 and all((v["long_acc"] or 0) >= 0.99 and v["serial"] for v in ff.values())) if len(ff) == 2 else "incomplete",
           "plateau_counts": {"STD": n_pl, "CONTROL": sum(bool(v["plateau"]) for v in ctrl.values()), "CONTROL_n": len(ctrl)}}
    for k in ("STD_S5", "CONTROL_S5", "FF_STD"):
        print(k, {s: (v["long_acc"], v["plateau"], v["serial"], v["carried_norm"]) for s, v in res[k].items()})
    print("PLATEAU-FIX", res["PLATEAU_FIX"], "| ACC", res["ACC"], "| FF guard", res["FF_GUARD"], "| plateau counts", res["plateau_counts"])
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "score_s125.json"), "w", newline="\n"), indent=1, default=float)


if __name__ == "__main__":
    main()
