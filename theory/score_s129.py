"""THEORY §129 scoring (rules fixed in §129; written before any §129 run; Theory is the only scorer).
Runs: boxes/runs_fsq/s129_s5_s{seed} (§120 FINAL S5, RTX 3090 / 3090 Ti). Phase 1 = the first 2k steps (loss lines without
"phase", or "phase": 1). If 123-3's flag fires at 2k, the head (ln, W_down, W_up) is re-drawn with fresh Adam moments for those
parameters; the body and its Adam state are kept; a {"redraw": true} record is written. Phase 2 = loss lines with "phase": 2
and "clock" (steps since the re-draw; the schedule restarts). The flag is applied again at clock 2k; then 24k phase-2 steps
end with eval.json (long_acc).
PANEL: the first 4 seeds, in the fixed order 2, 1, 3, 4, 101..104, 201..204, ..., whose phase-1 flag fires; a seed that does not flag
is stopped at 2k, reported, and not in the panel.
PASS(seed): phase-2 flag does not fire AND long_acc >= 0.99.
HEAD: >= 3/4 panel seeds PASS.  BODY: the phase-2 flag fires in all 4.  Otherwise: descriptive (1-2/4 is consistent with a
head-init lottery at the fresh-seed rate).
    python theory/score_s129.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "boxes", "runs_fsq")
ORDER = [2, 1, 3, 4] + [o + k for o in (100, 200, 300, 400) for k in (1, 2, 3, 4)]   # §129: fixed seed order; first 4 flagged = PANEL


def rows(d):
    p = os.path.join(R, d, "log.jsonl")
    return [json.loads(x) for x in open(p)] if os.path.exists(p) else None


def flag(L):
    if not L or max(L) < 2000:
        return None
    g = lambda t: L[min(L, key=lambda s: abs(s - t))]  # noqa: E731
    return bool(g(2000) >= 0.9 * g(1000) and g(2000) >= 0.25 * g(100))


def seed_result(d):
    rs = rows(d)
    if rs is None:
        return None
    p1 = {x["step"]: x["loss"] for x in rs if "loss" in x and x.get("phase", 1) == 1 and x["step"] <= 2000}
    p2 = {x["clock"]: x["loss"] for x in rs if "loss" in x and x.get("phase") == 2}
    f1, f2 = flag(p1), flag(p2)
    ev = None
    if os.path.exists(os.path.join(R, d, "eval.json")):
        ev = json.load(open(os.path.join(R, d, "eval.json")))
    a = json.load(open(os.path.join(R, d, "args.json"))) if os.path.exists(os.path.join(R, d, "args.json")) else {}
    out = {"phase1_flag": f1, "redrawn": any(x.get("redraw") for x in rs), "phase2_flag": f2,
           "long_acc": ev.get("long_acc") if ev else None, "gpu": a.get("train_gpu"), "driver": a.get("train_driver")}
    if f1 is False:
        out["status"] = "not in panel (phase-1 flag did not fire)"
    elif f1 is None:
        out["status"] = "running (phase 1)"
    elif f2 is None:
        out["status"] = "running (phase 2, before clock 2k)"
    elif f2:
        out["status"] = "re-plateaued"; out["pass"] = False
    elif ev is None:
        out["status"] = "running (phase 2, escaped)"
    else:
        out["status"] = "done"; out["pass"] = (ev.get("long_acc") or 0) >= 0.99
    return out


def main():
    dirs = [d for d in ("s129_s5_s%d" % s for s in ORDER) if os.path.isdir(os.path.join(R, d))]
    res = {d: seed_result(d) for d in dirs}
    panel = [d for d in dirs if res[d] and res[d]["phase1_flag"]][:4]
    decided = [d for d in panel if "pass" in res[d]]
    n_pass = sum(res[d]["pass"] for d in decided); n_replat = sum(res[d]["status"] == "re-plateaued" for d in panel)
    if len(panel) == 4 and n_pass >= 3:
        verdict = "HEAD"
    elif len(panel) == 4 and n_replat == 4:
        verdict = "BODY"
    elif len(panel) == 4 and len(decided) == 4:
        verdict = "descriptive (%d / 4 pass)" % n_pass
    else:
        verdict = "incomplete (panel %d / 4, decided %d)" % (len(panel), len(decided))
    for d in dirs:
        print(d, res[d])
    print("PANEL", panel, "| PASS", n_pass, "| re-plateaued", n_replat, "| VERDICT", verdict)
    json.dump({"runs": res, "panel": panel, "verdict": verdict}, open(os.path.join(ROOT, "theory", "results", "score_s129.json"), "w", newline="\n"), indent=1)


if __name__ == "__main__":
    main()
