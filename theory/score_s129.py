"""THEORY §129 scoring (rules fixed in §129; written before any §129 run; Theory is the only scorer).
Runs: boxes/runs_fsq/s129_s5_s{seed} (§120 FINAL S5, RTX 3090 / 3090 Ti). Phase 1 = the first 2k steps (loss lines without
"phase", or "phase": 1). If 123-3's flag fires at 2k, the head (ln, W_down, W_up) is re-drawn with fresh Adam moments for those
parameters; the body and its Adam state are kept; a {"redraw": true} record is written. Phase 2 = loss lines with "phase": 2
and "clock" (steps since the re-draw; the schedule restarts). The flag is applied again at clock 2k; then 24k phase-2 steps
end with eval.json (long_acc).
PANEL: the first 4 seeds, in the fixed order 2, 1, 3, 4, 101..104, 201..204, ..., whose phase-1 flag fires; a seed that does not flag
is stopped at 2k, reported, and not in the panel.
CLOCK-ONLY control (Literature 3fd1f1fe): from the same 2k checkpoint (model + both optimizers' state), restart the schedule
clock with no re-draw; dir s129_s5_s{seed}_clock, phase-2 lines only (clock), flag at clock 2k, 24k if it escapes.
PASS(arm, seed): the phase-2 flag does not fire AND long_acc >= 0.99.  ESC(arm, seed): the phase-2 flag does not fire.
Verdicts, in this order: LR = CLOCK ESC in >= 3/4 (the warm restart alone kicks the plateau; REDRAW reported beside);
HEAD = REDRAW PASS in >= 3/4 AND REDRAW ESC - CLOCK ESC >= 2 seeds (Audit); BODY = REDRAW's flag fires in 4/4 (by 2k the body ignores a fresh
head); otherwise descriptive (1-2/4 REDRAW passes are consistent with a head-init lottery at the fresh-seed rate).
    python theory/score_s129.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "boxes", "runs_fsq")
NL = chr(10)
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


def arm2(d):
    """Phase-2-only run (CLOCK-ONLY, resumed from the 2k checkpoint)."""
    rs = rows(d)
    if rs is None:
        return None
    p2 = {x["clock"]: x["loss"] for x in rs if "loss" in x and x.get("phase") == 2}
    f2 = flag(p2); ev = json.load(open(os.path.join(R, d, "eval.json"))) if os.path.exists(os.path.join(R, d, "eval.json")) else None
    out = {"phase2_flag": f2, "long_acc": ev.get("long_acc") if ev else None}
    if f2 is None:
        out["status"] = "running (before clock 2k)"
    elif f2:
        out["status"] = "re-plateaued"; out["esc"] = False; out["pass"] = False
    else:
        out["esc"] = True
        if ev is None:
            out["status"] = "running (escaped)"
        else:
            out["status"] = "done"; out["pass"] = (ev.get("long_acc") or 0) >= 0.99
    return out


def main():
    dirs = [d for d in ("s129_s5_s%d" % s for s in ORDER) if os.path.isdir(os.path.join(R, d))]
    res = {d: seed_result(d) for d in dirs}
    panel = [d for d in dirs if res[d] and res[d]["phase1_flag"]][:4]
    clk = {d: arm2(d + "_clock") for d in panel}
    for d in panel:
        if res[d].get("phase2_flag") is not None:
            res[d]["esc"] = not res[d]["phase2_flag"]
    def count(m, key):
        return sum(1 for v in m if v and v.get(key) is True)
    red = [res[d] for d in panel]; ck = [clk[d] for d in panel]
    red_done = all("pass" in v for v in red) and len(panel) == 4
    ck_esc_known = all(v and "esc" in v for v in ck) and len(panel) == 4
    ck_done = all(v and "pass" in v for v in ck) and len(panel) == 4
    n_red_pass = count(red, "pass"); n_red_fire = sum(1 for v in red if v.get("phase2_flag") is True)
    n_ck_esc = count(ck, "esc"); n_ck_fire = sum(1 for v in ck if v and v.get("phase2_flag") is True)
    if ck_esc_known and n_ck_esc >= 3:
        verdict = "LR (clock-only escapes %d / 4; REDRAW pass %d / 4)" % (n_ck_esc, n_red_pass)
    elif red_done and ck_esc_known and n_red_pass >= 3 and count(red, "esc") - n_ck_esc >= 2:
        verdict = "HEAD"                                                     # Audit: REDRAW beats CLOCK-ONLY by >= 2 seeds
    elif len(panel) == 4 and n_red_fire == 4 and ck_esc_known:
        verdict = "BODY (by 2k the body ignores a fresh head)"
    elif red_done and ck_done:
        verdict = "descriptive (REDRAW pass %d / 4, CLOCK escape %d / 4)" % (n_red_pass, n_ck_esc)
    else:
        verdict = "incomplete (panel %d / 4)" % len(panel)
    for d in dirs:
        print(d, res[d], "| CLOCK", clk.get(d))
    print("PANEL", panel, "| REDRAW pass", n_red_pass, "fire", n_red_fire, "| CLOCK esc", n_ck_esc, "fire", n_ck_fire, "| VERDICT", verdict)
    json.dump({"runs": res, "clock": clk, "panel": panel, "verdict": verdict}, open(os.path.join(ROOT, "theory", "results", "score_s129.json"), "w", newline=NL), indent=1)


if __name__ == "__main__":
    main()
