"""§136 trigger record (Audit 2026-10-07): for every fetched s136 run dir, the widenings the run logged and the trigger steps recomputed
independently from its raw logged losses with fsq120.s136_fire (sc 1), so a fire can be checked against the loss records.
    python scratch/s136_triggers.py        -> boxes/runs_fsq/s136_triggers.json (+ one line per run on stdout)"""
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from boxes.fsq120 import s136_fire  # noqa: E402


def one(d):
    a = json.load(open(os.path.join(d, "args.json"))); recs = [json.loads(l) for l in open(os.path.join(d, "log.jsonl")) if l.strip()]
    L = {r["step"]: r["loss"] for r in recs if "loss" in r and "step" in r}; thv = 2.5 if a.get("grow_trigger") in (None, "plateau") else float(a["grow_trigger"])
    thr = plat = None
    for t in sorted(L):                                                     # every logged step, as the run checks them (first hit)
        sub = {s: v for s, v in L.items() if s <= t}; th, pl = s136_fire(sub, t, 1.0, thv)
        thr = t if (th and thr is None) else thr; plat = t if (pl and plat is None) else plat
    ev = [{"step": r["step"], "from": r["grow"]["from"], "to": r["grow"]["to"], "loss_at_step": L.get(r["step"])} for r in recs if "grow" in r]
    e = json.load(open(os.path.join(d, "eval.json"))) if os.path.exists(os.path.join(d, "eval.json")) else {}
    s = e.get("s136") or {}
    used = "plateau" if a.get("grow_trigger") == "plateau" else ("threshold" if a.get("grow") else None)
    first = ev[0]["step"] if ev else None; expect = {"plateau": plat, "threshold": thr}.get(used)
    return {"run": os.path.basename(d), "grow": a.get("grow"), "trigger": a.get("grow_trigger"), "staged": a.get("grow_staged"), "nested": a.get("nested"),
            "gpu": a.get("train_gpu"), "driver": a.get("train_driver"), "last_step": max(L) if L else None, "events": ev,
            "recomputed": {"threshold": thr, "plateau": plat}, "run_reported": s.get("trigger_steps"), "width_final": s.get("width_final"),
            "first_fire_matches_recomputed": None if not used else (first == expect), "evaluated": bool(e)}


def main():
    out = []
    for d in sorted(glob.glob(os.path.join(ROOT, "boxes", "runs_fsq", "s136_*_s5_s*"))):  # void attempts (host cuts) kept, marked (Audit 02:48)
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "log.jsonl")):
            r = one(d); r["void"] = os.path.basename(d).split(".", 1)[1] if "." in os.path.basename(d) else None; out.append(r)
    json.dump(out, open(os.path.join(ROOT, "boxes", "runs_fsq", "s136_triggers.json"), "w", newline="\n"), indent=1)
    for r in out:
        print(r["run"], "events", [(e["step"], e["from"], e["to"]) for e in r["events"]], "recomputed", r["recomputed"], "match", r["first_fire_matches_recomputed"])


if __name__ == "__main__":
    main()
