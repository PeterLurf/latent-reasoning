"""§240 pilot extraction (Audit: commit only the norm / ratio fields; no eval, no accuracy read): from scratch/_tmp/s240/pilot_{ctrl,fh}/log.jsonl keep ONLY
the step-0 norm_guard / hold_ratios record, each 1k bin's "hold" dict, and the 100-step records' gn_plain_mean100 / gn_preclip_mean100 / step -> scratch/gate_s240/pilot.json.
Every other field (loss, answer_ce, by_d acc, aux, ...) is dropped unread."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); o = {}
for arm in ("ctrl", "fh"):
    p = os.path.join(ROOT, "scratch", "_tmp", "s240", "pilot_" + arm, "log.jsonl"); rows = [json.loads(l_) for l_ in open(p)] if os.path.exists(p) else []
    step0 = [{k: r[k] for k in ("norm_guard", "hold_ratios")} for r in rows if r.get("step") == 0 and "norm_guard" in r]
    bins = [{"step": r["step"], "hold": r.get("hold")} for r in rows if "by_d" in r]
    gn = [{"step": r["step"], "gn_plain_mean100": r.get("gn_plain_mean100"), "gn_preclip_mean100": r.get("gn_preclip_mean100")} for r in rows if "gn_plain_mean100" in r]
    o[arm] = {"n_rows": len(rows), "step0": step0[0] if step0 else None, "bins": bins, "gn": gn}
    if step0 and bins and bins[-1]["hold"]:
        z = step0[0]["norm_guard"]; b = bins[-1]["hold"]["norm_guard"]
        o[arm]["norm_rel_change_last_bin"] = {k: (b[k] - z[k]) / z[k] for k in ("s8", "sd2")}
o["def"] = ("§240 norm-guard pilot (Audit G-D 184f60c7): df_plaw4_s1 init, the real 8k schedule, 1,000 steps (bin 1 = steps 1-1000; penalty steps 500-1000), micro 8, "
            "the 4060; norm / ratio / grad-norm fields only")
os.makedirs(os.path.join(ROOT, "scratch", "gate_s240"), exist_ok=True)
json.dump(o, open(os.path.join(ROOT, "scratch", "gate_s240", "pilot.json"), "w", newline="\n"), indent=1); print("PILOT EXTRACTED")
