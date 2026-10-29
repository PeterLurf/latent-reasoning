"""§243 pilot extraction (DC_v1 §2 fix 6: read only the gn_plain windows and the per-d norm guard): from scratch/_tmp/s243/pilot_{dc,ctrl}/log.jsonl keep ONLY
the step-0 norm_guard / hold_ratios record, each 1k bin's hold.norm_guard_by_d + hold.norm_guard, and the 100-step records' gn_plain_mean100 / gn_preclip_mean100 / step
-> scratch/gate_s243/pilot.json. Every other field (loss, answer_ce, by_d acc, aux, ...) is dropped unread."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); o = {}
for arm in ("dc", "ctrl"):
    p = os.path.join(ROOT, "scratch", "_tmp", "s243", "pilot_" + arm, "log.jsonl"); rows = [json.loads(l_) for l_ in open(p)] if os.path.exists(p) else []
    step0 = [{k: r[k] for k in ("norm_guard", "hold_ratios")} for r in rows if r.get("step") == 0 and "norm_guard" in r]
    bins = [{"step": r["step"], "norm_guard": (r.get("hold") or {}).get("norm_guard"), "norm_guard_by_d": (r.get("hold") or {}).get("norm_guard_by_d")} for r in rows if "by_d" in r]
    gn = [{"step": r["step"], "gn_plain_mean100": r.get("gn_plain_mean100"), "gn_preclip_mean100": r.get("gn_preclip_mean100")} for r in rows if "gn_plain_mean100" in r]
    o[arm] = {"n_rows": len(rows), "step0": step0[0] if step0 else None, "bins": bins, "gn": gn}
o["def"] = ("§243 LP0 pilot (DC_v1 §2 fix 6): lh_fh_s1 init, the real 8k schedule (hold_start 500), think_k 10, 1,000 steps (bin 0 = steps 1-1000; penalty steps 500-1000), "
            "micro 16, the 4060; DC d_max 6 / CTRL d_max 4; gn and norm-guard fields only (s8 = ||S_K||, K 10)")
os.makedirs(os.path.join(ROOT, "scratch", "gate_s243"), exist_ok=True)
json.dump(o, open(os.path.join(ROOT, "scratch", "gate_s243", "pilot.json"), "w", newline="\n"), indent=1); print("PILOT EXTRACTED")
