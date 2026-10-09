"""THEORY §168 scoring (rules fixed in §168; written before any §168 implementation or run; Theory is the only scorer).
DEEP-ProsQA, step 1: DIRECT = GPT-2 small answering in one pass (the graph + question in one context -> the answer; no CoT, no latents) on prosqa/p8data
(make_deep.py: train depth 3-10, test 3-14), standard (p8_*) and shortcut-free (p8_na_*), seeds 1-2 each; candidate accuracy (chance 0.5) by answer
depth on the test file. Per seed D*_s = the smallest depth in 3..14 with accuracy < 0.90; D* = the lower of the two seeds ("boundary uncertain" if they
differ); D* = 3 reads "no single-pass regime". The NA boundary governs; standard is descriptive (the proximity shortcut, §57).
Step 2 (the latent arm, §169's INJECT harness) is scored by its own section once registered.
Dirs boxes/runs_fsq/s168_direct_{std,na}_s{1,2}.
    python theory/score_s168.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "boxes", "runs_fsq")
DEPTHS = list(range(3, 15))
BASE = {"arm": "direct", "steps": 24000}


def load(d, f="eval.json"):
    p = os.path.join(R, d, f)
    return json.load(open(p)) if os.path.exists(p) else None


def seed_boundary(d, data):
    r = load(d); a = load(d, "args.json") or {}
    if r is None:
        return {"status": "running" if a else "open"}
    bad = {k: a.get(k) for k, v in dict(BASE, data=data).items() if str(a.get(k)) != str(v)}
    if bad:
        return {"status": "error", "error": bad}
    ab = r.get("acc_by_depth") or {}
    fail = [L for L in DEPTHS if str(L) in ab and float(ab[str(L)]) < 0.90]
    return {"status": "done", "acc_by_depth": ab, "D_star": fail[0] if fail else None, "acc_ctl": r.get("acc_ctl")}


def best_step(abs_):
    """note 168-3: the validation-best eval_at step (ties -> the earlier step); never test-selected."""
    st = sorted((int(k), v) for k, v in (abs_ or {}).items() if isinstance(v, dict) and v.get("valid") is not None)
    if not st:
        return None, None
    bv = max(v["valid"] for _, v in st)
    k, v = next((k, v) for k, v in st if v["valid"] == bv)
    return k, v


def sanity_gate():
    """note 168-3: G1 (reproduce GPT-2 No-CoT on standard ProsQA, 0.75 +- 0.05 at the valid-best step) and G2 (p8 std depth 3, diagnostic)."""
    out = {}
    for tag, d, data in (("g1", "s168_g1_prosqa_s1", "prosqa_std"), ("g2", "s168_g2_p8std_s3", "std")):
        r = load(d); a = load(d, "args.json") or {}
        if r is None:
            out[tag] = {"status": "running" if a else "open"}; continue
        bad = {k: a.get(k) for k, v in dict(BASE, data=data).items() if str(a.get(k)) != str(v)}
        if bad:
            out[tag] = {"status": "error", "error": bad}; continue
        abs_ = r.get("acc_by_step") or {}; k, v = best_step(abs_)
        fin = abs_.get("24000") or {}
        out[tag] = {"status": "done", "best_step": k, "test_at_best": (v or {}).get("test"), "valid_at_best": (v or {}).get("valid"),
                    "depth3_at_best": ((v or {}).get("acc_by_depth") or {}).get("3"), "test_final": fin.get("test", r.get("acc")),
                    "decay": None if v is None or fin.get("test") is None else bool(fin["test"] < v["test"] - 0.05),
                    "acc_by_step": {s_: {kk: vv for kk, vv in e.items() if kk != "acc_by_depth"} for s_, e in abs_.items()}}
    g1, g2 = out.get("g1", {}), out.get("g2", {})
    if g1.get("status") != "done" or g2.get("status") != "done":
        out["branch"] = "pending (G1 %s, G2 %s)" % (g1.get("status"), g2.get("status"))
    elif g1["test_at_best"] is None or g1["test_at_best"] < 0.70:
        out["branch"] = "TRAINER OFF: G1 %s < 0.70 at the valid-best step; D* is not readable (a dated note fixes the trainer)" % g1["test_at_best"]
    elif g1["decay"] or g2["decay"]:
        out["branch"] = "DECAY: the 24k recipe over-fits (G1 decay %s, G2 decay %s); NA s1 / s2 re-run with a held-out validation split, D* at each seed's valid-best step" % (g1["decay"], g2["decay"])
    else:
        out["branch"] = "STAND: the registered DIRECT runs stand; D* is read from them"
    return out


def main():
    res = {}
    gate = sanity_gate(); res["sanity_gate_168_3"] = gate
    print("SANITY GATE (168-3):", gate.get("branch"), {t: {k: gate[t].get(k) for k in ("status", "best_step", "test_at_best", "depth3_at_best", "test_final", "decay")} for t in ("g1", "g2")})
    for data in ("na", "std"):
        seeds = {s: seed_boundary("s168_direct_%s_s%d" % (data, s), data) for s in (1, 2)}
        done = [v for v in seeds.values() if v.get("status") == "done"]
        if len(done) < 2:
            b = {"status": "pending (%d / 2 seeds)" % len(done)}
        else:
            ds = [v["D_star"] for v in done]; fin = [d for d in ds if d is not None]
            ls = min(fin) if fin else None
            b = {"status": "fixed", "D_star": ls, "per_seed": ds, "boundary_uncertain": ds[0] != ds[1],
                 "no_single_pass_regime": ls == 3, "passes_everywhere": ls is None}
        res[data] = {"seeds": seeds, "boundary": b}
        print(data.upper(), "|", b, "|", {s: v.get("acc_by_depth") for s, v in seeds.items() if v.get("status") == "done"})
    gov = res["na"]["boundary"]
    if not str(gate.get("branch", "")).startswith("STAND"):
        gov = dict(gov, readable=False, why="note 168-3 gate: %s" % gate.get("branch"))
    print("GOVERNING (NA):", gov)
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "score_s168.json"), "w", newline=chr(10)), indent=1, default=float)


if __name__ == "__main__":
    main()
