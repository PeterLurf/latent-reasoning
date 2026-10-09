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
    bad = {k: a.get(k) for k, v in dict(BASE, data=data).items() if a.get(k) != v}
    if bad:
        return {"status": "error", "error": bad}
    ab = r.get("acc_by_depth") or {}
    fail = [L for L in DEPTHS if str(L) in ab and float(ab[str(L)]) < 0.90]
    return {"status": "done", "acc_by_depth": ab, "D_star": fail[0] if fail else None, "acc_ctl": r.get("acc_ctl")}


def main():
    res = {}
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
    print("GOVERNING (NA):", gov)
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "score_s168.json"), "w", newline=chr(10)), indent=1, default=float)


if __name__ == "__main__":
    main()
