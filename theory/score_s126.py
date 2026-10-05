"""THEORY §126 scoring (rules fixed in §126; written before any §126 run; Theory is the only scorer).
Runs: boxes/runs_fsq/s126_s5_rawh_s{1..4}, s126_s5_anneal_s{1..4} (eval.json: long_acc, donor, carried_norm; log.jsonl).
Plateau flag (123-3): loss(2k) >= 0.9 loss(1k) and loss(2k) >= 0.25 loss(100).
BOTTLENECK-CAUSE: RAW-H seed 2 not plateaued AND RAW-H plateau count <= 1/4.
ANNEAL-INHERITS (stage 2, only if RAW-H s2 escaped; on the RAW-H non-plateau seeds): ANNEAL long_acc >= 0.99 at
lambda = 1 in all of them.
STOP RULE: RAW-H does not rescue seed 2 AND its plateau count is not below 2/4 -> adopt restart-on-plateau (engineering).
    python theory/score_s126.py
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


def arm(pre):
    out = {}
    for s in (1, 2, 3, 4):
        d = "%s_s%d" % (pre, s); r = load(d)
        if r is not None:
            out[s] = {"long_acc": r.get("long_acc"), "plateau": plateau(d), "gpu": r.get("train_gpu", r.get("gpu")), "carried_norm": r.get("carried_norm")}
    return out


def main():
    raw, ann = arm("s126_s5_rawh"), arm("s126_s5_anneal")
    n_pl = sum(bool(v["plateau"]) for v in raw.values())
    full = len(raw) == 4
    bc = (full and 2 in raw and not raw[2]["plateau"] and n_pl <= 1) if full else "incomplete (%d / 4)" % len(raw)
    nonpl = [s for s, v in raw.items() if not v["plateau"]]
    # Stage 2 runs ANNEAL only if RAW-H s2 escaped, on the RAW-H non-plateau seeds
    if full and (2 not in raw or raw[2]["plateau"]):
        ai = "skipped (RAW-H s2 plateaued; stage 2 not triggered)"
    else:
        ai = (bool(nonpl) and all(s in ann and (ann[s]["long_acc"] or 0) >= 0.99 for s in nonpl)) if all(s in ann for s in nonpl) and nonpl else "incomplete (%d / %d)" % (len(ann), len(nonpl))
    stop = (full and (2 not in raw or raw[2]["plateau"]) and n_pl >= 2) if full else "incomplete"
    res = {"RAW_H": raw, "ANNEAL": ann, "BOTTLENECK_CAUSE": bc, "ANNEAL_INHERITS": ai, "STOP_RULE_FIRES": stop, "rawh_plateau_count": n_pl}
    print("RAW-H", {s: (v["long_acc"], v["plateau"], v["gpu"]) for s, v in raw.items()})
    print("ANNEAL", {s: (v["long_acc"], v["plateau"], v["gpu"]) for s, v in ann.items()})
    print("BOTTLENECK-CAUSE", bc, "| ANNEAL-INHERITS", ai, "| STOP RULE fires", stop)
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "score_s126.json"), "w", newline="\n"), indent=1, default=float)


if __name__ == "__main__":
    main()
