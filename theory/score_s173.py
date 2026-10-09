"""THEORY §173 scoring (rules fixed in §173; written before any §165 K16 / K8, §169, §170 or §171 output is read; Theory is the only scorer).
CONV-HALT (eval only): a(t) = the decoder's greedy answer read from M_t (a(0) = the K' = 0 answer); h(x) = the first t >= 2 with a(t) = a(t-1) = a(t-2),
capped at 2K; the answer is a(h). Rule key "answer2"; "read_0.01" / "delta_0.01" / "full_0.01" (memory-change criteria) are descriptive (Audit: the
additive heads' norm growth biases the full-state relative change toward halting).
HALT-EQUAL(run): at the training-distribution size (4x4; §170: pooled n* <= n <= 8 over the halt grid), accuracy at h >= fixed-K accuracy - 0.01 with mean h < K.
ADAPTIVE(run): size-means of h non-decreasing and largest >= smallest + 1 pass; read only if fixed-K accuracy at the smallest size >= 0.90.
C5(arm): HALT-EQUAL and ADAPTIVE in >= 2 / 3 seeds.
Fields: eval.json `halt` / `halt_fixed_k`, or eval_s173_halt.json (post hoc, same fields). §170 runs stay blinded until score_s170.N_STAR_FIXED is set.
    python theory/score_s173.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_s170 as S170  # noqa: E402  (n*, and §170's blinding)

R = os.path.join(ROOT, "boxes", "runs_fsq")
RULE = "answer2"
MULT = ["4x4", "4x6", "4x8", "4x16"]
NEST = ["2", "4", "6", "8", "10", "12", "16"]
ARMS = {"s165_k16": (16, "s165_k16_4x4_s%d", MULT), "s165_k8": (8, "s165_k8_4x4_s%d", MULT),
        "s169_k16": (16, "s169_k16_4x4_s%d", MULT), "s169_k8": (8, "s169_k8_4x4_s%d", MULT),
        "s170_k8": (8, "s170_k8_s%d", NEST), "s170_k16": (16, "s170_k16_s%d", NEST),
        "s171_k8": (8, "s171_k8_4x4_s%d", MULT)}


def load(d, f):
    p = os.path.join(R, d, f)
    return json.load(open(p)) if os.path.exists(p) else None


def fields(d):
    h = load(d, "eval_s173_halt.json")
    if h is None:
        h = load(d, "eval.json")
    if not h or "halt" not in h:
        return None
    return h.get("halt") or {}, h.get("halt_fixed_k") or {}


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def score_run(K, d, sizes, train_sizes):
    f = fields(d)
    if f is None:
        return {"status": "open"}
    halt, fixed = f; he = halt.get(RULE) or {}
    sz = [s for s in sizes if s in he and s in fixed]
    if not sz:
        return {"status": "no fields"}
    tr = [s for s in train_sizes if s in he and s in fixed]
    a_h = mean([he[s].get("acc") for s in tr]); a_k = mean([fixed[s] for s in tr]); mh = mean([he[s].get("mean_h") for s in tr])
    halt_equal = None if None in (a_h, a_k, mh) else bool(a_h >= a_k - 0.01 and mh < K)
    means = [he[s].get("mean_h") for s in sz]
    adaptive = None
    if fixed.get(sz[0]) is not None and fixed[sz[0]] >= 0.90 and None not in means and len(means) >= 2:
        adaptive = bool(all(means[i + 1] >= means[i] for i in range(len(means) - 1)) and means[-1] >= means[0] + 1)
    return {"status": "done", "train_acc_halt": a_h, "train_acc_fixed": a_k, "train_mean_h": mh, "halt_equal": halt_equal, "adaptive": adaptive,
            "mean_h_by_size": dict(zip(sz, means)), "acc_halt_by_size": {s: he[s].get("acc") for s in sz}, "acc_fixed_by_size": {s: fixed[s] for s in sz},
            "frac_capped_by_size": {s: he[s].get("frac_capped") for s in sz}}


def main():
    res = {}
    for arm, (K, pat, sizes) in ARMS.items():
        if arm.startswith("s170"):
            ns_ = S170.N_STAR_FIXED
            if ns_ is None:
                res[arm] = {"status": "blinded (§170 n* note not committed)"}; print(arm, "| blinded"); continue
            train = [s for s in NEST if ns_ <= int(s) <= 8]
        else:
            train = ["4x4"]
        runs = {s: score_run(K, pat % s, sizes, train) for s in (1, 2, 3)}
        done = [v for v in runs.values() if v["status"] == "done"]
        he = sum(1 for v in done if v["halt_equal"]); ad = sum(1 for v in done if v["adaptive"])
        verdict = "C5 (HALT-EQUAL %d, ADAPTIVE %d)" % (he, ad) if he >= 2 and ad >= 2 else (
            "not C5 (HALT-EQUAL %d, ADAPTIVE %d of %d)" % (he, ad, len(done)) if len(done) == 3 or len(done) - min(he, ad) >= 2 else "pending (%d / 3)" % len(done))
        res[arm] = {"runs": runs, "verdict": verdict}
        print(arm, "|", verdict, "|", {s: {k: v.get(k) for k in ("train_acc_halt", "train_acc_fixed", "train_mean_h", "mean_h_by_size")} for s, v in runs.items() if v["status"] == "done"})
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "score_s173.json"), "w", newline=chr(10)), indent=1, default=float)


if __name__ == "__main__":
    main()
