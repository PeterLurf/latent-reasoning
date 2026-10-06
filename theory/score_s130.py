"""THEORY §130 scoring (rules fixed in §130; written before any §130 replay; Theory is the only scorer).
Replays: theory/results/replay130/{card}_s{seed}.jsonl from scratch/replay_basin.py (bf16, --steps 2000 --every 50), card in
{"4060", "3070"}. FIT = the 4060 seeds (3..10, extended 19..26 if needed); TEST = the 3070 seeds (11..18, extended 27..34).
Per seed: FLAG = 123-3's flag on the replay's own loss (nearest logged steps to 100 / 1k / 2k); PULL = the first probe step with
L1_dmean > 0.05; COLLAPSE = zb_std_in < 0.04 at two consecutive probes at or after step 600; marker(theta, window) = zb_std_in <
theta at two consecutive probes inside the window.
FIT (on FIT seeds only): choose (theta, window) from THETAS x WINDOWS maximising min(precision, recall) of the marker against FLAG;
ties -> the earlier window end, then the smaller theta. The choice is frozen in THEORY (note 130-1) before any TEST seed is read:
    python theory/score_s130.py fit
TEST: the frozen marker's precision and recall against FLAG on TEST seeds; PASS = both >= 0.9 (needs >= 3 flagged and >= 3
non-flagged TEST seeds; otherwise "incomplete"):
    python theory/score_s130.py test <theta> <window_lo> <window_hi>
H-127-7 (descriptive, all seeds): every FLAG seed has COLLAPSE; every non-FLAG seed reaches L1_dmean >= 0.1 by 2k.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "theory", "results", "replay130")
FIT_SEEDS = list(range(3, 11)) + list(range(19, 27))
TEST_SEEDS = list(range(11, 19)) + list(range(27, 35))
THETAS = (0.03, 0.04, 0.05, 0.06, 0.08)
WINDOWS = ((600, 1000), (700, 1000), (700, 1200), (600, 1500))


def load(card, seed):
    p = os.path.join(D, "%s_s%d.jsonl" % (card, seed))
    if not os.path.exists(p):
        return None
    rows = [json.loads(x) for x in open(p) if x.strip()]
    rows = [r for r in rows if "step" in r]
    return rows if rows and max(r["step"] for r in rows) >= 2000 else None


def features(rows):
    L = {r["step"]: r["loss"] for r in rows if r.get("loss") is not None}
    g = lambda t: L[min(L, key=lambda s: abs(s - t))]  # noqa: E731
    flag = bool(g(2000) >= 0.9 * g(1000) and g(2000) >= 0.25 * g(100))
    pr = [r for r in rows if r.get("zb_std_in") is not None]
    pull = next((r["step"] for r in pr if (r.get("L1_dmean") or 0) > 0.05), None)
    low = [r["step"] for i, r in enumerate(pr[1:], 1) if r["step"] >= 600 and r["zb_std_in"] < 0.04 and pr[i - 1]["zb_std_in"] < 0.04]
    dmax = max([(r.get("L1_dmean") or 0) for r in pr if r["step"] <= 2000] or [0])
    return {"flag": flag, "pull": pull, "collapse": bool(low), "collapse_step": low[0] if low else None, "dmean_max": dmax,
            "loss_1k": g(1000), "loss_2k": g(2000), "probes": [(r["step"], r["zb_std_in"]) for r in pr]}


def marker(f, theta, lo, hi):
    pr = [(s, z) for s, z in f["probes"] if lo <= s <= hi]
    return any(pr[i][1] < theta and pr[i - 1][1] < theta for i in range(1, len(pr)))


def prf(fs, theta, lo, hi):
    tp = sum(1 for f in fs if f["flag"] and marker(f, theta, lo, hi)); fp = sum(1 for f in fs if not f["flag"] and marker(f, theta, lo, hi))
    fn = sum(1 for f in fs if f["flag"] and not marker(f, theta, lo, hi))
    prec = tp / (tp + fp) if tp + fp else None; rec = tp / (tp + fn) if tp + fn else None
    return prec, rec


def collect(card, seeds):
    return {s: features(r) for s in seeds for r in [load(card, s)] if r is not None}


def h1277(fs):
    return {"flag_seeds_with_collapse": "%d / %d" % (sum(f["collapse"] for f in fs if f["flag"]), sum(f["flag"] for f in fs)),
            "nonflag_seeds_with_pull_0.1": "%d / %d" % (sum(f["dmean_max"] >= 0.1 for f in fs if not f["flag"]), sum(not f["flag"] for f in fs))}


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "fit"
    if mode == "fit":
        F = collect("4060", FIT_SEEDS); fs = list(F.values())
        print("FIT seeds", {s: (f["flag"], f["pull"], f["collapse_step"], round(f["dmean_max"], 3)) for s, f in F.items()})
        nf, nn = sum(f["flag"] for f in fs), sum(not f["flag"] for f in fs)
        if nf < 3 or nn < 3:
            print("FIT incomplete: %d flagged, %d non-flagged (need >= 3 each; extend with seeds 19..26)" % (nf, nn)); return
        best = None
        for (lo, hi) in WINDOWS:
            for th in THETAS:
                p, r = prf(fs, th, lo, hi); sc = min(p or 0, r or 0)
                key = (-sc, hi, th)
                if best is None or key < best[0]:
                    best = (key, th, lo, hi, p, r)
        _, th, lo, hi, p, r = best
        print("FIT choice: theta %s window %d-%d | fit precision %s recall %s" % (th, lo, hi, p, r)); print("H-127-7 (FIT)", h1277(fs))
        json.dump({"mode": "fit", "seeds": {s: {k: v for k, v in f.items() if k != "probes"} for s, f in F.items()}, "theta": th, "window": [lo, hi],
                   "precision": p, "recall": r, "h1277": h1277(fs)}, open(os.path.join(ROOT, "theory", "results", "score_s130_fit.json"), "w", newline="\n"), indent=1)
    else:
        th, lo, hi = float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
        T = collect("3070", TEST_SEEDS); fs = list(T.values())
        nf, nn = sum(f["flag"] for f in fs), sum(not f["flag"] for f in fs)
        p, r = prf(fs, th, lo, hi)
        ok = (nf >= 3 and nn >= 3)
        verdict = (bool(p is not None and r is not None and p >= 0.9 and r >= 0.9) if ok else
                   "incomplete (%d flagged, %d non-flagged; extend with seeds 27..34)" % (nf, nn))
        print("TEST seeds", {s: (f["flag"], marker(f, th, lo, hi), f["collapse_step"]) for s, f in T.items()})
        print("TEST precision %s recall %s | PASS %s" % (p, r, verdict)); print("H-127-7 (TEST)", h1277(fs))
        json.dump({"mode": "test", "theta": th, "window": [lo, hi], "seeds": {s: {k: v for k, v in f.items() if k != "probes"} for s, f in T.items()},
                   "precision": p, "recall": r, "PASS": verdict, "h1277": h1277(fs)}, open(os.path.join(ROOT, "theory", "results", "score_s130_test.json"), "w", newline="\n"), indent=1)


if __name__ == "__main__":
    main()
