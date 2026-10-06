"""THEORY §130 scoring (rules fixed in §130; written before any §130 replay; Theory is the only scorer).
Replays: theory/results/replay130/{card}_s{seed}.jsonl from scratch/replay_basin.py (bf16, --steps 2000 --every 50), card in
{"4060", "3070"}. FIT = 4060 seeds 3..10 (extended 35..42 if < 3 flagged / non-flagged); TEST-CROSS (primary) = 3070 seeds 11..18,
27..34, 43..58; TEST-SAME (secondary) = 4060 seeds 19..26, 59..82; each TEST block runs until >= 8 flagged and >= 8 non-flagged
(Literature 722b9f4c), and is read only after the freeze note. Clopper-Pearson 95 % CIs are reported; FIT reports the chosen
cell's neighbours.
Per seed: FLAG = 123-3's flag on the replay's own loss (nearest logged steps to 100 / 1k / 2k); PULL = the first probe step with
L1_dmean > 0.05; COLLAPSE = zb_std_in < 0.04 at two consecutive probes at or after step 600; marker(theta, window) = zb_std_in <
theta at two consecutive probes inside the window.
FIT (on FIT seeds only): choose (theta, window) from THETAS x WINDOWS maximising min(precision, recall) of the marker against FLAG;
ties -> the earlier window end, then the smaller theta, then the earlier window start (130-1). The choice is frozen in THEORY (note 130-1) before any TEST seed is read:
    python theory/score_s130.py fit
TEST: the frozen marker's precision and recall against FLAG; PASS (primary = CROSS) = both >= 0.9 with >= 8 flagged and >= 8
non-flagged seeds per block (otherwise "incomplete"); COLLAPSE needs both probes at or after 600:
    python theory/score_s130.py test   (reads theory/results/score_s130_frozen.json, written in the 130-1 freeze commit: the registered
    primary cell dip|flag, plus secondary cells dip|basin, persist|flag and persist|basin; BASIN = FLAG and no PULL by 2k)
H-127-7 (descriptive, all seeds): every FLAG seed has COLLAPSE; every non-FLAG seed reaches L1_dmean >= 0.1 by 2k.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "theory", "results", "replay130")
FIT_SEEDS = list(range(3, 11)) + list(range(35, 43))                       # 4060
SAME_SEEDS = list(range(19, 27)) + list(range(59, 83))                      # 4060, disjoint held-out block (secondary)
CROSS_SEEDS = list(range(11, 19)) + list(range(27, 35)) + list(range(43, 59))   # 3070 (primary)
NEED = 8                                                                    # Literature 722b9f4c: >= 8 flagged and >= 8 non-flagged
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
    low = [r["step"] for i, r in enumerate(pr[1:], 1) if pr[i - 1]["step"] >= 600 and r["zb_std_in"] < 0.04 and pr[i - 1]["zb_std_in"] < 0.04]
    dmax = max([(r.get("L1_dmean") or 0) for r in pr if r["step"] <= 2000] or [0])
    rec = next((r["step"] for r in pr if low and r["step"] > low[0] and r["zb_std_in"] >= 0.05), None)   # 130-1: recovery after collapse
    return {"flag": flag, "basin": bool(flag and pull is None), "pull": pull, "collapse": bool(low), "collapse_step": low[0] if low else None, "recovery_step": rec, "dmean_max": dmax,
            "loss_1k": g(1000), "loss_2k": g(2000), "probes": [(r["step"], r["zb_std_in"]) for r in pr]}


def marker(f, theta, lo, hi, family="dip"):
    """dip (registered): zb_std_in < theta at two consecutive probes inside [lo, hi].
    persist (130-1 secondary, Audit): zb_std_in < theta at EVERY probe in [lo, hi]."""
    pr = [(s, z) for s, z in f["probes"] if lo <= s <= hi]
    if family == "persist":
        return bool(pr) and all(z < theta for _, z in pr)
    return any(pr[i][1] < theta and pr[i - 1][1] < theta for i in range(1, len(pr)))


PERSIST_WINDOWS = ((700, 1000), (700, 1200), (700, 1500))


def prf(fs, theta, lo, hi, label="flag", family="dip"):
    tp = sum(1 for f in fs if f[label] and marker(f, theta, lo, hi, family)); fp = sum(1 for f in fs if not f[label] and marker(f, theta, lo, hi, family))
    fn = sum(1 for f in fs if f[label] and not marker(f, theta, lo, hi, family))
    prec = tp / (tp + fp) if tp + fp else None; rec = tp / (tp + fn) if tp + fn else None
    return prec, rec


def cp(x, n, alpha=0.05):
    """Exact Clopper-Pearson interval for x / n, by bisection on the binomial tail (no scipy)."""
    from math import comb
    if n == 0:
        return (None, None)
    def tail_ge(p):
        return sum(comb(n, k) * p ** k * (1 - p) ** (n - k) for k in range(x, n + 1))
    def tail_le(p):
        return sum(comb(n, k) * p ** k * (1 - p) ** (n - k) for k in range(0, x + 1))
    def bis(f, target, increasing):
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = (lo + hi) / 2
            if (f(mid) < target) == increasing:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2
    lo = 0.0 if x == 0 else bis(tail_ge, alpha / 2, True)
    hi = 1.0 if x == n else bis(tail_le, alpha / 2, False)
    return (lo, hi)


def counts(fs, theta, lo, hi, label="flag", family="dip"):
    tp = sum(1 for f in fs if f[label] and marker(f, theta, lo, hi, family)); fp = sum(1 for f in fs if not f[label] and marker(f, theta, lo, hi, family))
    fn = sum(1 for f in fs if f[label] and not marker(f, theta, lo, hi, family))
    return tp, fp, fn


def collect(card, seeds):
    return {s: features(r) for s in seeds for r in [load(card, s)] if r is not None}


def h1277(fs):
    return {"flag_seeds_with_collapse": "%d / %d" % (sum(f["collapse"] for f in fs if f["flag"]), sum(f["flag"] for f in fs)),
            "nonflag_seeds_with_pull_0.1": "%d / %d" % (sum(f["dmean_max"] >= 0.1 for f in fs if not f["flag"]), sum(not f["flag"] for f in fs))}


def evaluate(card, seeds, th, lo, hi, label, truth="flag", family="dip"):
    T = collect(card, seeds); fs = list(T.values())
    nf, nn = sum(f[truth] for f in fs), sum(not f[truth] for f in fs)
    tp, fp, fn = counts(fs, th, lo, hi, truth, family)
    p = tp / (tp + fp) if tp + fp else None; r = tp / (tp + fn) if tp + fn else None
    ok = nf >= NEED and nn >= NEED
    verdict = (bool(p is not None and r is not None and p >= 0.9 and r >= 0.9) if ok else
               "incomplete (%d flagged, %d non-flagged; need >= %d each)" % (nf, nn, NEED))
    out = {"label": label, "card": card, "n": len(fs), "flagged": nf, "nonflagged": nn, "precision": p, "recall": r,
           "precision_ci95": cp(tp, tp + fp), "recall_ci95": cp(tp, tp + fn), "PASS": verdict, "h1277": h1277(fs),
           "escape_rate": nn / len(fs) if fs else None,
           "truth": truth, "seeds": {s: {"flag": f["flag"], "basin": f["basin"], "marker": marker(f, th, lo, hi, family), "pull": f["pull"], "collapse_step": f["collapse_step"]} for s, f in T.items()}}
    print(label, "card", card, "| n", len(fs), "flagged", nf, "| P %s %s R %s %s | PASS %s" % (p, out["precision_ci95"], r, out["recall_ci95"], verdict))
    print("  H-127-7", out["h1277"], "| escape rate", out["escape_rate"])
    return out


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "fit"
    if mode == "fit":
        F = collect("4060", FIT_SEEDS); fs = list(F.values())
        print("FIT seeds (flag, basin, pull, collapse, recovery, dmean_max)",
              {s: (f["flag"], f["basin"], f["pull"], f["collapse_step"], f["recovery_step"], round(f["dmean_max"], 3)) for s, f in F.items()})
        nf, nn = sum(f["flag"] for f in fs), sum(not f["flag"] for f in fs)
        if nf < 3 or nn < 3:
            print("FIT incomplete: %d flagged, %d non-flagged (need >= 3 each; extend with seeds 35..42)" % (nf, nn)); return
        out = {"mode": "fit", "seeds": {s: {k: v for k, v in f.items() if k != "probes"} for s, f in F.items()}, "h1277": h1277(fs)}
        for fam, lab in (("dip", "flag"), ("dip", "basin"), ("persist", "flag"), ("persist", "basin")):
            key = "%s|%s" % (fam, lab); wins = WINDOWS if fam == "dip" else PERSIST_WINDOWS
            if sum(f[lab] for f in fs) == 0 or all(f[lab] for f in fs):
                print("FIT [%s]: label has a single class on FIT; no cell" % key); out[key] = None; continue
            grid = {}
            for (lo, hi) in wins:
                for th in THETAS:
                    p, r = prf(fs, th, lo, hi, lab, fam); grid[(th, lo, hi)] = min(p or 0, r or 0)
            best = min(grid, key=lambda k: (-grid[k], k[2], k[0], k[1]))   # ties: earlier window end, smaller theta, then earlier window start (130-1)
            th, lo, hi = best; p, r = prf(fs, th, lo, hi, lab, fam)
            i = THETAS.index(th)
            nb = {"theta %s" % THETAS[j]: grid[(THETAS[j], lo, hi)] for j in (i - 1, i + 1) if 0 <= j < len(THETAS)}
            nb.update({"window %d-%d" % w: grid[(th, w[0], w[1])] for w in wins if w != (lo, hi)})
            print("FIT [%s] choice: theta %s window %d-%d | fit precision %s recall %s" % (key, th, lo, hi, p, r))
            print("FIT [%s] neighbours (min(P, R)):" % key, nb)
            out[key] = {"family": fam, "label": lab, "theta": th, "window": [lo, hi], "precision": p, "recall": r, "neighbours": nb,
                        "grid": {"%s|%d-%d" % k: v for k, v in grid.items()}}
        print("H-127-7 (FIT)", h1277(fs))
        json.dump(out, open(os.path.join(ROOT, "theory", "results", "score_s130_fit.json"), "w", newline=chr(10)), indent=1)
    else:
        fz = json.load(open(os.path.join(ROOT, "theory", "results", "score_s130_frozen.json")))   # written in the 130-1 freeze commit
        res = {"frozen": fz}
        for name, c in fz["cells"].items():
            th, (lo, hi), fam, lab = c["theta"], c["window"], c["family"], c["label"]
            res[name] = {"CROSS (3070)": evaluate("3070", CROSS_SEEDS, th, lo, hi, name + " CROSS", lab, fam),
                         "SAME (4060)": evaluate("4060", SAME_SEEDS, th, lo, hi, name + " SAME", lab, fam),
                         "role": c["role"]}
        print("PRIMARY (registered):", fz["primary"], "->", res[fz["primary"]]["CROSS (3070)"]["PASS"])
        json.dump(res, open(os.path.join(ROOT, "theory", "results", "score_s130_test.json"), "w", newline=chr(10)), indent=1)


if __name__ == "__main__":
    main()
