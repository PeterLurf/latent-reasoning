"""Basin entry (THEORY §187 s2 note, definition pinned before any §188 / §190 output; Audit's fixed-bin form).
Training loss from <run>/log.jsonl, deduped by step (the last occurrence wins: resumed runs log some steps twice).
Bins b = 0, 1, ...: steps 1000b + 1 .. 1000(b + 1). Bin mean = the mean "loss" over logged steps in the bin.
entry(thr) = the end step 1000(b + 1) of the first bin b such that this and every later non-empty bin has mean < thr; None if no such bin.
CONFOUNDED-BY-ENTRY(cell, comp) = |entry(0.5) difference| > 5000, or either run has entry(0.1) None (its loss never stays < 0.1).
    python theory/basin_entry.py <run dir name> [...]
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "boxes", "runs_fsq")


def losses(d):
    out = {}
    for line in open(os.path.join(R, d, "log.jsonl")):
        r = json.loads(line)
        if "step" in r and "loss" in r and r["loss"] is not None:
            out[int(r["step"])] = float(r["loss"])
    return out


def bins(l):
    b = {}
    for s, v in l.items():
        b.setdefault((s - 1) // 1000, []).append(v)
    return {k: sum(v) / len(v) for k, v in sorted(b.items())}


def entry(d, thr):
    bm = bins(losses(d)); ks = sorted(bm)
    for i, k in enumerate(ks):
        if all(bm[j] < thr for j in ks[i:]):
            return 1000 * (k + 1)
    return None


def summary(d):
    bm = bins(losses(d))
    return {"entry_0.5": entry(d, 0.5), "entry_0.1": entry(d, 0.1), "min_bin": min(bm.values()), "last_bin": bm[max(bm)]}


def confounded(cell, comp):
    a, b = summary(cell), summary(comp)
    flag = (a["entry_0.5"] is None or b["entry_0.5"] is None or abs(a["entry_0.5"] - b["entry_0.5"]) > 5000
            or a["entry_0.1"] is None or b["entry_0.1"] is None)
    return {"cell": a, "comp": b, "confounded_by_entry": bool(flag)}


if __name__ == "__main__":
    for d in sys.argv[1:]:
        print(d, summary(d))
