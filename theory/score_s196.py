"""THEORY §196 scoring (GROKFAST-S3; rules fixed in §196; written before any §196 run; Theory is the only scorer).
s196_rmsgf_s3 = s185_rms_s3's args + --grokfast ema (alpha 0.98, lambda 2.0); exact control = s185_rms_s3 (same seed, same GPU class).
Pinned basin entry (theory/basin_entry.py): ACCELERATES = entry(0.5) <= 25000 and final 1k-bin loss < 0.1; NO-EFFECT = entry(0.5) None or > 40000; else partial.
Descriptive: the control's loss slope over its last 6k (per 1k), DEEP / DEPTH / STABLE vs RMS s1 / s2, clip fraction.
    python theory/score_s196.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import basin_entry as BE  # noqa: E402
import score_s185 as S185  # noqa: E402
import score_s187 as S187  # noqa: E402

R = os.path.join(ROOT, "boxes", "runs_fsq")
CELL, CTRL, EPS = "s196_rmsgf_s3", "s185_rms_s3", 1e-9


def slope_last6k(d):
    bm = BE.bins(BE.losses(d)); ks = sorted(bm)[-6:]
    return (bm[ks[-1]] - bm[ks[0]]) / max(1, len(ks) - 1)


def main():
    res = {}; want = dict(S185.ARMS["rms"], seed=3, grokfast="ema")
    a = json.load(open(os.path.join(R, CELL, "args.json"))) if os.path.exists(os.path.join(R, CELL, "args.json")) else {}
    if not os.path.exists(os.path.join(R, CELL, "log.jsonl")) or not os.path.exists(os.path.join(R, CELL, "eval.json")):
        rd = "pending (%s)" % ("running" if a else "open")
    else:
        bad = {k: a.get(k) for k, v in want.items() if str(a.get(k)) != str(v)}
        if bad:
            rd = "args error %s" % bad
        else:
            c, k = BE.summary(CELL), BE.summary(CTRL); e = c["entry_0.5"]
            lab = "ACCELERATES" if e is not None and e <= 25000 and c["last_bin"] < 0.1 - EPS else "NO-EFFECT" if e is None or e > 40000 else "partial"
            r = S185.read(CELL, want); dd = S187.read(CELL, want)
            res.update(cell_entry=c, ctrl_entry=k, ctrl_slope_last6k=slope_last6k(CTRL), quality=r, depth=dd.get("depth_9_12_n2"))
            rd = "%s: entry(0.5/0.1) %s/%s, final bin %.4f; control s3 entry %s, final %.4f, slope over last 6k %+.4f per 1k" % (
                lab, e, c["entry_0.1"], c["last_bin"], k["entry_0.5"], k["last_bin"], res["ctrl_slope_last6k"])
            if r.get("status") == "done":
                rd += "; DEEP %s, K'8 %.3f, K'16 %.3f, DEPTH %s (RMS s1 0.587 / s2 0.662)" % (r["deep"], r["pooled_k8"], r["pooled_k16"], None if dd.get("depth_9_12_n2") is None else round(dd["depth_9_12_n2"], 3))
    res["reading"] = rd; print("READING:", rd)
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "score_s196.json"), "w", newline=chr(10)), indent=1, default=str)


if __name__ == "__main__":
    main()
