# Replicates theory/basin_entry.py's fixed-bin entry (read-only on explicit log.jsonl paths).
import json, os, sys
sys.stdout.reconfigure(encoding="utf-8")
R = "D:/Latent/boxes/runs_fsq"
runs = sys.argv[1:]
def losses(d):
    out = {}
    for line in open(os.path.join(R, d, "log.jsonl"), encoding="utf-8"):
        try: r = json.loads(line)
        except Exception: continue
        if "step" in r and r.get("loss") is not None:
            out[int(r["step"])] = float(r["loss"])
    return out
def bins(l):
    b = {}
    for s, v in l.items(): b.setdefault((s - 1) // 1000, []).append(v)
    return {k: sum(v) / len(v) for k, v in sorted(b.items())}
def entry(bm, thr):
    ks = sorted(bm)
    for i, k in enumerate(ks):
        if all(bm[j] < thr for j in ks[i:]): return 1000 * (k + 1)
    return None
def first_below(bm, thr):
    for k in sorted(bm):
        if bm[k] < thr: return 1000 * (k + 1)
    return None
print("run | last_step | entry0.5 | entry0.1 | first_bin<0.5 | bin8k | bin12k | bin16k | last_bin")
for d in runs:
    l = losses(d); bm = bins(l)
    g = lambda k: round(bm.get(k, float('nan')), 3)
    print(d, "|", max(l), "|", entry(bm, 0.5), "|", entry(bm, 0.1), "|", first_below(bm, 0.5), "|", g(7), "|", g(11), "|", g(15), "|", round(bm[max(bm)], 4))
