"""GSM8K length split prep (user decision via Audit, 2026-10-02): the per-problem step count of GSM8K-Aug
(Internalize_CoT_Step_by_Step @ e06a32ee, the Coconut repo's data/gsm_{train,valid,test}.json).
Step count n = number of non-empty reference steps (each one equation; empty strings dropped; non-empty steps that are
not a clean "<<expr=value>>", e.g. a stray "<<<", are still counted and their number is reported). Writes gsmlen/step_counts.json {split: [n per item, file order]}
and prints the distribution.
    python gsmlen/steps.py [--data D:/ext/coconut/data]
"""
import argparse
import collections
import hashlib
import json
import re

EQ = re.compile(r"<<[^<>]*=[^<>]*>>")


def nsteps(item):
    return sum(1 for x in item["steps"] if x.strip())


def malformed(item):
    return sum(1 for x in item["steps"] if x.strip() and not EQ.fullmatch(x.strip()))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="D:/ext/coconut/data"); a = ap.parse_args()
    out, dist = {}, {}
    for sp in ("train", "valid", "test"):
        p = f"{a.data}/gsm_{sp}.json"
        d = json.load(open(p, encoding="utf-8")); n = [nsteps(x) for x in d]; out[sp] = n
        c = collections.Counter(n)
        dist[sp] = {"n": len(n), "sha256_8": hashlib.sha256(open(p, "rb").read()).hexdigest()[:8],
                    "mean": round(sum(n) / len(n), 3), "empty_steps_dropped": sum(1 for x in d for s in x["steps"] if not s.strip()),
                    "malformed_nonempty_steps": sum(malformed(x) for x in d),
                    "counts": {str(k): c[k] for k in sorted(c)}}
    json.dump(out, open("gsmlen/step_counts.json", "w", newline="\n"))
    json.dump(dist, open("gsmlen/step_dist.json", "w", newline="\n"), indent=1)
    for sp, r in dist.items():
        cum, tot = 0, r["n"]; row = []
        for k, v in r["counts"].items():
            cum += v; row.append(f"{k}:{v} ({100 * cum / tot:.1f}%)")
        print(sp, tot, r["sha256_8"], "mean", r["mean"], "empty dropped", r["empty_steps_dropped"], "malformed", r["malformed_nonempty_steps"]); print("   ", "  ".join(row))


if __name__ == "__main__":
    main()
