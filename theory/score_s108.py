"""THEORY §108 scoring (rules fixed in §108; written before any §108 implementation or run; Theory is the only scorer).
Runs: boxes/runs_s5/s108_mk{4,8}_s{1..5}/eval.json. Keys: cumulative {"512"}, extension_2048.cumulative {"2048"},
per_handoff_acc, removal.{zero,mean}.acc_after_first_ablated_slot (secondary), and the §104 donor protocol computed in
the eval: donor {"slot<j>": {donor_consistent, recipient_consistent}} for every in-chunk slot j = 1..k-1 (single patch:
the latent fed into segment j+1 <- the donor's).
ACC: cumulative >= 0.99 at 512 in >= 4/5 seeds.  SERIAL (model): every in-chunk slot donor-consistent >= 0.90 and
recipient-consistent < 0.10.  C1-MK: ACC and SERIAL in >= 4/5 seeds.  C6: cumulative >= 0.99 at 2,048 in >= 4/5.
    python theory/score_s108.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    res = {"arms": {}}
    for k in (4, 8):
        runs = {}
        for s in range(1, 6):
            f = os.path.join(ROOT, "boxes", "runs_s5", "s108_mk%d_s%d" % (k, s), "eval.json")
            if os.path.exists(f):
                runs[s] = json.load(open(f))
        acc = [s for s, r in runs.items() if r["cumulative"]["512"] >= 0.99]
        ser = [s for s, r in runs.items() if r.get("donor") and all(v["donor_consistent"] >= 0.90 and v["recipient_consistent"] < 0.10 for v in r["donor"].values())]
        c6 = [s for s, r in runs.items() if (r.get("extension_2048", {}).get("cumulative", {}).get("2048") or 0) >= 0.99]
        A = {"n": len(runs), "acc_seeds": acc, "ACC": len(acc) >= 4, "serial_seeds": ser,
             "C1_MK": len(acc) >= 4 and len(set(acc) & set(ser)) >= 4, "C6": len(c6) >= 4,
             "cum512": {s: r["cumulative"]["512"] for s, r in runs.items()},
             "ext2048": {s: r.get("extension_2048", {}).get("cumulative", {}).get("2048") for s, r in runs.items()},
             "per_handoff": {s: r.get("per_handoff_acc") for s, r in runs.items()},
             "donor": {s: r.get("donor") for s, r in runs.items()}}
        res["arms"]["MK-%d" % k] = A
        print("MK-%d n %d ACC %s C1-MK %s C6 %s | cum512 %s | ext2048 %s | serial %s" % (k, A["n"], A["ACC"], A["C1_MK"], A["C6"],
              {s: round(v, 4) for s, v in A["cum512"].items()}, A["ext2048"], ser))
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "score_s108.json"), "w", newline="\n"), indent=1, default=float)


if __name__ == "__main__":
    main()
