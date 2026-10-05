"""THEORY §118 scoring (rules fixed in §118; written before any §118 run; Theory is the only scorer).
Runs: boxes/runs_s5/s118_anchor_s{71..75}, s118_long_s{81..85}; control §111 READ boxes/runs_s5/s111_mk4_read_s{6..10}.
Each run dir: eval.json (cumulative, donor (single patch, k = 4 chunks), ...) and eval_s115.json (nosnap.cumulative
{128, 256, 512, 2048}, trigger, fixed_k4).
NOSNAP-512: nosnap cumulative >= 0.99 at 512 in >= 4/5 with single-slot donor SERIAL. EXIST: >= 1 seed.
NOSNAP-2048: nosnap cumulative >= 0.99 at 2,048 in >= 4/5.  Guard: snapped cumulative >= 0.99 at 512 in passing seeds.
Extension: NOSNAP-512 (per seed, with SERIAL) in >= 2/3 of the first three seeds.
    python theory/score_s118.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "boxes", "runs_s5")
ARMS = {"ANCHOR": ("s118_anchor", list(range(71, 76))), "LONG": ("s118_long", list(range(81, 86))),
        "CONTROL (§111 READ)": ("s111_mk4_read", list(range(6, 11)))}


def serial(r):
    d = r.get("donor") or {}
    return bool(d) and all(v["donor_consistent"] >= 0.90 and v["recipient_consistent"] < 0.10 for v in d.values())


def main():
    res = {"arms": {}}
    for arm, (pre, seeds) in ARMS.items():
        runs = {}
        for s in seeds:
            d = os.path.join(R, "%s_s%d" % (pre, s))
            if os.path.exists(os.path.join(d, "eval.json")) and os.path.exists(os.path.join(d, "eval_s115.json")):
                runs[s] = (json.load(open(os.path.join(d, "eval.json"))), json.load(open(os.path.join(d, "eval_s115.json"))))
        ns = {s: e115.get("nosnap", {}).get("cumulative", {}) for s, (e, e115) in runs.items()}
        p512 = [s for s, (e, _) in runs.items() if (ns[s].get("512") or 0) >= 0.99 and serial(e)]
        p2048 = [s for s in runs if (ns[s].get("2048") or 0) >= 0.99]
        guard = {s: runs[s][0]["cumulative"]["512"] >= 0.99 for s in p512}
        first3 = [s for s in seeds[:3] if s in p512]
        A = {"n": len(runs), "nosnap": ns, "NOSNAP512_seeds": p512, "NOSNAP512": len(p512) >= 4, "EXIST": len(p512) >= 1,
             "NOSNAP2048_seeds": p2048, "NOSNAP2048": len(p2048) >= 4, "guard": guard,
             "extend": len(first3) >= 2 if all(s in runs for s in seeds[:3]) else None,
             "snapped512": {s: e["cumulative"]["512"] for s, (e, _) in runs.items()},
             "trigger_snaps": {s: (e115.get("trigger") or {}).get("snaps_per_problem") for s, (_, e115) in runs.items()}}
        res["arms"][arm] = A
        print("%-20s n %d NOSNAP512 %s %s EXIST %s | NOSNAP2048 %s %s | guard %s | extend %s | nosnap %s" % (
            arm, A["n"], A["NOSNAP512"], p512, A["EXIST"], A["NOSNAP2048"], p2048, guard, A["extend"], ns))
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "score_s118.json"), "w", newline="\n"), indent=1, default=float)


if __name__ == "__main__":
    main()
