"""THEORY §121 scoring (rules fixed in §121; written before any §121 run; Theory is the only scorer).
Runs: boxes/runs_mult/a121_anchorread_4x4_s{1..3}/eval.json, a121_read_4x4_s{1,2}/eval.json (keys as §99: test_acc,
per_slot_decode_acc, removal, donor {slot1..3}) and the MECH donor test from theory/results/mult_donor_diag.json keyed by
run name (SAMEPREFIX.sum_donor, SAMESUM.recipient).
CARRY-3: op-3 decode (per_slot_decode_acc[2]) >= 0.95 in >= 2/3 (ANCHOR+READ). ACC: test_acc >= 0.95 in >= 2/3.
C1: donor SERIAL (slots 1-3 donor >= 0.90, recipient < 0.10) in passing seeds. MECH: SAMEPREFIX sum_donor >= 0.80 and
SAMESUM recipient >= 0.80 in passing seeds. READ-only: descriptive against the same thresholds.
    python theory/score_s121.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "boxes", "runs_mult")
ARMS = {"ANCHOR+READ": ("a121_anchorread_4x4", [1, 2, 3]), "READ": ("a121_read_4x4", [1, 2])}


def serial(d):
    return bool(d) and all(v["donor_consistent"] >= 0.90 and v["recipient_consistent"] < 0.10 for v in d.values())


def main():
    mp = os.path.join(ROOT, "theory", "results", "mult_donor_diag.json"); M = json.load(open(mp)) if os.path.exists(mp) else {}
    res = {"arms": {}}
    for arm, (pre, seeds) in ARMS.items():
        runs = {}
        for s in seeds:
            p = os.path.join(R, "%s_s%d" % (pre, s), "eval.json")
            if os.path.exists(p):
                runs[s] = json.load(open(p))
        op3 = {s: r["per_slot_decode_acc"][2] for s, r in runs.items()}
        carry = [s for s in runs if op3[s] >= 0.95]; acc = [s for s, r in runs.items() if r.get("test_acc", 0) >= 0.95]
        passing = [s for s in carry if s in acc]
        mech = {}
        for s in passing:
            k = "%s_s%d" % (pre, s); m = M.get(k)
            mech[s] = None if m is None else (m["SAMEPREFIX"]["sum_donor"] >= 0.80 and m["SAMESUM"]["recipient"] >= 0.80)
        need = 2 if arm == "ANCHOR+READ" else None
        A = {"n": len(runs), "op3": op3, "test_acc": {s: r.get("test_acc") for s, r in runs.items()},
             "CARRY3_seeds": carry, "ACC_seeds": acc, "CARRY3": len(carry) >= 2 if need else None, "ACC": len(acc) >= 2 if need else None,
             "C1_serial": {s: serial(runs[s].get("donor")) for s in passing}, "MECH": mech}
        res["arms"][arm] = A
        print("%-12s n %d | op3 %s | test %s | CARRY-3 %s ACC %s | C1 %s | MECH %s" % (arm, A["n"], op3, A["test_acc"], A["CARRY3"], A["ACC"], A["C1_serial"], mech))
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "score_s121.json"), "w", newline="\n"), indent=1, default=float)


if __name__ == "__main__":
    main()
