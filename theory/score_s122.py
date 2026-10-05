"""THEORY §122 scoring (rules fixed in §122; written before any §122 run; Theory is the only scorer).
Runs: boxes/runs_mult/a122_textmixanchor_4x4_s{1..3}/eval.json (primary) and a122_textmix_4x4_s{1,2}/eval.json (control),
keys as §121 (test_acc, per_slot_decode_acc, donor) plus T1 / text-path accuracy curves; MECH from
theory/results/mult_donor_diag.json keyed by run name.
CARRY-3 / ACC: >= 2/3 in the primary arm. C1 donor SERIAL and MECH (SAMEPREFIX sum_donor >= 0.80, SAMESUM recipient
>= 0.80) in passing seeds. Attribution to the anchor: primary passes AND the control fails CARRY-3 (op 3 < 0.95 in both).
STOP RULE: primary fails CARRY-3 -> the multiplication latent-carriage line pauses.
    python theory/score_s122.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "boxes", "runs_mult")
ARMS = {"TEXTMIX+ANCHOR+READ": ("a122_textmixanchor_4x4", [1, 2, 3]), "TEXTMIX+READ": ("a122_textmix_4x4", [1, 2])}


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
        passing = [s for s in carry if s in acc]; mech = {}
        for s in passing:
            m = M.get("%s_s%d" % (pre, s))
            mech[s] = None if m is None else (m["SAMEPREFIX"]["sum_donor"] >= 0.80 and m["SAMESUM"]["recipient"] >= 0.80)
        A = {"n": len(runs), "op3": op3, "test_acc": {s: r.get("test_acc") for s, r in runs.items()}, "CARRY3_seeds": carry, "ACC_seeds": acc,
             "C1_serial": {s: serial(runs[s].get("donor")) for s in passing}, "MECH": mech}
        if arm == "TEXTMIX+ANCHOR+READ":
            A["CARRY3"] = len(carry) >= 2; A["ACC"] = len(acc) >= 2
        res["arms"][arm] = A
        print("%-20s n %d | op3 %s | test %s | carry %s acc %s | C1 %s | MECH %s" % (arm, A["n"], op3, A["test_acc"], carry, acc, A["C1_serial"], mech))
    P, C = res["arms"]["TEXTMIX+ANCHOR+READ"], res["arms"]["TEXTMIX+READ"]
    res["ATTRIBUTED_TO_ANCHOR"] = bool(P.get("CARRY3")) and C["n"] == 2 and not C["CARRY3_seeds"]
    res["STOP_RULE_FIRES"] = P["n"] == 3 and not P.get("CARRY3")
    print("attributed to anchor:", res["ATTRIBUTED_TO_ANCHOR"], "| STOP RULE fires:", res["STOP_RULE_FIRES"])
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "score_s122.json"), "w", newline="\n"), indent=1, default=float)


if __name__ == "__main__":
    main()
