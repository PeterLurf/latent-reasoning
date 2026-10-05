"""THEORY §119 scoring (rules fixed in §119; written before any §119 run; Theory is the only scorer).
Runs (chunked host): boxes/runs_s5/s119_{coconut,simcot,codi}_s{1..3}/eval.json with keys
  intact_acc (in-chunk final-state accuracy, random starts), donor {slot1..3: {donor_consistent, recipient_consistent}}
  (CODI: donor {block: {...}}), cumulative {512}, extension_2048, faithful (bool, from the method's faithfulness check).
Competence gate: intact_acc >= 0.90, else "did not learn". A method counts only if faithful and >= 2 competent seeds.
SERIAL: every slot donor >= 0.90 and recipient < 0.10. CONTROLLED-C1: neither Coconut nor SIM-CoT is SERIAL in
>= 2/3 of its competent seeds (each counted method), with §111 READ SERIAL 5/5 (111-4). CODI: block donor, descriptive.
    python theory/score_s119.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "boxes", "runs_s5")


def serial(d):
    return bool(d) and all(v["donor_consistent"] >= 0.90 and v["recipient_consistent"] < 0.10 for v in d.values())


def main():
    res = {"methods": {}}
    counted = []
    for meth in ("coconut", "simcot", "codi"):
        runs = {}
        for s in range(1, 4):
            p = os.path.join(R, "s119_%s_s%d" % (meth, s), "eval.json")
            if os.path.exists(p):
                runs[s] = json.load(open(p))
        comp = [s for s, r in runs.items() if (r.get("intact_acc") or 0) >= 0.90]
        faithful = all(r.get("faithful") for r in runs.values()) if runs else None
        M = {"n": len(runs), "competent": comp, "did_not_learn": [s for s in runs if s not in comp], "faithful": faithful,
             "intact_acc": {s: r.get("intact_acc") for s, r in runs.items()},
             "cum512": {s: (r.get("cumulative") or {}).get("512") for s, r in runs.items()},
             "donor": {s: r.get("donor") for s, r in runs.items()}}
        if meth != "codi":
            M["serial_seeds"] = [s for s in comp if serial(runs[s].get("donor"))]
            M["counts"] = bool(faithful) and len(comp) >= 2
            M["serial_majority"] = M["counts"] and len(M["serial_seeds"]) * 3 >= 2 * len(comp)
            if M["counts"]:
                counted.append(meth)
        res["methods"][meth] = M
        print("%-8s n %d faithful %s competent %s did-not-learn %s serial %s counts %s | intact %s | cum512 %s" % (
            meth, M["n"], faithful, comp, M["did_not_learn"], M.get("serial_seeds"), M.get("counts"), M["intact_acc"], M["cum512"]))
    res["counted"] = counted
    res["CONTROLLED_C1"] = (len(counted) >= 1 and not any(res["methods"][m]["serial_majority"] for m in counted)) if counted else None
    res["inconclusive"] = [m for m in ("coconut", "simcot") if not res["methods"][m].get("counts")]
    print("CONTROLLED-C1:", res["CONTROLLED_C1"], "| counted", counted, "| inconclusive", res["inconclusive"])
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "score_s119.json"), "w", newline="\n"), indent=1, default=float)


if __name__ == "__main__":
    main()
