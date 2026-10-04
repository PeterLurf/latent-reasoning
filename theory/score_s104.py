"""THEORY §104 scoring (rules fixed in §104's registration; written after the run printed §92's partial patch values,
disclosed; Theory is the only scorer). Input: theory/results/s104_serial.json.
USE(j): single-patch donor-consistent >= 0.90.  BYPASS(j): single-patch recipient-consistent >= 0.10.
SERIAL (model): USE(j) for j = 1, 2, 3 and no BYPASS(j).  Family SERIAL: >= 4/5 (§92) or >= 2/3 (§97 GEN-4; §97 HYB-4).
Joint-only USE (descriptive): single fails but joint donor-consistent >= 0.90.
    python theory/score_s104.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    d = json.load(open(os.path.join(ROOT, "theory", "results", "s104_serial.json")))
    fam = {"l92": [k for k in d if k.startswith("l92_")], "gen4": [k for k in d if k.startswith("g97_gen4")],
           "hyb4": [k for k in d if k.startswith("g97_hyb4")]}
    res = {"models": {}, "families": {}}
    for k, r in d.items():
        p = r["patch"]; row = {}
        for j in (1, 2, 3):
            s, jt = p["slot%d_single" % j], p["slot%d_joint" % j]
            row[j] = {"use": s["donor_consistent"] >= 0.90, "bypass": s["recipient_consistent"] >= 0.10,
                      "joint_use": jt["donor_consistent"] >= 0.90, "single": s, "joint": jt}
        serial = all(row[j]["use"] and not row[j]["bypass"] for j in (1, 2, 3))
        joint_serial = all(row[j]["joint_use"] for j in (1, 2, 3))
        res["models"][k] = {"slots": row, "SERIAL": serial, "joint_all_slots": joint_serial}
        print("%-14s SERIAL %-5s joint-all %-5s | single donor %s | single recip %s | joint donor %s" % (
            k, serial, joint_serial, [round(row[j]["single"]["donor_consistent"], 3) for j in (1, 2, 3)],
            [round(row[j]["single"]["recipient_consistent"], 3) for j in (1, 2, 3)],
            [round(row[j]["joint"]["donor_consistent"], 3) for j in (1, 2, 3)]))
    for f, ks in fam.items():
        n_ser = sum(res["models"][k]["SERIAL"] for k in ks); need = 4 if f == "l92" else 2
        res["families"][f] = {"n": len(ks), "serial": n_ser, "SERIAL": len(ks) > 0 and n_ser >= need,
                              "joint_all": sum(res["models"][k]["joint_all_slots"] for k in ks)}
    print(res["families"])
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "score_s104.json"), "w", newline="\n"), indent=1, default=float)


if __name__ == "__main__":
    main()
