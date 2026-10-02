"""Score THEORY §76(B) (ge3 teachers, primary) and amendment 76-2's B-std (std teachers, secondary) for one group.
Committed before any §76(B) student result existed.
Per seed 2051-2060 and teacher curriculum: the teacher file theory/results/c1_tab_local/<G>_<cur>_s<seed>.json (its
certified flag) and the students c1/runs/p76_<cur>_<student>_s<seed>/eval.json (pfstd, ptstd).
Rule (§76): a student succeeds at >= 9/10 seeds certified (certificate on the student's own hard table), counting a
seed whose teacher is uncertified (or missing) as a failure. Reported alongside: teacher-alone certified count, per seed
held-out-row accuracy (composition vs copying), the held-out-patched certificate, accuracy at N 8-128, codes reached
outside R. A verdict with fewer than 10 resolved seeds is "incomplete".
    python c1/score_76b.py [S4]
"""
import json
import os
import sys

SEEDS = range(2051, 2061); STUDENTS = ("pfstd", "ptstd")


def main():
    G = sys.argv[1] if len(sys.argv) > 1 else "S4"
    res = {}
    for cur in ("ge3", "std", "ge3full", "stdfull"):                   # *full: amendment 76-3, descriptive (no rule)
        r = {"teacher_certified": 0, "teacher_files": 0, "per_seed": {}}
        k = {s: 0 for s in STUDENTS}; resolved = 0
        for sd in SEEDS:
            tf = f"theory/results/c1_tab_local/{G}_{cur.replace('full', '')}_s{sd}.json"
            row = {}
            if os.path.exists(tf):
                t = next(iter(json.load(open(tf)).values())); r["teacher_files"] += 1
                row["teacher_certified"] = bool(t["certified"]); r["teacher_certified"] += int(bool(t["certified"]))
            else:
                row["teacher_certified"] = None
            done = row["teacher_certified"] is False
            for s in STUDENTS:
                f = f"c1/runs/p76_{cur}_{s}_s{sd}/eval.json"
                if os.path.exists(f):
                    e = json.load(open(f)); c = bool(e["certificate"]["certified"]) and bool(e["evidential"])
                    k[s] += int(c)
                    row[s] = {"certified": c, "heldout_row_acc": e["heldout_row_acc"] if e["heldout_row_acc"] is not None else float("nan"), "train_row_acc": e["train_row_acc"],
                              "patched_certified": e["certificate_heldout_patched"]["certified"], "acc": e["acc"],
                              "outside_R": e["outside_R"], "bad_pairs": e["certificate"]["bad_pairs"]}
            done = done or all(s in row for s in STUDENTS)
            resolved += int(done)
            r["per_seed"][sd] = row
        for s in STUDENTS:
            v = "succeeds" if k[s] >= 9 else ("fails" if k[s] + (10 - resolved) < 9 else "open")
            r[s] = {"certified": k[s], "of": 10, "resolved_seeds": resolved,
                    "verdict": v if resolved == 10 or v == "fails" else f"incomplete ({v})"}
        r["descriptive_only"] = cur.endswith("full"); res[cur] = r
    os.makedirs("theory/results", exist_ok=True)
    json.dump(res, open(f"theory/results/score_s76b_{G}.json", "w", newline="\n"), indent=1)
    for cur, r in res.items():
        print(f"{G} {cur}: teachers certified {r['teacher_certified']}/{r['teacher_files']} files;",
              " ".join(f"{s} {r[s]['certified']}/10 -> {r[s]['verdict']}" for s in STUDENTS))
        for sd, row in r["per_seed"].items():
            if any(s in row for s in STUDENTS):
                print("  ", sd, "teacher", row["teacher_certified"],
                      " ".join(f"{s}: cert {row[s]['certified']} heldout {row[s]['heldout_row_acc']:.3f} acc128 {row[s]['acc']['128']:.3f}"
                               for s in STUDENTS if s in row))


if __name__ == "__main__":
    main()
