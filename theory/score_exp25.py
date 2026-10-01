"""Score THEORY §51 (TH1, TH2) from theory/results/exp25_target_halt.json. Scored by Experiments (handover 2026-09-30).
TH1: under the last rule, median stop over reachable targets non-decreasing across distance bins 1-6, 7-12, 13-18, 19-24
     and strictly larger at 19-24 than at 1-6, in >= 7/10 models.
TH2: held-out accuracy (last rule + learned halt) >= (max rule + learned halt) - 0.02 in >= 7/10 models.
    python theory/score_exp25.py
"""
import json

R = json.load(open("theory/results/exp25_target_halt.json"))
BINS = ["1-6", "7-12", "13-18", "19-24"]
th1, th2, rows = 0, 0, {}
for m, v in R.items():
    s = [v["last"]["median_stop_by_dist_reach"][b] for b in BINS]
    t1 = all(s[i] <= s[i + 1] for i in range(3)) and s[3] > s[0]
    t2 = v["last"]["acc"] >= v["max"]["acc"] - 0.02
    th1 += t1; th2 += t2
    rows[m] = {"last_stops": s, "max_stops": [v["max"]["median_stop_by_dist_reach"][b] for b in BINS], "TH1": t1,
               "acc_last": v["last"]["acc"], "acc_max": v["max"]["acc"], "TH2": t2}
out = {"n": len(R), "TH1_count": th1, "TH1": "met" if th1 >= 7 else "not met", "TH2_count": th2,
       "TH2": "met" if th2 >= 7 else "not met", "models": rows}
json.dump(out, open("theory/results/score_exp25.json", "w", newline="\n"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "models"}))
for m, r in rows.items():
    print(m, r)
