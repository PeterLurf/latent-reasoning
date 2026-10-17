"""RT: self-pool consumption per n under the draft's round-1 mixture + self curriculum (stage A n in {9,10} -> stage B n in {9..12}).
Cases: stage A ends at global step a_end (early advance ~2k, cap 4k) or after 4k SELF steps. bs 64, 12k steps, p_self 0.5. 200 sims."""
import json, sys
import numpy as np
out = {}
for case in ("advance_2k_global", "cap_4k_global", "cap_4k_self_steps"):
    cons = []
    for s in range(200):
        r = np.random.default_rng(s); c = {n: 0 for n in range(9, 13)}; self_steps = 0
        for step in range(1, 12001):
            if r.random() >= 0.5:
                continue
            self_steps += 1
            stageA = (step <= 2000) if case == "advance_2k_global" else (step <= 4000) if case == "cap_4k_global" else (self_steps <= 4000)
            ns = r.integers(9, 11, 64) if stageA else r.integers(9, 13, 64)
            for n in ns:
                c[int(n)] += 1
        cons.append([c[n] for n in range(9, 13)])
    a = np.array(cons)
    out[case] = {"mean_per_n_9_12": a.mean(0).round(0).tolist(), "max_per_n_9_12": a.max(0).tolist(), "P_exceeds_120k_any_n": float((a.max(1) > 120000).mean())}
    print(case, out[case])
json.dump(out, open(sys.argv[1], "w", newline="\n"), indent=1)
