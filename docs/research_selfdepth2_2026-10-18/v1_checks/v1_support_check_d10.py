"""SD2_v1 check 1b ($0, exact rationals): the D10 law run through PROOFS_SELFDEPTH's own support_check (research5/scripts/support_check_r1copy.py,
identical to PROOFS' script), not only through PROOFS_SD2's rewrite. The script's definitions (H, plain_law, mixture) are executed from its source text
up to its case table (its module-level case loop writes a JSON to argv[1], so it is not imported); the case table is replaced by the D10 law and the
original 50/50 round-2 case. Output v1_support_check_d10.json; compared with v1_support_d10.json (must agree to 1e-12)."""
import json
import os
from fractions import Fraction as Fr

HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(os.path.dirname(HERE), "scripts", "support_check_r1copy.py")).read()
head = src.split("U = lambda")[0]
ns = {}
exec(compile(head, "support_check_r1copy.py(head)", "exec"), ns)   # defines H, plain_law, mixture (no I/O)
D10 = {**{n: Fr(1, 12) for n in range(9, 13)}, **{n: Fr(1, 6) for n in range(13, 17)}}
F50 = {n: Fr(1, 8) for n in range(9, 17)}
out = {"D10": ns["mixture"](Fr(1, 2), D10, ns["plain_law"]("excl"), kmax_self=20),
       "5050": ns["mixture"](Fr(1, 2), F50, ns["plain_law"]("excl"), kmax_self=20)}
mine = json.load(open(os.path.join(HERE, "v1_support_d10.json")))
cmp_ = {}
for tag, key in (("D10", "S1_D10"), ("5050", "S1_5050_regression")):
    d = []
    for t, v in out[tag].items():
        m = mine[key][str(t)]
        if v["P"] is not None:
            d.append(abs(v["P"] - m["P"])); d.append(abs(v["E_unf_lossw"] - m["deadline_cost_lossw"]))
    cmp_[tag] = max(d)
assert max(cmp_.values()) < 1e-12, cmp_
out["max_abs_diff_vs_v1_support_d10"] = cmp_
json.dump(out, open(os.path.join(HERE, "v1_support_check_d10.json"), "w", newline="\n"), indent=1, default=str)
print({t: round(v["P"], 4) for t, v in out["D10"].items() if v["P"] is not None}, cmp_)
