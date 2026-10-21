"""REVIEW_R3 check 4 ($0, CPU): independent closed-form recompute of R3_draft §5 Stage A OC, plus two nulls the draft omits.
 (i) draft null: delta = 0.  (ii) GENERIC-FREEZE null: any freeze (or any slowing of self-item learning) raises STAB by delta_g with no
 localisation; delta_g in {0.02, 0.04, 0.06} [ASSUMED; FT-CTRL - FT-SELF = 0.207 is the scale of 'less self learning'].
 (iii) P(Stage B launch | null) = P(both) + P(s1 pass, s2 fail) (draft §6 launch rule: MECH-PARTIAL with s1's SEED-PASS also launches B).
 SEED-PASS = D in [0.08, base + 0.10 - twin] and ABSORB (p_abs 0.95 under nulls), D ~ N(delta, sqrt(2) s_rep). Writes rv_oc_recheck.json."""
import json, math, os
PHI = lambda z: 0.5 * (1 + math.erf(z / math.sqrt(2)))  # noqa: E731
BASE = {"s1": 0.782, "s2": 0.684}; TWIN = 0.489
def sp(delta, s, base, p_abs):
    sd = math.sqrt(2) * s
    return p_abs * (PHI((base + 0.10 - TWIN - delta) / sd) - PHI((0.08 - delta) / sd))
out = {"def": __doc__, "rows": {}}
for s in (0.02, 0.04, 0.06):
    for dg in (0.0, 0.02, 0.04, 0.06):
        p1, p2 = sp(dg, s, BASE["s1"], 0.95), sp(dg, s, BASE["s2"], 0.95)
        out["rows"]["s_rep %.2f | delta_g %.2f" % (s, dg)] = {"seed_s1": round(p1, 4), "both": round(p1 * p2, 4), "B_launch": round(p1 * p2 + p1 * (1 - p2), 4)}
    p1, p2 = sp(0.207, s, BASE["s1"], 0.85), sp(0.109, s, BASE["s2"], 0.85)
    out["rows"]["s_rep %.2f | full effect"] = None
    out["rows"]["s_rep %.2f | full effect" % s] = {"seed_s1": round(p1, 4), "seed_s2": round(p2, 4), "both": round(p1 * p2, 4)}
out["rows"] = {k: v for k, v in out["rows"].items() if v is not None}
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rv_oc_recheck.json"), "w", newline="\n"), indent=1)
for k, v in out["rows"].items():
    print("%-28s %s" % (k, v))
