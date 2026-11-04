"""§270 FULL-PERM gates G2 RENDER + G3 DISJOINT (FP_v1 §7; CPU) -> scratch/gate_s272/g2_render.json, g3_disjoint.json.
G2 (20k scope-all re-renders of REG training-stream items, d 2-4, PR.train_perm_all at seed 1, steps 1.., k 0..): fields + OW.solve == answer (PR.check), the statement
  multiset equal, the question last, never the identity order; op relative order uniform (chi^2 over the 6 orders of S_1..S_3 at d 3, p > 0.01); s0_pos histogram (chi^2
  vs uniform reported); N1-N4 break rates (fp_parts.flags); tok_aw targets / cues equal to the unpermuted item (aux, ans, cue count, every cue's suffix after the think
  rows); the §257 inserts (IR.distract_mix, s256 / chain / givop) after the permutation still solve.
G3 (the gate-only instrumented trainer of g1_offbit, --perm_scope all, of_s1's flags without the inits, seed 1, 2000 steps of item draws): the in-trainer asserts (TP / INS /
  FP + SG texts, the multiset check) hold on every drawn item; multiset_rejects (resampled base items) read from the trainer's MSREJ at the end; seeds 27000 / 27001 / 27010
  re-grepped unused outside the §270 files.
    python -B scratch/s272_build/g2_g3.py [g2] [g3]"""
import json
import os
import re
import subprocess
import sys
from collections import Counter

NEW = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")).replace("\\", "/"); sys.path.insert(0, NEW); sys.path.insert(0, NEW + "/scratch"); os.chdir(NEW)
OUT = NEW + "/scratch/gate_s272"; os.makedirs(OUT, exist_ok=True)


def chi2_p(counts, k):
    """Pearson chi^2 vs uniform over k cells -> P(X >= x), X ~ chi^2(k - 1) (regularised upper gamma; series / continued fraction; no scipy here)."""
    import math
    obs = [counts.get(i, 0) for i in range(k)]; e = sum(obs) / k; x = sum((o - e) ** 2 / e for o in obs); a, z = (k - 1) / 2, x / 2
    if z <= 0:
        return 1.0
    if z < a + 1:                                                           # lower series P(a, z), then 1 - P
        t = s = 1 / a; n = a
        while abs(t) > 1e-15 * abs(s):
            n += 1; t *= z / n; s += t
        return float(1 - s * math.exp(-z + a * math.log(z) - math.lgamma(a)))
    b = z + 1 - a; c = 1e300; dd = 1 / b; h = dd                            # Lentz continued fraction for Q(a, z)
    for i in range(1, 500):
        an = -i * (i - a); b += 2; dd = an * dd + b; dd = 1 / (dd if abs(dd) > 1e-300 else 1e-300); c = b + an / c; c = c if abs(c) > 1e-300 else 1e-300
        h *= dd * c
        if abs(dd * c - 1) < 1e-15:
            break
    return float(math.exp(-z + a * math.log(z) - math.lgamma(a)) * h)


def g2():
    import numpy as np
    import boxes.fp_parts as FPP
    import boxes.inject_aw as AW
    import boxes.insert_render as IR
    import boxes.ow_words as OW
    import boxes.perm_render as PR
    from boxes.inject_ow import OWKW
    s = OW.Stream(OW.SEED_TRAIN, (), frames=OW.FRAME_TRAIN, mul_s_p=0, **OWKW); n = 0; ord3 = Counter(); s0 = {d: Counter() for d in (2, 3, 4)}; viol = Counter(); fl = Counter()
    ins_ok = 0; ins_n = 0; perms = [tuple(p) for p in __import__("itertools").permutations((1, 2, 3))]
    for k in range(20000):
        d = 2 + k % 3; it = s.draw(d); pit = PR.train_perm_all(it, 1, 1 + k // 64, k % 64)
        assert pit["order"] != it["order"] and sorted(pit["order"]) == sorted(it["order"]), "never identity; same statements"
        assert FPP.msig(pit) == FPP.msig(it) and pit["text"].endswith(" " + pit["question"]), "multiset / question last"
        t0, t1 = AW.tok_aw(it, "value", "step", "answer"), AW.tok_aw(pit, "value", "step", "answer")
        assert t0["aux"] == t1["aux"] and t0["ans"] == t1["ans"] and len(t0["think_cues"]) == len(t1["think_cues"]) and all(
            q1[len(t1["think"]):] == q0[len(t0["think"]):] for q1, q0 in zip(t1["think_cues"], t0["think_cues"])), "targets / cues"
        f = FPP.flags(pit["order"], d); s0[d][f["s0_pos"]] += 1
        for key in ("n3", "n4", "given_after_consumer"):
            fl[key] += f[key]
        viol[f["n_viol"]] += 1
        if d == 3:
            ord3[perms.index(tuple(int(x[1:]) for x in pit["order"] if x in ("S1", "S2", "S3")))] += 1
        if k % 4 == 0:
            r2, n_ = IR.distract_mix(pit, 1, 1 + k // 64, k % 64, ("s256", "chain", "givop"))
            if n_:
                ins_n += 1; ins_ok += OW.solve(r2["text"]) == it["answer"]
        n += 1
    s0p = {str(d): {"hist": dict(sorted(c.items())), "p_uniform": chi2_p(c, max(c) + 1)} for d, c in s0.items()}
    out = {"n": n, "op_order_d3": {"hist": {str(perms[i]): c for i, c in sorted(ord3.items())}, "chi2_p": chi2_p(ord3, 6)}, "s0_pos": s0p,
           "break_rates": {k_: v / n for k_, v in fl.items()}, "n_viol_hist": dict(sorted(viol.items())), "inserts_after_perm": {"n": ins_n, "solve_ok": ins_ok}}
    out["pass"] = out["op_order_d3"]["chi2_p"] > 0.01 and ins_ok == ins_n and ins_n > 0
    json.dump(out, open(OUT + "/g2_render.json", "w", newline="\n"), indent=1); print(json.dumps(out)); return out["pass"]


def g3():
    import g1_offbit as G1
    src = open(NEW + "/boxes/_inject_aw_g1_s272.py", encoding="utf-8").read() if os.path.exists(NEW + "/boxes/_inject_aw_g1_s272.py") else None
    G1.instrument(NEW); p = NEW + "/boxes/_inject_aw_g1_s272.py"; s = open(p, encoding="utf-8").read()
    s = s.replace('        _gj.dump(_G1, open(_go.environ["G1_OUT"], "w")); _gs.exit(0)\n',
                  '        _gj.dump({"n": len(_G1), "msrej": _gs._getframe(1).f_locals["MSREJ"][0]}, open(_go.environ["G1_OUT"], "w")); _gs.exit(0)\n', 1)
    open(p, "w", encoding="utf-8", newline="\n").write(s)
    tag, rc, h = G1.run(NEW, 1, ["--perm_scope", "all"], "g3_all_s1") if G1.N == 2000 else (None, None, None)
    res = json.load(open(G1.TMP + "/g3_all_s1/hashes.json")) if os.path.exists(G1.TMP + "/g3_all_s1/hashes.json") else {}
    hits = subprocess.run(["git", "grep", "-nE", r"\b270(00|01|10)\b", "--", "boxes/*.py", "theory/*.py", "scratch/*.py", "scratch/*/*.py"], capture_output=True, text=True).stdout
    own = ("boxes/perm_render.py", "boxes/fp_parts.py", "theory/score_fp.py", "scratch/s272_build/")
    foreign = [ln for ln in hits.splitlines() if not ln.startswith(own)]
    out = {"rc": rc, "steps": res.get("n"), "multiset_rejects": res.get("msrej"), "seed_grep_foreign_hits": foreign,
           "pass": rc == 0 and res.get("n") == 2000 and not foreign}
    json.dump(out, open(OUT + "/g3_disjoint.json", "w", newline="\n"), indent=1); print(json.dumps(out)); return out["pass"]


if __name__ == "__main__":
    which = sys.argv[1:] or ["g2", "g3"]; ok = True
    if "g3" in which:
        sys.argv = [sys.argv[0], "2000"]
    for w in which:
        ok &= {"g2": g2, "g3": g3}[w]()
    sys.exit(0 if ok else 1)
