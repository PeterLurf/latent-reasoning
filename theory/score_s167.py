"""THEORY §167 scoring (rules fixed in §167; written before any §167 implementation or run; Theory is the only scorer).
DAS-EDIT: distributed alignment search (2303.02536) on the flattened memory of existing models; post hoc, no LM training. A learned orthonormal k-subspace
R (k in {16, 64}); intervention S_rec <- S_rec + R R^T (S_don - S_rec); IIA = exact counterfactual answer on 1,000 held-out pairs.
  MULT (§161 MIX s1-s3): variable = the in-progress column accumulator after pair t in {4, 8, 12}; counterfactual = the recipient's later digits from the
  donor's accumulator (donor_delta's expected digits).  S5 (§162 SHORT / LONG models): variable = the current state g after item p* = l_max / 2.
Controls (fixed now; the subspace-patching illusion, 2311.17030): RANDOM-SUBSPACE (an unfitted random k-subspace) IIA <= 0.2 and SHUFFLED-TARGETS (DAS fit
to a mismatched donor's counterfactuals, held-out IIA against those targets) <= 0.2.
DECOMPOSABLE(model, t): fitted IIA >= 0.8 at some k <= 64 AND IIA - unchanged_target_share >= 0.5 (Audit: the null rate) AND both controls <= 0.2 AND
necessity_drop >= 0.3 (Literature: mean-ablating the fitted subspace in unpatched runs lowers accuracy) AND random-model IIA <= 0.2 at k 16 / 64 (note 167-1). k in {1, 2, 4, 8, 16, 64} (+ 2048 reference). NOT-DECOMPOSABLE: fitted IIA < 0.5 at k = 64 with the controls valid.
ILLUSION-RISK: shuffled-target IIA > 0.2 (the method fits arbitrary targets here; no reading). FIT-FAILURE (note 167-3): a would-be NOT-DECOMPOSABLE point
whose train IIA is non-monotone in k (some fitted k more than 0.05 below a smaller k's: the fit did not converge) - uninformative. MULT arm: >= 2 / 3 MIX models DECOMPOSABLE at the same t.
Output per model: <run>/eval_das.json {"family", "points": {t: {"k": {k: {"iia", "iia_train", "n"}}, "unchanged_target_share", "random_subspace": {k: iia}, "shuffled_targets": {k: iia},
"subspace_var_share": {k: share}}}}.
    python theory/score_s167.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "boxes", "runs_fsq")
MULT = ["s161_mix_4x4_s%d" % s for s in (1, 2, 3)]
S5 = ["s162_short_s1", "s162_short_s2", "s162_long_s2"]


def load(d, f="eval_das.json"):
    p = os.path.join(R, d, f)
    return json.load(open(p)) if os.path.exists(p) else None


def point_kind(pt):
    ks = pt.get("k") or {}; rnd = pt.get("random_subspace") or {}; shf = pt.get("shuffled_targets") or {}
    if any((v or 0) > 0.2 for v in shf.values()):
        return "ILLUSION-RISK", None
    best = max(((int(k), (v or {}).get("iia") or 0) for k, v in ks.items() if int(k) <= 64), default=(None, 0))
    rnd_ok = all((v or 0) <= 0.2 for v in rnd.values()) and bool(rnd)
    un = pt.get("unchanged_target_share")
    nd = (pt.get("necessity_drop") or {}).get(str(best[0])) if best[0] is not None else None
    rm = pt.get("random_model") or {}; rm_ok = bool(rm) and all((v or 0) <= 0.2 for v in rm.values())   # note 167-1: random-init model baseline required
    if best[1] >= 0.8 and un is not None and best[1] - un >= 0.5 and rnd_ok and shf and nd is not None and nd >= 0.3 and rm_ok:
        return "DECOMPOSABLE", best
    i64 = (ks.get("64") or {}).get("iia")
    tr = sorted((int(k), v.get("iia_train")) for k, v in ks.items() if isinstance(v, dict) and v.get("iia_train") is not None)   # fitted k only (k = D has none)
    converged = bool(tr) and all(tr[j][1] >= max(x[1] for x in tr[:j]) - 0.05 for j in range(1, len(tr)))   # note 167-3: nested subspaces -> monotone in k
    if i64 is not None and i64 < 0.5 and rnd_ok and shf:
        return ("NOT-DECOMPOSABLE" if converged else "FIT-FAILURE"), best
    return "INTERMEDIATE", best


def model(d):
    r = load(d)
    if r is None:
        return {"status": "pending"}
    return {"status": "done", "points": {t: dict(zip(("kind", "best"), point_kind(pt)), iia={k: (v or {}).get("iia") for k, v in (pt.get("k") or {}).items()}, iia_train={k: (v or {}).get("iia_train") for k, v in (pt.get("k") or {}).items()},
                                              unchanged=pt.get("unchanged_target_share"), necessity=pt.get("necessity_drop"), random_model=pt.get("random_model"),
                                              random=pt.get("random_subspace"), shuffled=pt.get("shuffled_targets"), var_share=pt.get("subspace_var_share"))
                                         for t, pt in (r.get("points") or {}).items()}}


def main():
    res = {"mult": {d: model(d) for d in MULT}, "s5": {d: model(d) for d in S5}}
    done = [v for v in res["mult"].values() if v["status"] == "done"]
    if len(done) < 3:
        rd = "pending (MULT %d / 3 models)" % len(done)
    else:
        ts = set.intersection(*[set(v["points"]) for v in done]) if done else set()
        dec = sorted(t for t in ts if sum(1 for v in done if v["points"][t]["kind"] == "DECOMPOSABLE") >= 2)
        ill = sorted(t for t in ts if any(v["points"][t]["kind"] == "ILLUSION-RISK" for v in done))
        rd = ("DECOMPOSABLE (MULT, >= 2 / 3 models) at pairs %s: the in-progress sum is causally carried in a compact subspace; the C1 / C7 edit exists beyond the "
              "probe direction" % dec) if dec else "MULT not decomposable at k <= 64 (whole-state SERIAL only)"
        if ill:
            rd += "; ILLUSION-RISK at %s (no reading there)" % ill
    res["reading_mult"] = rd
    print("MULT:", {d: v.get("points") or v["status"] for d, v in res["mult"].items()})
    print("S5:", {d: v.get("points") or v["status"] for d, v in res["s5"].items()})
    print("READING (MULT):", rd)
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "score_s167.json"), "w", newline=chr(10)), indent=1, default=float)


if __name__ == "__main__":
    main()
