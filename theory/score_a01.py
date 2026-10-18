"""THEORY note 191-4 (A0-1, $0, descriptive): ANCHOR-CUE-CONTRIBUTION of the kept §191 models; Theory scores, Audit reproduces.
Per model <run>/eval_a01.json (prosqa168s2 --a01, 2070 fp32, eval_bs 16): items [{i, depth, ext, pred, correct, margin}], meta {test_file, ...}.
  gate: |acc_by_depth(eval_a01) - committed eval.json acc_by_depth| <= 0.02 at every depth with >= 100 items, else UNREADABLE.
  strata (docs/research_lang_2026-10-18/scripts/anticue.py, unchanged logic): ext >= 2; heur_panel.anchors + panel_for(...)["proximity"]
    1.0 -> pro, 0.0 -> anti, 0.5 -> tie. Readings pooled over depth >= 5 (and per depth): acc(pro), acc(tie), acc(anti) with Wilson 95 % CIs;
  CUE-RELIANT iff acc(tie) <= acc(pro) - 0.15 and acc(tie) <= 0.75; CONTRIBUTION = acc(pro) - acc(tie) (Wald CI on the difference).
  Per 191-3 the cue cannot be §191's whole mechanism; A0-1 measures a contribution only. No registered §191 reading changes.
    python theory/score_a01.py
"""
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "boxes", "runs_fsq")
OUT = os.path.join(ROOT, "theory", "results", "score_a01.json")
sys.path.insert(0, os.path.join(ROOT, "docs", "research_lang_2026-10-18", "scripts"))
from heur_panel import anchors, panel_for  # noqa: E402

MODELS = ["s191_stepaux_relabel_s1", "s191_textcot_relabel_s1"]
TOL, MIN_N, DMIN = 0.02, 100, 5


def wilson(k, n, z=1.959964):
    if n == 0:
        return None
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]


def strata(test_file):
    data = json.load(open(test_file if os.path.isabs(test_file) else os.path.join(ROOT, test_file)))
    st = {}
    for i, x in enumerate(data):
        if int(x.get("ext", 0)) < 2:
            continue
        a, b = anchors(x); v = panel_for(x, a, b)["proximity"]
        st[i] = "pro" if v == 1.0 else "anti" if v == 0.0 else "tie"
    return st


def model(d):
    f = os.path.join(R, d, "eval_a01.json"); e = os.path.join(R, d, "eval.json")
    if not os.path.exists(f):
        return {"cell": d, "status": "PENDING", "label": "pending (missing eval_a01.json)"}
    A, E = json.load(open(f)), json.load(open(e)); items = A["items"]; m = A.get("meta") or {}
    by = {}
    for it in items:
        by.setdefault(int(it["depth"]), []).append(int(it["correct"]))
    acc = {k: sum(v) / len(v) for k, v in by.items()}; ref = {int(k): v for k, v in (E.get("acc_by_depth") or {}).items()}
    diffs = {k: abs(acc[k] - ref[k]) for k in acc if k in ref and len(by[k]) >= MIN_N}
    gate = bool(diffs) and all(v <= TOL + 1e-12 for v in diffs.values())
    o = {"cell": d, "meta": m, "GATE": {"ok": gate, "abs_diff_by_depth": diffs, "tol": TOL}, "n_items": len(items)}
    if not gate:
        o.update(status="UNREADABLE", label="UNREADABLE (fp32 2070 vs committed bf16 acc_by_depth: %s)" % diffs); return o
    st = strata(m["test_file"]); cnt = {}
    for it in items:
        s = st.get(int(it["i"]))
        if s is None or int(it["depth"]) < DMIN:
            continue
        c = cnt.setdefault(s, [0, 0]); c[0] += int(it["correct"]); c[1] += 1
    r = {s: {"acc": c[0] / c[1], "n": c[1], "ci95": wilson(c[0], c[1])} for s, c in cnt.items()}
    pro, tie = r.get("pro"), r.get("tie")
    contrib = (pro["acc"] - tie["acc"]) if (pro and tie) else None
    se = math.sqrt(pro["acc"] * (1 - pro["acc"]) / pro["n"] + tie["acc"] * (1 - tie["acc"]) / tie["n"]) if (pro and tie) else None
    reliant = bool(pro and tie and tie["acc"] <= pro["acc"] - 0.15 + 1e-12 and tie["acc"] <= 0.75 + 1e-12)
    perd = {}
    for it in items:
        s = st.get(int(it["i"]))
        if s is None or int(it["depth"]) < DMIN:
            continue
        c = perd.setdefault(int(it["depth"]), {}).setdefault(s, [0, 0]); c[0] += int(it["correct"]); c[1] += 1
    o.update(status="SCORED", strata=r, CONTRIBUTION=contrib, CONTRIBUTION_ci95=[contrib - 1.96 * se, contrib + 1.96 * se] if se is not None else None,
             CUE_RELIANT=reliant, per_depth={k: {s: {"acc": c[0] / c[1], "n": c[1]} for s, c in v.items()} for k, v in sorted(perd.items())},
             label="%s | pro %.3f (n %d) tie %.3f (n %d) anti %s | contribution %.3f" % (
                 "CUE-RELIANT" if reliant else "not CUE-RELIANT", pro["acc"], pro["n"], tie["acc"], tie["n"],
                 ("%.3f (n %d)" % (r["anti"]["acc"], r["anti"]["n"])) if "anti" in r else "n/a", contrib))
    return o


def main():
    out = {"registered": "THEORY note 191-4", "models": {d: model(d) for d in MODELS}}
    print("READING: " + " || ".join("%s %s" % (d, v["label"]) for d, v in out["models"].items()))
    json.dump(out, open(OUT, "w", newline=chr(10)), indent=1, default=str)
    return out


if __name__ == "__main__":
    main()
