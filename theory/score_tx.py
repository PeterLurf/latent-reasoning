"""THEORY §266 SAME-DATA BASELINES scorer (TX_v1 in THEORY §266; Theory is the only scorer; committed before output).
X3 TEXT-CoT-D4: GPT-2 small text CoT trained on bun's REG d <= 4 data (same generator, frames 0-5, §257 distract mix at p 1.0), §227 CurGate recipe;
  cells tx_s1 / tx_s2 (sentence steps "<name> has <v> <unit>.") and txv_s1 (value-only steps). eval_textcot.json: by_d {d: {acc, parsed, n, gen_tokens}} on TEST2
  d 1-10 (free greedy decode, final-answer acc).
X5 ONE-HOP-PIPELINE: on bun_s1 / bun_s2, eval_pipeline.json by_d {d: {acc_bun, acc_ask, n}}: for t = 1..d, S_t is asked of a one-hop rendering in which
  S_t-1's op sentence is replaced by the stated given "S_t-1 has v^_t-1 <u>." (v^ = the pipeline's own previous answer); acc_bun = bun at K' 1 with the cue
  S_t; acc_ask = s233_askname_s1 one pass (descriptive).
Bun reference: bun_s1 / bun_s2 eval_acc by_d[d].acc (K' 8), mean over seeds.
X3 verdict: a text cell is VALID if TEST2 d 1-4 acc >= min(0.98, bun's min d 1-4 - 0.01) (Note 266-1, Audit; else UNDERTRAINED; an UNDERTRAINED cell is
  extended once by +12k, pre-stated, and never counts as evidence for the latent). No valid cell -> X3 UNDERTRAINED -> VOID-EXTEND. Over valid cells, t7 = max d 7 acc,
  t8 = max d 8 acc: LATENT-SPECIFIC if t7 <= bun7 - 0.10 and t8 <= bun8 - 0.10, else NOT-LATENT-SPECIFIC.
Wilson 95% CIs on the binding cells; NEAR appended when a bar lies within 2 SE (Audit standing rule, 11-03).
C7 TEXT-SUBSTITUTE (Audit C7_EDITABLE P1; §247 bars), pooled over cells (d, t) (3,1) (3,2) (4,2) (4,3) per seed: EXPOSED if acc_intact - acc_neutral >= 0.30;
  C7-EDIT if EXPOSED and rt >= 0.95 and follow >= 0.50 and follow >= floor + 0.30; else NO-EDIT / UNEXPOSED. Both seeds C7-EDIT -> C7 partial (scoped).
X5 PIPELINE-MATCHES (Note 266-2, Audit): pipeline acc_bun at d 7 and d 8 >= the bun mean - 0.02 in both seeds -> C1 is worded 'a one-hop executor of a given plan'.
X3P PLAN-GIVEN (Note 266-3, BINDING for X3; Audit F4 / 14_REDTEAM): eval-only on tx_s1 / tx_s2, eval_textcot_plan.json (same schema as eval_textcot.json, incl.
  trail): greedy decode with the per-step plan forced: at the start of text step t (t = 1..d) the cued name S_t is inserted, the model completes the step; after
  step d " The answer is" is inserted and the number read. Same models / items / decode otherwise. VALID gate as above (in-range + Note 266-2 trailing gate,
  both on the plan decode). X3 verdict (LATENT-SPECIFIC / NOT-LATENT-SPECIFIC) is read over VALID plan cells; no valid plan cell -> X3 VOID (UNDERTRAINED-PLAN):
  never LATENT-SPECIFIC. The free-decode cells (tx_s1 / tx_s2 / txv_s1) are reported as descriptive 'x3_free' with the pre-stated rule.
X5 verdict: VALID if acc_bun d 1 >= 0.98 in both seeds; PIPELINE-SOLVES if acc_bun d 7 >= 0.95 and d 8 >= 0.90 in both seeds, else PIPELINE-FAILS.
    python -B theory/score_tx.py [--fixtures]
"""
import argparse, json, math, os, sys, tempfile
_HR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(_HR, "boxes", "runs_fsq")
OUT = os.path.join(_HR, "theory", "results", "score_tx.json")
TEXT = ("tx_s1", "tx_s2", "txv_s1")
PLAN = ("tx_s1", "tx_s2")   # Note 266-3: plan-given decode (eval_textcot_plan.json); txv has no names to force
BUN = ("bun_s1", "bun_s2")


def jl(p):
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return None


def wilson(p, n, z=1.96):
    """Audit standing rule 11-03 (1): Wilson 95% CI on every binding proportion."""
    if not n:
        return (0.0, 1.0)
    c = (p + z * z / (2 * n)) / (1 + z * z / n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n); return (c - h, c + h)


def near(p, n, bar):
    """NEAR: the bar lies within 2 SE of the reading."""
    return bool(n) and abs(p - bar) <= 2 * math.sqrt(max(p * (1 - p), 1e-12) / n)


def score(runs=R):
    o = {"section": 266}
    tx = {c: jl(os.path.join(runs, c, "eval_textcot.json")) for c in TEXT}; txp = {c: jl(os.path.join(runs, c, "eval_textcot_plan.json")) for c in PLAN}; bn = {b: jl(os.path.join(runs, b, "eval_acc.json")) for b in BUN}
    pp = {b: jl(os.path.join(runs, b, "eval_pipeline.json")) for b in BUN}
    if any(v is None for v in list(tx.values()) + list(txp.values()) + list(bn.values()) + list(pp.values())):
        o["verdict"] = "pending"; return o
    b7 = sum(bn[b]["by_d"]["7"]["acc"] for b in BUN) / 2; b8 = sum(bn[b]["by_d"]["8"]["acc"] for b in BUN) / 2
    bmin = min(bn[b]["by_d"][str(d)]["acc"] for b in BUN for d in (1, 2, 3, 4)) if all("1" in bn[b]["by_d"] for b in BUN) else 1.0
    # Note 266-1 (Audit): in-distribution validity gate min(0.98, bun's min d 1-4 - 0.01)
    mk = lambda v: {"valid": all(v["by_d"][str(d)]["acc"] >= min(0.98, bmin - 0.01) for d in (1, 2, 3, 4)) and trail_ok(v), "trail": trail_desc(v), **{"d%d" % d: v["by_d"][str(d)]["acc"] for d in range(1, 11)},
                     "parsed_min": min(v["by_d"][str(d)]["parsed"] for d in range(1, 11)), "gen_tokens": {str(d): v["by_d"][str(d)].get("gen_tokens") for d in range(1, 11)}}
    pcells = {c: mk(v) for c, v in txp.items()}
    cells = {c: {"valid": all(v["by_d"][str(d)]["acc"] >= min(0.98, bmin - 0.01) for d in (1, 2, 3, 4)) and trail_ok(v), "trail": trail_desc(v), **{"d%d" % d: v["by_d"][str(d)]["acc"] for d in range(1, 11)},
                 "parsed_min": min(v["by_d"][str(d)]["parsed"] for d in range(1, 11)), "gen_tokens": {str(d): v["by_d"][str(d)].get("gen_tokens") for d in range(1, 11)}}
             for c, v in tx.items()}
    void = [c for c in TEXT if min(tx[c]["by_d"][str(d)]["parsed"] for d in (1, 2, 3, 4)) < 0.9]   # Audit SCORER_BUGS MED-5: parse failure = build bug
    void += [c + "/plan" for c in PLAN if min(txp[c]["by_d"][str(d)]["parsed"] for d in (1, 2, 3, 4)) < 0.9]
    if void:
        o["verdict"] = "VOID-RERUN once (parsed < 0.9 at d 1-4: %s)" % ",".join(void); o["text_void"] = void; return o
    val = [c for c in TEXT if cells[c]["valid"]]
    # Note 266-3: the free-decode reading is descriptive (pre-stated rule kept visible)
    if not val:
        x3f = "x3_free UNDERTRAINED"
    else:
        f7 = max(cells[c]["d7"] for c in val); f8 = max(cells[c]["d8"] for c in val)
        x3f = "x3_free %s (free text max d7 %.3f / d8 %.3f)" % ("LATENT-SPECIFIC" if f7 <= b7 - 0.10 and f8 <= b8 - 0.10 else "NOT-LATENT-SPECIFIC", f7, f8)
    o["x3_free_desc"] = x3f
    fcells = cells
    cells, tx = pcells, txp   # X3 binds on the plan-given decode (Note 266-3)
    val = [c for c in PLAN if cells[c]["valid"]]
    if not val:
        x3 = "X3 VOID (UNDERTRAINED-PLAN: no plan-given cell VALID: d 1-4 >= min(0.98, bun - 0.01) and the Note 266-2 trailing-number gate) -> VOID-EXTEND once (+12k)"
    else:
        t7 = max(cells[c]["d7"] for c in val); t8 = max(cells[c]["d8"] for c in val)
        x3 = "X3 %s (plan-given text max d7 %.3f / d8 %.3f vs bun %.3f / %.3f)" % ("LATENT-SPECIFIC" if t7 <= b7 - 0.10 and t8 <= b8 - 0.10 else "NOT-LATENT-SPECIFIC", t7, t8, b7, b8)
        for c in val:
            n_ = tx[c]["by_d"]["7"]["n"]
            cells[c]["ci"] = {"d7": wilson(cells[c]["d7"], n_), "d8": wilson(cells[c]["d8"], tx[c]["by_d"]["8"]["n"])}
            cells[c]["near"] = {"d7": near(cells[c]["d7"], n_, b7 - 0.10), "d8": near(cells[c]["d8"], tx[c]["by_d"]["8"]["n"], b8 - 0.10)}
        if any(any(cells[c]["near"].values()) for c in val):
            x3 += " NEAR"
    pv = all(pp[b]["by_d"]["1"]["acc_bun"] >= 0.98 for b in BUN)
    if not pv:
        x5 = "X5 INVALID (one-hop d 1 < 0.98)"
    else:
        o["pipeline_ci"] = {b: {d: wilson(pp[b]["by_d"][d]["acc_bun"], pp[b]["by_d"][d]["n"]) for d in ("1", "7", "8")} for b in BUN}
        bun_by = {d: sum(json.load(open(os.path.join(runs, bb, "eval_acc.json")))["by_d"][d]["acc"] for bb in BUN) / 2 for d in ("7", "8")}
        o["pipeline_matches"] = all(pp[bb]["by_d"][d]["acc_bun"] >= bun_by[d] - 0.02 for bb in BUN for d in ("7", "8"))   # Note 266-2 (Audit): PIPELINE-MATCHES
        x5 = "X5 " + ("PIPELINE-SOLVES" if all(pp[b]["by_d"]["7"]["acc_bun"] >= 0.95 and pp[b]["by_d"]["8"]["acc_bun"] >= 0.90 for b in BUN) else "PIPELINE-FAILS")
    c = c7(runs)
    if c is None:
        o["verdict"] = "pending"; return o
    o["c7"] = c; x5 = x5 + (" (PIPELINE-MATCHES)" if o.get("pipeline_matches") else "") + "; " + c["verdict"]
    o.update({"bun_d7": b7, "bun_d8": b8, "text_plan": cells, "text_free_desc": fcells, "pipeline": {b: pp[b]["by_d"] for b in BUN}, "x3": x3, "x5": x5, "verdict": x3 + "; " + x5 + "; " + x3f}); return o


def trail_desc(v):
    """Note 266-2 (Audit lead, 2605.22870 'small-model text CoT copies the last number'): eval_textcot.json "trail" {d: {"acc_trailing": x,
    "copy_rate": y, "base_copy": z}} for d 1-4. acc_trailing = acc on the same items with a trailing irrelevant sentence '<absent name> has x <u>.'
    appended after the last op sentence (x random, seed 26610); copy_rate = P(pred == the prompt's last number) on the trailing rendering;
    base_copy = P(ans == that number)."""
    t = v.get("trail")
    if not t:
        return None
    return {d: {"drop": v["by_d"][d]["acc"] - t[d]["acc_trailing"], "copy_excess": t[d]["copy_rate"] - t[d]["base_copy"]} for d in t}


def trail_ok(v):
    """VALID gate (Note 266-2): mean over d 1-4 of drop <= 0.05 and of copy_excess <= 0.05; missing trail fields -> not valid."""
    td = trail_desc(v)
    if not td or not all(str(d) in td for d in (1, 2, 3, 4)):
        return False
    return sum(td[str(d)]["drop"] for d in (1, 2, 3, 4)) / 4 <= 0.05 and sum(td[str(d)]["copy_excess"] for d in (1, 2, 3, 4)) / 4 <= 0.05


def c7(runs):
    """C7 TEXT-SUBSTITUTE (Audit C7_EDITABLE P1): eval_text_sub.json on bun_s1 / bun_s2, cells (d, t) in (3,1) (3,2) (4,2) (4,3), keys per cell:
    n, rt (acc with S_t's op sentence replaced by the given "S_t has v_t <u>." for pass t's input only), follow (P(final == the program recomputed with v_t := x')
    under x' != v_t), floor (P(intact final == that recomputed answer)), acc_intact, acc_neutral (S_t's op sentence replaced by "<absent name> has x' <u>.")."""
    f = {b: jl(os.path.join(runs, b, "eval_text_sub.json")) for b in BUN}
    if any(v is None for v in f.values()):
        return None
    out = {}
    for b, v in f.items():
        C = v["cells"]; n = sum(c["n"] for c in C.values()); pool = lambda k: sum(c[k] * c["n"] for c in C.values()) / n
        rt, fo, fl, ai, an = pool("rt"), pool("follow"), pool("floor"), pool("acc_intact"), pool("acc_neutral")
        valid = ai - an >= 0.30
        ok = valid and rt >= 0.95 and fo >= 0.50 and fo >= fl + 0.30
        out[b] = {"rt": rt, "follow": fo, "floor": fl, "acc_intact": ai, "acc_neutral": an, "exposed": valid, "ci_follow": wilson(fo, n), "near_follow": near(fo, n, 0.50),
                  "label": "C7-EDIT" if ok else ("UNEXPOSED" if not valid else "NO-EDIT")}
    labs = [x["label"] for x in out.values()]
    out["verdict"] = "C7 TEXT-SUBSTITUTE " + ("C7-EDIT 2/2" if labs == ["C7-EDIT"] * 2 else "/".join(labs)); return out


def fixtures():
    res = []

    def world(t14=1.0, t7=0.5, t8=0.3, tv7=0.4, und2=False, p1=1.0, p7=0.99, p8=0.97, rt=1.0, fo=0.8, an=0.3, tdrop=0.0, cex=0.0, parsed=1.0,
              q14=None, q7=None, q8=None, qcex=None, qparsed=None):
        dd = tempfile.mkdtemp()

        def put(run, name, obj):
            os.makedirs(os.path.join(dd, run), exist_ok=True); json.dump(obj, open(os.path.join(dd, run, name), "w"))
        for c in TEXT:
            a7 = tv7 if c == "txv_s1" else t7; lo = t14 if not (und2 and c != "tx_s1") else 0.5
            put(c, "eval_textcot.json", {"trail": {str(d): {"acc_trailing": lo - tdrop, "copy_rate": 0.1 + cex, "base_copy": 0.1} for d in range(1, 5)},
                                      "by_d": {str(d): {"acc": lo if d <= 4 else (0.9 if d < 7 else a7 if d == 7 else t8 if d == 8 else 0.1), "parsed": parsed, "n": 100, "gen_tokens": 10 * d}
                                               for d in range(1, 11)}})
        for c in PLAN:   # Note 266-3 plan-given decode; defaults mirror the free cells
            l2 = t14 if q14 is None else q14; a7 = t7 if q7 is None else q7; a8 = t8 if q8 is None else q8
            ce = cex if qcex is None else qcex; pr = parsed if qparsed is None else qparsed
            if und2 and c != "tx_s1" and q14 is None:
                l2 = 0.5
            put(c, "eval_textcot_plan.json", {"trail": {str(d): {"acc_trailing": l2 - tdrop, "copy_rate": 0.1 + ce, "base_copy": 0.1} for d in range(1, 5)},
                                           "by_d": {str(d): {"acc": l2 if d <= 4 else (0.9 if d < 7 else a7 if d == 7 else a8 if d == 8 else 0.1), "parsed": pr, "n": 100, "gen_tokens": 8 * d}
                                                    for d in range(1, 11)}})
        for b in BUN:
            put(b, "eval_acc.json", {"by_d": {"7": {"acc": 0.86}, "8": {"acc": 0.53}}})
            put(b, "eval_text_sub.json", {"cells": {k: {"n": 100, "rt": rt, "follow": fo, "floor": 0.1, "acc_intact": 1.0, "acc_neutral": an} for k in ("3_1", "3_2", "4_2", "4_3")}})
            put(b, "eval_pipeline.json", {"by_d": {str(d): {"acc_bun": p1 if d == 1 else p7 if d == 7 else p8 if d == 8 else 0.99, "acc_ask": 0.9, "n": 100} for d in range(1, 11)}})
        return dd
    for name, kw, want in (("latent-specific + pipeline solves", {}, "X3 LATENT-SPECIFIC"), ("pipeline solves", {}, "X5 PIPELINE-SOLVES"),
                           ("text extrapolates d7", {"t7": 0.8}, "X3 NOT-LATENT-SPECIFIC"), ("text extrapolates d8 only", {"t8": 0.5}, "X3 NOT-LATENT-SPECIFIC"),
                           ("value arm extrapolates (free, descriptive)", {"tv7": 0.85}, "x3_free NOT-LATENT-SPECIFIC"), ("all undertrained", {"t14": 0.8}, "X3 VOID (UNDERTRAINED-PLAN"),
                           ("0.97 in-range is undertrained", {"t14": 0.97}, "X3 VOID (UNDERTRAINED-PLAN"),
                           ("trailing-number copy shortcut voids text cells", {"cex": 0.2, "t7": 0.85}, "X3 VOID (UNDERTRAINED-PLAN"),
                           ("trailing sentence breaks text cells", {"tdrop": 0.2, "t7": 0.85}, "X3 VOID (UNDERTRAINED-PLAN"),
                           ("266-3: plan arm VALID fail (copy) -> X3 VOID even if free text is valid", {"qcex": 0.2}, "X3 VOID (UNDERTRAINED-PLAN"),
                           ("266-3: plan arm in-range fail -> VOID, never LATENT-SPECIFIC", {"q14": 0.9}, "X3 VOID (UNDERTRAINED-PLAN"),
                           ("266-3: plan-given text wins at d7 -> NOT-LATENT-SPECIFIC though free text loses", {"q7": 0.85}, "X3 NOT-LATENT-SPECIFIC"),
                           ("266-3: plan-given text wins at d8 only -> NOT-LATENT-SPECIFIC", {"q8": 0.5}, "X3 NOT-LATENT-SPECIFIC"),
                           ("266-3: free text wins but plan-given loses -> X3 LATENT-SPECIFIC, free reading descriptive", {"t7": 0.85, "q7": 0.5}, "X3 LATENT-SPECIFIC"),
                           ("266-3: unparsed plan decode is VOID-RERUN", {"qparsed": 0.5}, "VOID-RERUN"),
                           ("unparsed text cell is VOID-RERUN (MED-5)", {"parsed": 0.5}, "VOID-RERUN"),
                           ("one valid cell decides", {"und2": True, "t7": 0.5}, "X3 LATENT-SPECIFIC"), ("pipeline invalid", {"p1": 0.9}, "X5 INVALID"),
                           ("pipeline fails d8", {"p8": 0.8}, "PIPELINE-FAILS"), ("c7 edit", {}, "C7-EDIT 2/2"), ("c7 no edit", {"fo": 0.2}, "NO-EDIT/NO-EDIT"),
                           ("c7 round trip", {"rt": 0.9}, "NO-EDIT/NO-EDIT"), ("c7 unexposed", {"an": 0.9}, "UNEXPOSED/UNEXPOSED")):
        o = score(world(**kw)); res.append((name, want in o["verdict"], o["verdict"]))
    for r in res:
        print("FIXTURE", "PASS" if r[1] else "FAIL", r[0], "->", r[2])
    k = sum(bool(r[1]) for r in res); print("fixtures %d/%d" % (k, len(res))); return k == len(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--fixtures", action="store_true"); a = ap.parse_args()
    if a.fixtures:
        sys.exit(0 if fixtures() else 1)
    o = score(); print(o["verdict"])
    os.makedirs(os.path.dirname(OUT), exist_ok=True); json.dump(o, open(OUT, "w", encoding="utf-8", newline=chr(10)), indent=1, default=str)
