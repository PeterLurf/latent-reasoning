"""THEORY §217 scoring (CAUSAL-ACROSS-ROUNDS; registered text: THEORY.md '## 217.' + note 217-1; Theory is the only scorer). Eval-only, $0, local RTX 4060.
Per model, eval_causal_deep.json (boxes/vbank_causal.py, note 217-2: vbank_eval's VALUE-SWAP class A + STALL, unchanged except the n-range):
  gate (n 4-8 on the 4060 vs the committed 3070 eval_vb): |follow_ii - ref|, |own_ii - ref|, |STALL recovery - ref| <= 0.005, else UNREADABLE (fails CAUSAL-DEEP).
  competent band B = {n in 9..20 : a(n) >= 0.95}, a(n) = eval_vb acc_n2 (n <= 16) / eval_deep acc_n2 (n 17-20; no eval_deep -> those n unknown, excluded,
    listed); an n with < 100 clean pairs or < 500 STALL items is excluded and listed.
  CAUSAL-DEEP iff B non-empty and, pooled over B: follow_ii >= 0.95 and own_ii <= 0.02 and STALL recovery (mean stalled / mean intact) >= 0.95, and every n in
    B has follow_ii >= 0.90. Empty B -> NOT-APPLICABLE.
  CAUSAL-PRESERVED (lineage) iff CAUSAL-DEEP for every model of the lineage with a non-empty B; else ERODED-AT the first failing round (component named;
    UNREADABLE models fail with component 'gate'). Pending while any of the lineage's models with eval files is pending.
  Descriptive (217-1): bands with a(n); per-n overlap comparison between models of a lineage; clean-pair counts and clean shares per n; follow_iv per n;
    follow_ii / STALL at every n 9-20 (outside B too).
File contract (eval_causal_deep.json): meta {card, dtype, eval_bs, n_lo, n_hi, test_n_max, run}; gate {vswap: vswap_class output at n 4..8, stall: part_stall
  output at n 4..8}; vswap: vswap_class output at n 9..20 (by_n {n: {n_pairs, follow_ii, own_ii}}, clean_share_by_n, plus by_n_iv {n: {follow_iv, own_iv}} and
  candidates_by_n {n: int}); stall {recovery, by_n {n: {intact_kn, stalled_kn1, stall_t, items}}}; pairs [...] and stall_items {...} (per-item store).
    python theory/score_s217.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "boxes", "runs_fsq")
OUT = os.path.join(ROOT, "theory", "results", "score_s217.json")
EPS = 1e-9
LINEAGES = {"s1": ["s213_d3t_s1", "s215_ftself_s1", "s216_r2_s1"], "s2": ["s216_d3t_s2", "s216_r1_s2", "s216_r2_s2"]}
EXTRA = ["s215_ftctrl_s1"]                                                  # descriptive (no lineage row)
CARD_CLASS, GATE_TOL, BAND_ACC = "4060", 0.005, 0.95
MIN_PAIRS, MIN_STALL = 100, 500


def jl(d, f):
    p = os.path.join(R, d, f)
    return json.load(open(p)) if os.path.exists(p) else None


def accs(d):
    """a(n) for n 9..20 from the committed eval_vb (n <= 16) and eval_deep (n 17..20); None where the file is missing."""
    ev, dp = jl(d, "eval_vb.json"), jl(d, "eval_deep.json"); a = {}
    for n in range(9, 21):
        src = ev if n <= 16 else dp
        a[n] = float(src["acc_n2"][str(n)]) if (src and str(n) in (src.get("acc_n2") or {})) else None
    return a


def gate(f, ev):
    """the n 4-8 reproduction gate on the 4060 vs the committed 3070 eval_vb."""
    g = (f or {}).get("gate") or {}; gv, gs = g.get("vswap") or {}, g.get("stall") or {}; rv, rs = (ev or {}).get("vswap") or {}, (ev or {}).get("stall") or {}
    d = {}
    for k, a, b in (("follow_ii", gv.get("follow_ii"), rv.get("follow_ii")), ("own_ii", gv.get("own_ii"), rv.get("own_ii")),
                    ("stall", gs.get("recovery"), rs.get("recovery"))):
        d[k] = None if (a is None or b is None) else abs(float(a) - float(b))
    ok = all(v is not None and v <= GATE_TOL + EPS for v in d.values())
    return {"abs_delta": d, "ok": ok}


def model(d):
    o = {"cell": d}; f = jl(d, "eval_causal_deep.json"); ev = jl(d, "eval_vb.json")
    if ev is None:
        o.update(status="ABSENT", label="absent (no eval_vb.json)"); return o
    if f is None:
        o.update(status="PENDING", label="pending (missing eval_causal_deep.json)"); return o
    m = f.get("meta") or {}
    if CARD_CLASS not in str(m.get("card", "")) or m.get("dtype") != "fp32":
        o.update(status="PENDING", label="pending (card / dtype %s / %s, registered %s fp32)" % (m.get("card"), m.get("dtype"), CARD_CLASS)); return o
    g = gate(f, ev); o["GATE"] = g
    a = accs(d); V = (f.get("vswap") or {}); BY = V.get("by_n") or {}; IV = V.get("by_n_iv") or {}; CAND = V.get("candidates_by_n") or {}
    ST = ((f.get("stall") or {}).get("by_n") or {})
    per_n, band, excluded = {}, [], {}
    for n in range(9, 21):
        b, s = BY.get(str(n)) or {}, ST.get(str(n)) or {}
        per_n[n] = {"a": a[n], "clean_pairs": b.get("n_pairs"), "candidates": CAND.get(str(n)), "clean_share": (V.get("clean_share_by_n") or {}).get(str(n)),
                    "follow_ii": b.get("follow_ii"), "own_ii": b.get("own_ii"), "follow_iv": (IV.get(str(n)) or {}).get("follow_iv"),
                    "stall_intact": s.get("intact_kn"), "stall_stalled": s.get("stalled_kn1"), "stall_items": s.get("items")}
        if a[n] is None:
            excluded[n] = "a(n) unknown (no eval_deep)" if n > 16 else "a(n) unknown"; continue
        if a[n] < BAND_ACC - EPS:
            continue
        if (b.get("n_pairs") or 0) < MIN_PAIRS or (s.get("items") or 0) < MIN_STALL:
            excluded[n] = "thin (%s clean pairs, %s STALL items)" % (b.get("n_pairs"), s.get("items")); continue
        band.append(n)
    o.update(BAND=band, BAND_EXCLUDED=excluded, PER_N=per_n)
    if not g["ok"]:
        o.update(status="UNREADABLE", CAUSAL_DEEP=False, component="gate", label="UNREADABLE (4060 n 4-8 gate failed: %s; rule b)" % g["abs_delta"]); return o
    if not band:
        o.update(status="NOT-APPLICABLE", CAUSAL_DEEP=None, label="NOT-APPLICABLE (empty competent band; excluded %s)" % excluded); return o
    fp = sum(BY[str(n)]["follow_ii"] * BY[str(n)]["n_pairs"] for n in band) / sum(BY[str(n)]["n_pairs"] for n in band)
    op = sum(BY[str(n)]["own_ii"] * BY[str(n)]["n_pairs"] for n in band) / sum(BY[str(n)]["n_pairs"] for n in band)
    si = sum(ST[str(n)]["intact_kn"] for n in band) / len(band); ss = sum(ST[str(n)]["stalled_kn1"] for n in band) / len(band)
    st = ss / si if si > 0 else None; worst = min(BY[str(n)]["follow_ii"] for n in band)
    comp = [c for c, bad in (("swap follow", fp < 0.95 - EPS or worst < 0.90 - EPS), ("own", op > 0.02 + EPS), ("STALL", st is None or st < 0.95 - EPS)) if bad]
    ok = not comp
    o.update(status="SCORED", follow_ii_pooled=fp, own_ii_pooled=op, stall_recovery_pooled=st, follow_ii_min_n=worst, CAUSAL_DEEP=ok, component=comp or None,
             label="%s | band n %s | follow %.3f (min %.3f) own %.3f STALL %.3f" % ("CAUSAL-DEEP" if ok else "not CAUSAL-DEEP (%s)" % ", ".join(comp),
                                                                                      "%d-%d" % (min(band), max(band)), fp, worst, op, st if st is not None else -1))
    return o


def lineage(name, M):
    rows = [M[d] for d in LINEAGES[name]]
    if any(r["status"] == "PENDING" for r in rows):
        return {"status": "PENDING", "label": "pending (%s)" % [r["cell"] for r in rows if r["status"] == "PENDING"]}
    scored = [r for r in rows if r["status"] in ("SCORED", "UNREADABLE")]
    bad = [r for r in scored if not r["CAUSAL_DEEP"]]
    absent = [r["cell"] for r in rows if r["status"] == "ABSENT"]
    if bad:
        r = bad[0]; return {"status": "ERODED-AT", "label": "ERODED-AT %s (%s)" % (r["cell"], r.get("component")), "absent": absent}
    if not scored:
        return {"status": "NOT-APPLICABLE", "label": "no model of the lineage has a competent band", "absent": absent}
    lab = "CAUSAL-PRESERVED over %s" % [r["cell"] for r in scored] + (" (not yet run: %s)" % absent if absent else "")
    return {"status": "CAUSAL-PRESERVED" if not absent else "CAUSAL-PRESERVED (so far)", "label": lab, "absent": absent}


def overlap(name, M):
    """217-1: per-n side-by-side on the n where two models' bands overlap (descriptive)."""
    out = {}
    ds = [d for d in LINEAGES[name] if M[d].get("BAND")]
    for i, x in enumerate(ds):
        for y in ds[i + 1:]:
            ns = sorted(set(M[x]["BAND"]) & set(M[y]["BAND"]))
            if ns:
                out["%s vs %s" % (x, y)] = {n: {k: (M[x]["PER_N"][n][k], M[y]["PER_N"][n][k]) for k in ("follow_ii", "own_ii", "stall_stalled", "stall_intact")} for n in ns}
    return out


def score():
    M = {d: model(d) for d in sum(LINEAGES.values(), []) + EXTRA}
    out = {"registered": "THEORY §217 + note 217-1", "card_class": CARD_CLASS, "models": M,
           "lineages": {k: lineage(k, M) for k in LINEAGES}, "overlap": {k: overlap(k, M) for k in LINEAGES}}
    return out


def main():
    out = score()
    print("READING: " + " || ".join("%s %s" % (k, v["label"]) for k, v in out["lineages"].items()))
    json.dump(out, open(OUT, "w", newline=chr(10)), indent=1, default=str)
    return out


if __name__ == "__main__":
    main()
