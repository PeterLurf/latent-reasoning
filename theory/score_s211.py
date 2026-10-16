"""THEORY §211 scoring (VERSION-BANK + READY-RECODE, Stage 1; registered text docs/THEORY_VERSIONBANK_DRAFT_2026-10-15.md at the commit named
in §211 + its dated notes; Theory is the only scorer).

Cells (boxes/runs_fsq/): s211_vb_pa_s1 (N2 PURE-ANNEAL, main), s211_vb_carry_s1 (N1 CARRY, control), s211_vb_pamct_s1 (N3 N2 + MCT).
Per cell:
  log.jsonl    - "step", "loss", "answer_ce" (readiness-window answer CE). NO-ENTRY: no 1k-bin of answer_ce stays <= 0.5 to the end (basin_entry's
                 fixed-bin rule applied to answer_ce). Dead: last logged step < 48000.
  eval_vb.json - local fp32 at the original eval_bs (contract, note 211-1):
    acc_n2: {n: acc at K' = n + 2}, n = 1..16;  acc_k: {K': {n: acc}}, K' in 0..16, 24
    probe: {t: {acc_by_n: {n: acc}, items: [0/1 per test item]}}, t in {0,1,2,3,4,6,8,12};  test_meta: {n: [...], height: [...]} (same item order)
    vswap: {n_pairs, follow_i, own_i, follow_ii, own_ii, own_iv, [follow_iii, own_iii]}  (clean-ccc pairs, n 4..8 pooled)
    stall: {recovery}   (acc at K' = n + 1 with a copied pass at ceil(n/2), / intact acc at K' = n, n 4..8 pooled)
    stepcap: {copy_slot_acc, value_stepcap_n6_8, stationary_slope}
    carry_likeness (N2/N3), acc_tf_n3_8 (N3: pooled n 3..8 at K' = n + 2 with MCT applied to every version at eval)
Readings (draft §7 + Addendum 1):
  DEEP = acc_n2 >= 0.90 at every n 3..8.  PROBE-VALID = PROBE(12) pooled n 3..8 >= ACC(n+2 pooled) - 0.05 (else every PROBE reading fails).
  PROBE-GAIN = ACC(n+2 pooled n 3..8) - max(PROBE(0), PROBE(1)) pooled >= 0.50.
  SERIAL-ONSET: onset_i = min t in T with probe-correct at every t' in T, t <= t' <= n_i + 4 (never -> inf); coverage >= 0.8 at every n 3..8;
    OLS slope of median onset on tree height (heights with >= 30 items, n 3..8) >= 0.8; median onset(n 8) - median onset(n 3) >= 3.
  ONESHOT-PRECOMPUTE = PROBE(2) at n 8 >= 0.8.  SERIAL-STEP = vswap n_pairs >= 100 and follow_ii >= 0.6 and own_ii <= 0.3 (< 100 -> UNREADABLE = not met).
  NO-CLOCK = stall recovery >= 0.9.  VB-C1 = DEEP and PROBE-GAIN and SERIAL-ONSET and SERIAL-STEP and NO-CLOCK.
  Labels: NO-ENTRY, LOSES-ONE-PASS (not DEEP and PROBE-GAIN value < 0.2), CARRY-NO-STEP (copy_slot_acc >= 0.99 and value_stepcap_n6_8 <= 0.8),
  COUNT-KEYED (not NO-CLOCK, or DEEP and REACH(16) <= 8 and DEPTH < 0.5), RECOMPUTE-V0 (own_ii > 0.3 and own_iv <= 0.1),
  TF-ONLY (N3: not DEEP and acc_tf_n3_8 >= 0.9). Censoring (rule b): dead or NO-ENTRY -> every reading fails; missing eval -> pending.
    python theory/score_s211.py
"""
import json
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "boxes", "runs_fsq")
sys.path.insert(0, os.path.join(ROOT, "theory"))
CELLS = {"N2": "s211_vb_pa_s1", "N1": "s211_vb_carry_s1", "N3": "s211_vb_pamct_s1"}
T_SET = (0, 1, 2, 3, 4, 6, 8, 12)
NS = range(3, 9)
EPS, FULL = 1e-9, 48000
INF = float("inf")


def jl(d, f):
    p = os.path.join(R, d, f)
    return json.load(open(p)) if os.path.exists(p) else None


def log_recs(d):
    out = {}
    p = os.path.join(R, d, "log.jsonl")
    if not os.path.exists(p):
        return out
    for line in open(p):
        r = json.loads(line)
        if "step" in r:
            out[int(r["step"])] = r                                         # last occurrence wins (resumes)
    return out


def entry_on(recs, key, thr):
    b = {}
    for s, r in recs.items():
        if r.get(key) is not None:
            b.setdefault((s - 1) // 1000, []).append(float(r[key]))
    bm = {k: sum(v) / len(v) for k, v in sorted(b.items())}; ks = sorted(bm)
    for i, k in enumerate(ks):
        if all(bm[j] < thr for j in ks[i:]):
            return 1000 * (k + 1)
    return None


def pooled(by_n, ns=NS):
    v = [by_n[str(n)] for n in ns if str(n) in by_n]
    return sum(v) / len(v) if len(v) == len(list(ns)) else None


def ols_slope(xs, ys):
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    den = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den > 0 else None


def onset_reading(ev):
    pr, meta = ev["probe"], ev["test_meta"]; ns, hs = meta["n"], meta["height"]; out = {}
    on = []
    for i, n in enumerate(ns):
        o = INF
        for t in T_SET:
            ts = [u for u in T_SET if t <= u <= n + 4]
            if ts and all(pr[str(u)]["items"][i] for u in ts):
                o = t; break
        on.append(o)
    cov = {n: sum(1 for i, m in enumerate(ns) if m == n and on[i] < INF) / max(1, sum(1 for m in ns if m == n)) for n in NS}
    med_n = {n: statistics.median([on[i] for i, m in enumerate(ns) if m == n]) for n in NS if any(m == n for m in ns)}
    byh = {}
    for i, n in enumerate(ns):
        if 3 <= n <= 8 and on[i] < INF:
            byh.setdefault(hs[i], []).append(on[i])
    hk = sorted(h for h in byh if len(byh[h]) >= 30)
    slope_h = ols_slope(hk, [statistics.median(byh[h]) for h in hk]) if len(hk) >= 2 else None
    nk = [n for n in NS if n in med_n and med_n[n] < INF]
    slope_n = ols_slope(nk, [med_n[n] for n in nk]) if len(nk) >= 2 else None
    gap = (med_n.get(8, INF) - med_n.get(3, INF)) if med_n.get(3, INF) < INF and med_n.get(8, INF) < INF else None
    ok = (all(cov[n] >= 0.8 - EPS for n in NS) and slope_h is not None and slope_h >= 0.8 - EPS and gap is not None and gap >= 3 - EPS)
    out.update(coverage=cov, median_onset_by_n=med_n, slope_height=slope_h, slope_n=slope_n, gap_8_3=gap, SERIAL_ONSET=bool(ok))
    return out


def cell_read(tag, d):
    o = {"cell": d}; recs = log_recs(d); ev = jl(d, "eval_vb.json")
    last = max(recs) if recs else 0; o["last_step"] = last
    if ev is None:
        o["label"] = "pending (eval_vb.json missing)" if recs else "pending (not started)"; return o
    dead = last < FULL; e05 = entry_on(recs, "answer_ce", 0.5); o.update(dead=dead, entry_answer_ce_05=e05,
                                                                        entry_loss_05=entry_on(recs, "loss", 0.5), entry_answer_ce_01=entry_on(recs, "answer_ce", 0.1))
    no_entry = e05 is None
    a2 = ev["acc_n2"]; acc = pooled(a2); deep = all(a2[str(n)] >= 0.90 - EPS for n in NS)
    pr = ev["probe"]; p = {t: pooled(pr[str(t)]["acc_by_n"]) for t in T_SET}
    valid = p[12] is not None and p[12] >= acc - 0.05 - EPS
    pgain = acc - max(p[0], p[1])
    oneshot = pr["2"]["acc_by_n"]["8"] >= 0.8 - EPS
    on = onset_reading(ev) if valid else {"SERIAL_ONSET": False, "note": "PROBE-INVALID"}
    vs = ev.get("vswap") or {}; npairs = vs.get("n_pairs", 0)
    sstep = npairs >= 100 and vs["follow_ii"] >= 0.6 - EPS and vs["own_ii"] <= 0.3 + EPS
    stall = (ev.get("stall") or {}).get("recovery"); noclock = stall is not None and stall >= 0.9 - EPS
    sc = ev.get("stepcap") or {}
    depth = sum(a2[str(n)] for n in range(9, 13)) / 4
    reach = max([n for n in range(1, 17) if a2[str(n)] >= 0.8 - EPS] or [0])
    o.update(DEEP=deep, ACC_n2_pooled=acc, PROBE=p, PROBE_VALID=valid, PROBE_GAIN_value=pgain, PROBE_GAIN=bool(valid and pgain >= 0.50 - EPS),
             ONESHOT_PRECOMPUTE=bool(valid and oneshot), onset=on, vswap_pairs=npairs, SERIAL_STEP=bool(sstep), stall_recovery=stall, NO_CLOCK=bool(noclock),
             DEPTH_9_12=depth, REACH=reach, carry_likeness=ev.get("carry_likeness"))
    vb_c1 = deep and o["PROBE_GAIN"] and on["SERIAL_ONSET"] and sstep and noclock
    labels = []
    if dead:
        labels.append("FAILURE (log ends at %d < 48k; rule b)" % last)
    if no_entry:
        labels.append("NO-ENTRY (rule b)")
    if dead or no_entry:
        vb_c1 = False
    if vb_c1:
        labels.append("VB-C1")
    if o["ONESHOT_PRECOMPUTE"]:
        labels.append("ONESHOT-PRECOMPUTE (premise falsified)")
    if not deep and pgain < 0.2:
        labels.append("LOSES-ONE-PASS")
    if sc.get("copy_slot_acc") is not None and sc["copy_slot_acc"] >= 0.99 - EPS and sc.get("value_stepcap_n6_8") is not None and sc["value_stepcap_n6_8"] <= 0.8 + EPS:
        labels.append("CARRY-NO-STEP")
    if (stall is not None and not noclock) or (deep and reach <= 8 and depth < 0.5):
        labels.append("COUNT-KEYED")
    if npairs >= 100 and vs.get("own_ii", 0) > 0.3 + EPS and vs.get("own_iv") is not None and vs["own_iv"] <= 0.1 + EPS:
        labels.append("RECOMPUTE-V0")
    if npairs < 100:
        labels.append("VALUE-SWAP UNREADABLE (%d pairs; not met)" % npairs)
    if tag == "N3" and not deep and ev.get("acc_tf_n3_8") is not None and ev["acc_tf_n3_8"] >= 0.9 - EPS:
        labels.append("TF-ONLY")
    if not valid:
        labels.append("PROBE-INVALID (PROBE readings fail)")
    o["VB_C1"] = bool(vb_c1); o["labels"] = labels; o["label"] = ", ".join(labels) if labels else "NOT VB-C1"
    return o


def main():
    out = {k: cell_read(k, d) for k, d in CELLS.items()}
    line = " | ".join("%s %s: %s" % (k, CELLS[k], out[k]["label"]) for k in CELLS)
    out["reading"] = line; print("READING:", line)
    json.dump(out, open(os.path.join(ROOT, "theory", "results", "score_s211.json"), "w", newline=chr(10)), indent=1, default=str)


if __name__ == "__main__":
    main()
