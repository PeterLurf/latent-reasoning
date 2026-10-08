"""THEORY §162 scoring (rules fixed in §162; written before any §162 implementation or run; Theory is the only scorer).
DELTA-S5: §155 R's write-after-read memory (DeltaMem-R) as GPT-2's only cross-segment channel on S5 (one item per segment, FINAL only, §120's S5 data /
curriculum / recipe), against a measured single-pass boundary: DIRECT = GPT-2 with the whole chain in one context -> the final state text.
Boundary: each DIRECT seed's own L*_s = the smallest L in {2, 4, 8, 16, 32, 64} with final-state accuracy < 0.90; L* = the lower of s1 / s2 ("boundary
uncertain" if they differ; a seed below 0.90 at L = 1 did not train and is dropped). Fixed in a dated note before any DELTA-S5 output is read.
SHORT trains to l_max = L* / 2 (read descriptively if l_max < 4; not run if L* = 2); LONG to 32 (L* <= 16) or 64 (L* = 32, dirs s162_long64_*).
Probes are 120-way (items uniform over the 120 elements; chance 1/120). At p = min(16, l_max) (p >= 4):
  STATE-LIKE(run): the state edit propagates (propagation - null >= 0.5, random - null < 0.1; 155-3's form) AND state_probe >= 0.90 and >= the mean
  op-history probe over the older half (lags j >= p / 2).  BUFFER-LIKE(run): no propagation AND that older-half op probe >= 0.90.  Else MIXED.
Arm kind: both seeds agree -> that kind; they disagree -> s3 runs and the 2-of-3 majority decides, else "seed-dependent".
Readings need EXACT arms (final-state accuracy >= 0.99 at every training length, each seed).
CAPACITY / DEPTH at 512 (long_acc, cumulative): predicted >= 0.90 STATE-LIKE, < 0.50 BUFFER-LIKE. CAPACITY at 2,048 (acc_2048, cumulative, 500 chains;
past the 12 x 64 key budget): predicted < 0.50 for BUFFER-LIKE; a STATE-LIKE run below 0.90 there reads as drift (C6), not capacity.
SERIAL (§128's thresholds) is a sanity check here (a whole-memory swap = the hybrid input, in distribution; 158-1).
Dirs boxes/runs_fsq/s162_{direct,short,long,long64}_s{1,2,3}.
    python theory/score_s162.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "boxes", "runs_fsq")
GRID = (2, 4, 8, 16, 32, 64)
DELTA = {"family": "s5", "carrier": "deltamem", "mem_write": "top", "write_mix_start": 2000, "write_mix_end": 6000, "steps": 24000}
ERRORS = []


def load(d, f="eval.json"):
    p = os.path.join(R, d, f)
    return json.load(open(p)) if os.path.exists(p) else None


def serial(dn):
    return bool(dn) and all(v["donor_consistent"] >= 0.90 and v["recipient_consistent"] - v.get("chance_overlap", 0.0) < 0.10 for v in dn.values())


def trigger(r):
    sp = ((r.get("s155") or {}).get("write_grad_spikes_floor")) or []; nf = int(r.get("n_nonfinite") or 0)
    return {"floor_spikes": len(sp), "n_nonfinite": nf, "fires": bool(sp or nf)}


def boundary():
    """Per-seed L*_s; L* = the lower (Audit); 'uncertain' when the seeds' boundaries differ. None = passes at every grid length."""
    per, curves, notes, n_done = {}, {}, [], 0
    for s in (1, 2):
        r = load("s162_direct_s%d" % s); a = load("s162_direct_s%d" % s, "args.json") or {}
        if r is None or a.get("mem_check") or a.get("train_n") or int(a.get("steps", 0) or 0) != 24000:
            continue
        n_done += 1; ab = r.get("acc_by_length") or {}; curves[s] = ab
        if (ab.get("1") or 0) < 0.90:
            notes.append("direct s%d did not train (L = 1: %s)" % (s, ab.get("1"))); continue
        fail = [L for L in GRID if float(ab.get(str(L)) or 0) < 0.90]
        per[s] = fail[0] if fail else None
    if n_done < 2:
        return {"status": "pending (%d / 2 DIRECT seeds)" % n_done, "L_star": None, "curves": curves, "notes": notes}
    if not per:
        return {"status": "DIRECT did not train", "L_star": None, "curves": curves, "notes": notes}
    vals = list(per.values()); fin = [v for v in vals if v is not None]
    ls = min(fin) if fin else None
    unc = len(per) == 2 and vals[0] != vals[1]
    return {"status": "fixed", "L_star": ls, "per_seed": per, "boundary_uncertain": unc, "curves": curves, "notes": notes}


def plan(ls):
    """SHORT / LONG training lengths implied by L* (fixed rule)."""
    if ls is None:
        return {"short_lmax": None, "long_lmax": 32, "depth_reached": False}
    return {"short_lmax": ls // 2 if ls >= 4 else None, "long_lmax": 64 if ls == 32 else 32, "depth_reached": ls <= 32}


def run(arm, s, lmax):
    d = "s162_%s_s%d" % (arm, s); r = load(d); a = load(d, "args.json") or {}
    if r is None:
        return {"status": "running" if a else "open"}
    if a.get("mem_check") or a.get("train_n"):
        return {"status": "smoke"}
    bad = {k: a.get(k) for k, v in dict(DELTA, l_max=lmax).items() if a.get(k) != v}
    if bad:
        ERRORS.append((d, bad)); return {"status": "error", "error": bad}
    x = r.get("s162") or {}; fl = r.get("final_acc_by_length") or {}
    exact = all((fl.get(str(L)) or 0) >= 0.99 for L in (1, 2, 4, 8, 16, 32, 64) if L <= lmax)
    p = min(16, lmax); sp = (x.get("state_probe") or {}).get(str(p)); opp = (x.get("op_probe") or {}).get(str(p)) or {}
    old = [float(v) for j, v in opp.items() if int(j) >= p / 2]; old_m = sum(old) / len(old) if old else None
    e = x.get("state_edit") or {}; ps, rs, nv = e.get("propagation_share"), e.get("random_edit_share"), e.get("null_value")
    prop = bool(ps is not None and rs is not None and nv is not None and ps - nv >= 0.5 and rs - nv < 0.1)
    if p < 4:
        kind = "TOO SHORT"                                                   # Audit: buffer and state are nearly the same object at p < 4
    elif prop and sp is not None and old_m is not None and sp >= 0.90 and sp >= old_m:
        kind = "STATE-LIKE"
    elif (not prop) and old_m is not None and old_m >= 0.90:
        kind = "BUFFER-LIKE"
    else:
        kind = "MIXED"
    return {"status": "done", "exact": exact, "final_acc_by_length": fl, "long_acc_512": r.get("long_acc"), "acc_2048": r.get("acc_2048"),
            "state_probe_p": sp, "op_probe_old_mean": old_m, "probe_p": p, "probe_n": x.get("probe_n_score"),
            "edit": {"propagation_share": ps, "random_edit_share": rs, "null_value": nv, "n": e.get("n")}, "propagates": prop, "kind": kind,
            "serial_sanity": serial(r.get("donor")), "key_cos": x.get("key_cos"),
            "rho_loop": ((((r.get("s155") or {}).get("recurrence") or [{}])[-1]) or {}).get("rho_loop"), "fallback_trigger": trigger(r), "gpu": r.get("train_gpu")}


def arm_kind(runs):
    """(kind, why): kind None = pending; False = not exact; "SPLIT" = s1 / s2 disagree and s3 is not in; else the agreed / majority kind."""
    d12 = [runs[s] for s in (1, 2) if runs[s].get("status") == "done"]
    if len(d12) < 2:
        return None, "pending (%d / 2 seeds)" % len(d12)
    done = [v for v in runs.values() if v.get("status") == "done"]
    if not all(v["exact"] for v in done):
        return False, "not exact (%s)" % [v["exact"] for v in done]
    k1, k2 = d12[0]["kind"], d12[1]["kind"]
    if k1 == k2:
        return k1, "s1 / s2 agree"
    if runs[3].get("status") != "done":
        return "SPLIT", "s1 %s / s2 %s disagree: s3 runs before reading" % (k1, k2)
    ks = [v["kind"] for v in done]; maj = [k for k in set(ks) if ks.count(k) >= 2]
    return (maj[0], "2-of-3 majority %s" % ks) if maj else ("seed-dependent", "no majority %s" % ks)


def main():
    b = boundary(); pl = plan(b["L_star"]) if b["status"] in ("fixed", "DIRECT did not train") else None   # untrainable DIRECT: no boundary
    print("BOUNDARY:", b["status"], "| L* =", b["L_star"], "| per seed:", b.get("per_seed"), "| uncertain:", b.get("boundary_uncertain"), "|", b["notes"], "| plan:", pl)
    print("DIRECT curves:", b.get("curves"))
    res = {"boundary": b, "plan": pl}
    arms = {"long": ("long64" if pl and pl["long_lmax"] == 64 else "long", pl["long_lmax"] if pl else 32)}
    if pl and pl["short_lmax"]:
        arms["short"] = ("short", pl["short_lmax"])
    kinds = {}
    for nm, (arm, lm) in arms.items():
        runs = {s: run(arm, s, lm) for s in (1, 2, 3)}
        k, why = arm_kind(runs); kinds[nm] = k
        res[nm] = {"dir_arm": arm, "l_max": lm, "runs": runs, "kind": k, "why": why}
        print(nm, arm, "l_max", lm, "|", k, "|", why, "|", {s: {kk: v.get(kk) for kk in ("status", "exact", "kind", "state_probe_p", "op_probe_old_mean", "edit",
                                                                                          "long_acc_512", "acc_2048", "serial_sanity", "fallback_trigger", "error")}
                                                             for s, v in runs.items() if v.get("status") != "open"})
    lk, sk = kinds.get("long"), kinds.get("short")
    unc = " (boundary uncertain: the lower L* is used)" if b.get("boundary_uncertain") else ""
    if b["status"] not in ("fixed", "DIRECT did not train") or lk is None or ("short" in arms and sk is None):
        rd = "pending"
    elif lk in ("SPLIT", "seed-dependent") or sk in ("SPLIT", "seed-dependent"):
        rd = "seed-dependent or awaiting s3: LONG %s, SHORT %s" % (lk, sk)
    elif lk is False:
        rd = "LONG not exact: the memory does not learn S5 at these lengths; the cell is uninformative on depth"
    elif pl and not pl["depth_reached"]:
        rd = "the depth condition is not reached (DIRECT passes to 64, or did not train): LONG %s, read for capacity only" % lk
    elif "short" not in arms:
        rd = "SHORT not run (L* = 2): LONG %s" % lk
    elif sk is False:
        rd = "SHORT not exact: LONG %s; no below-boundary comparison" % lk
    elif sk == "TOO SHORT":
        rd = "SHORT read descriptively (l_max < 4, too short to discriminate): LONG %s" % lk
    elif lk == "STATE-LIKE" and sk == "BUFFER-LIKE":
        rd = "158-2 supported: the same memory computes serially only where the task exceeds a pass"
    elif lk == "STATE-LIKE" and sk == "STATE-LIKE":
        rd = "both state-like: the task class decides, not the length (158-2's depth condition not needed on S5)"
    elif lk == "BUFFER-LIKE" and sk == "BUFFER-LIKE":
        rd = "both buffer-like: 158-2 refuted for this memory (buffer + read-time compute handles S5 beyond the single-pass boundary)"
    else:
        rd = "mixed: LONG %s, SHORT %s (reported with numbers)" % (lk, sk)
    res["reading"] = rd + unc; print("READING:", res["reading"])
    cap = []                                                                 # §162's capacity / depth (512) and capacity (2,048) predictions, per exact run
    for nm in arms:
        for s, v in res[nm]["runs"].items():
            if v.get("status") != "done" or not v["exact"] or v["kind"] not in ("STATE-LIKE", "BUFFER-LIKE"):
                continue
            c = {"run": "%s_s%d" % (res[nm]["dir_arm"], s), "kind": v["kind"], "long_acc_512": v["long_acc_512"], "acc_2048": v["acc_2048"]}
            if v["long_acc_512"] is not None:
                c["cap_depth_512_as_predicted"] = v["long_acc_512"] >= 0.90 if v["kind"] == "STATE-LIKE" else v["long_acc_512"] < 0.50
            if v["acc_2048"] is not None:
                c["cap_2048"] = (("as predicted" if v["acc_2048"] < 0.50 else "NOT as predicted (the buffer is not capacity-limited at 2,048)")
                                 if v["kind"] == "BUFFER-LIKE" else ("extrapolates" if v["acc_2048"] >= 0.90 else "drift (C6), not capacity"))
            cap.append(c)
    res["capacity"] = cap; print("CAPACITY:", cap)
    if ERRORS:
        print("ERRORS:", ERRORS)
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "score_s162.json"), "w", newline=chr(10)), indent=1, default=float)


if __name__ == "__main__":
    main()
