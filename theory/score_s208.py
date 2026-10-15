"""THEORY §208 scoring (QUESTION-ONCE + ANSWER-BOTTLENECK + NOPASS; registered text docs/THEORY_QONCE_DRAFT_2026-10-15.md at 1c2dc4d3 + notes 208-1 / 208-2;
Theory is the only scorer). Cell boxes/runs_fsq/gsm_lat_ansbn_qonce_nopass_s1: eval.json (on-host bf16), eval_qonce.json (local fp32, scratch/qonce_eval.py +
qonce_read.py), log.jsonl. Comparator A gsm_lat_aug_s1/eval.json. T = 0.028, T_D = 0.034 (deep = m >= 3).
ITER-GAIN (fp32, decisive; credited only if eval_qonce.g6.pass) = ACC - max(MEAN-ITER(1), K'1) >= T pooled AND >= T_D on m >= 3.
ONSET(m) = smallest k in {1, 2, 3, 4, 6, 8, 10} with MEAN-ITER(k)(m) >= 0.8 ACC(m) (k = 10 is clean), strata m = 1, 2, 3, 4, 5+ (n >= 50, ACC(m) >= 2/n).
SERIAL-PROFILE: ONSET non-decreasing over the resolvable strata AND min ONSET(4, 5+) >= max ONSET(1, 2) + 2 (note 208-2); < 3 resolvable strata -> UNREADABLE.
ONESHOT-PROFILE: one ONSET for every resolvable stratum. TRANSPLANT-SERIAL: P_hyb >= 0.30 at t = 2 under the better convention (C / S) and P_hyb > max(P_own, P_don),
n >= 100. INT-PROGRESS: probe Ml1 >= M0 + 0.10 and M1 <= M0 + 0.05. Labels QO-SERIAL / TWO-STAGE / PASS1-PRECOMPUTE / MEMORY-FAILS; QO-CAPACITY-BOUND /
QO-COMPUTE-FAIL beside. Censoring (rule b): log < 24k -> every reading fails; G6 failing -> ITER-GAIN not credited; NO-ENTRY if answer_ce never <= 0.92.
    python theory/score_s208.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "boxes", "runs_fsq")
CELL = "gsm_lat_ansbn_qonce_nopass_s1"
T, TD, EPS, FULL, RA = 0.028, 0.034, 1e-9, 24000, 0.42
KS = (1, 2, 3, 4, 6, 8, 10)


def jl(*p):
    f = os.path.join(R, *p)
    return json.load(open(f)) if os.path.exists(f) else None


def pooled(by_m, n_by_m, ms):
    ms = [m for m in ms if str(m) in by_m and str(m) in n_by_m]
    w = sum(n_by_m[str(m)] for m in ms)
    return sum(by_m[str(m)] * n_by_m[str(m)] for m in ms) / w if w else None


def strata(n_by_m):
    out = {}
    for m in (1, 2, 3, 4):
        if str(m) in n_by_m:
            out[str(m)] = [m]
    big = [int(k) for k in n_by_m if int(k) >= 5]
    if big:
        out["5+"] = big
    return out


def main():
    e, q, A = jl(CELL, "eval.json"), jl(CELL, "eval_qonce.json"), jl("gsm_lat_aug_s1", "eval.json"); out = {}
    if e is None or q is None:
        print("READING: pending (eval.json %s, eval_qonce.json %s)" % (e is not None, q is not None))
        json.dump({"reading": "pending"}, open(os.path.join(ROOT, "theory", "results", "score_s208.json"), "w", newline=chr(10))); return
    L = {}
    f = os.path.join(R, CELL, "log.jsonl")
    if os.path.exists(f):
        for line in open(f):
            r = json.loads(line)
            if "step" in r:
                L[int(r["step"])] = r
    last = max(L) if L else 0; dead = last < FULL; out["last_step"] = last; out["dead"] = dead
    ent = [s for s in sorted(L) if L[s].get("answer_ce") is not None and L[s]["answer_ce"] <= 0.92 + EPS]; out["entry_step"] = ent[0] if ent else None
    no_entry = not ent
    g6 = bool((q.get("g6") or {}).get("pass")); out["G6"] = g6
    cl = q["clean_fp32"]; nbm = cl["n_by_m"]; acc = cl["acc"]; kp = q["kprime_fp32"]; mi = q["mean_iter"]
    mi1, k1 = mi["1"]["acc"], kp["1"]["acc"]
    deep_ms = [int(m) for m in nbm if int(m) >= 3]
    acc_d = pooled(cl["by_m"], nbm, deep_ms); ctrl_d = max(pooled(mi["1"]["by_m"], nbm, deep_ms), pooled(kp["1"]["by_m"], nbm, deep_ms))
    ig, igd = acc - max(mi1, k1), acc_d - ctrl_d
    iter_gain = (not dead) and g6 and ig >= T - EPS and igd >= TD - EPS
    out.update(ACC=acc, MEAN_ITER1=mi1, K1=k1, ITER_GAIN_pooled=ig, ITER_GAIN_deep=igd, ITER_GAIN=iter_gain,
               K1_rescaled_reported=(e.get("bn") or {}).get("acc_k1_rescaled"))
    # ONSET by stratum
    S = strata(nbm); onset = {}; excl = []
    for name, ms in S.items():
        n = sum(nbm[str(m)] for m in ms); a = pooled(cl["by_m"], nbm, ms)
        if n < 50 or a is None or a < 2.0 / n:
            excl.append(name); continue
        o = None
        for k in KS:
            ak = a if k == 10 else (pooled(mi[str(k)]["by_m"], nbm, ms) if str(k) in mi else None)
            if ak is not None and ak >= 0.8 * a - EPS:
                o = k; break
        onset[name] = o if o is not None else 10
    order = [s for s in ("1", "2", "3", "4", "5+") if s in onset]; out.update(ONSET=onset, ONSET_excluded=excl)
    if len(order) < 3:
        serial_prof, oneshot = False, False; out["profile"] = "UNREADABLE (< 3 resolvable strata)"
    else:
        vals = [onset[s] for s in order]; nondec = all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))
        lo = [onset[s] for s in ("1", "2") if s in onset]; hi = [onset[s] for s in ("4", "5+") if s in onset]
        serial_prof = bool(nondec and lo and hi and min(hi) >= max(lo) + 2); oneshot = len(set(vals)) == 1
        out["profile"] = "SERIAL-PROFILE" if serial_prof else ("ONESHOT-PROFILE (ONSET %d)" % vals[0] if oneshot else "neither")
    # transplant
    tp = q.get("transplant") or {}; best = None
    for conv in ("C_t2", "S_t2"):
        x = tp.get(conv)
        if x and x.get("n", 0) >= 100 and (best is None or x["P_hyb"] > best[1]["P_hyb"]):
            best = (conv, x)
    trans = bool(best and best[1]["P_hyb"] >= 0.30 - EPS and best[1]["P_hyb"] > max(best[1]["P_own"], best[1]["P_don"]) + EPS)
    out["TRANSPLANT"] = {"best": None if best is None else best[0], "P_hyb": None if best is None else best[1]["P_hyb"], "serial": trans,
                         "readable": best is not None}
    ip = ((q.get("probes") or {}).get("int_progress")) or {}
    intp = bool(ip.get("Ml1") is not None and ip.get("M0") is not None and ip["Ml1"] >= ip["M0"] + 0.10 - EPS and ip.get("M1", 1) <= ip["M0"] + 0.05 + EPS)
    out["INT_PROGRESS"] = intp
    chance = kp["0"]["acc"]; mem_carries = acc >= chance + T - EPS; out["chance_K0"] = chance
    if dead:
        label = "FAILURE (log ends at %d < 24k; rule b)" % last
    elif no_entry:
        label = "NO-ENTRY (answer_ce never <= 0.92; rule b)"
    elif acc < chance + T - EPS:
        label = "MEMORY-FAILS"
    elif not g6:
        label = "ITER-GAIN NOT CREDITED (G6 failed; censoring) - no route label"
    elif iter_gain and serial_prof and (trans or intp):
        label = "QO-SERIAL"
    elif iter_gain and oneshot and order and onset[order[0]] >= 2:
        label = "TWO-STAGE"
    elif (not iter_gain) and mem_carries:
        label = "PASS1-PRECOMPUTE"
    else:
        label = "UNCLASSIFIED (ITER-GAIN %s, profile %s)" % (iter_gain, out["profile"])
    out["label"] = label
    # capacity
    rq = q.get("R_QO") or {}; lr = ((q.get("probes") or {}).get("lit_recall")) or {}
    cap = comp = None
    if rq.get("R_QO") is not None and lr.get("1") and lr.get("10"):
        r1, rK = lr["1"]["Mt"], lr["10"]["Mt"]; fl1 = max(lr["1"].get(k_, {}).get("all", 0) for k_ in ("M0", "length_only", "y_only"))
        fid = r1["nlit_ge4"] <= r1["nlit_le2"] - 0.10 + EPS; ret = rK["all"] <= r1["all"] - 0.10 + EPS
        cap = bool(rq["R_QO"] <= 0.5 * RA + EPS and (fid or ret))
        comp = bool(min(r1["nlit_le2"], r1["nlit_ge4"], rK["nlit_le2"], rK["nlit_ge4"]) >= fl1 + 0.30 - EPS and not iter_gain and A and acc <= A["acc"] - T + EPS)
        out.update(R_QO=rq["R_QO"], FIDELITY=fid, RETENTION=ret, fidelity_note="the difficulty control (A's aux-slot operand drop) is not in eval_qonce; FIDELITY read without it")
    out.update(QO_CAPACITY_BOUND=cap, QO_COMPUTE_FAIL=comp)
    if A:
        d = e["acc"] - A["acc"]; out["ACC_vs_A"] = "LIFTS" if d >= T - EPS else ("HURTS" if d <= -T + EPS else "MATCHES"); out["acc_vs_A_delta"] = d
    k10 = kp.get("10", {}).get("acc"); out["STABLE_12"] = None if k10 is None or "12" not in kp else kp["12"]["acc"] >= k10 - T - EPS
    out["STABLE_16"] = None if k10 is None or "16" not in kp else kp["16"]["acc"] >= k10 - T - EPS
    out["reading"] = "%s | ITER-GAIN %s (pooled %+.3f, deep %+.3f; ACC %.3f vs max(MEAN-ITER1 %.3f, K'1 %.3f); G6 %s) | %s, ONSET %s | transplant %s | INT-PROGRESS %s | CAP %s / COMPUTE-FAIL %s | ACC vs A %s" % (
        label, iter_gain, ig, igd, acc, mi1, k1, g6, out["profile"], onset, out["TRANSPLANT"], intp, cap, comp, out.get("ACC_vs_A"))
    print("READING:", out["reading"])
    json.dump(out, open(os.path.join(ROOT, "theory", "results", "score_s208.json"), "w", newline=chr(10)), indent=1, default=str)


if __name__ == "__main__":
    main()
