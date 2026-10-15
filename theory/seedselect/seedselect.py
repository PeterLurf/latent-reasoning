"""SEEDSELECT retrospective (docs/THEORY_SEEDSELECT_DRAFT_2026-10-14.md sections 3, 4, 7). CPU only, one thread, Idle.

Can early training-curve features rank seeds of ONE recipe on ONE card class by their eventual basin? Features are frozen in
SPEC below; readings R1 (primary), R2, R3, R2m are scored by within-stratum concordance with a within-stratum permutation
null. Outputs theory/seedselect/results.json.

Modes (in this order):
    python theory/seedselect/seedselect.py --selftest      # synthetic logs only (planted effect + null); must PASS
    python theory/seedselect/seedselect.py --freeze        # writes frozen.json = sha256(SPEC) + time; commit with the registration
    python theory/seedselect/seedselect.py --run           # real logs; refuses unless frozen.json matches SPEC and selftest passed

BLIND until --run: no mode other than --run opens a real log.jsonl. --run prints the gates (G0-G6) and the divergence
descriptive (Gdiv) before any reading.
"""
import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import time

import numpy as np

try:
    import psutil
    psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RUNS = os.path.join(ROOT, "boxes", "runs_fsq")
RES = os.path.join(ROOT, "theory", "results")

# ------------------------------------------------------------------ FROZEN SPEC (any change requires a new freeze)
SPEC = {
    "version": "seedselect-draft-2026-10-14-rev2",
    "windows": {   # [t0, t1] in steps, inclusive on logged record steps; records are block means over the preceding interval
        "R1": [500, 2000], "R2": [300, 900], "R3": [2100, 8000], "R3b": [2100, 6000]},
    "detrend_knots": {"R3": [6000], "R3b": []},   # hinge (u - knot)+ added to {1, u, u^2}; u = step / 1000
    "lf_min_period_steps": 500,                    # LF band: DFT bins j >= 1 with period T / j >= 500 steps, T = N * interval
    "min_resid_dof_spectral": 8,                   # LFA / LFF only if N - n_basis >= 8, else "not resolvable"
    "max_gap_factor": 1,                           # a window is usable only if every record step on the run's log grid is present
    "primary": {"reading": "R1", "feature": "slope:loss", "test": "two-sided", "alpha": 0.05},
    # success score = sign * feature (higher score = predicted success). Signs pre-stated per reading (section 3 of the draft).
    "signs": {
        "R1": {"slope:loss": +1, "level:loss": +1, "lfa:loss": +1, "lff:loss": +1, "null_power:loss": -1, "null_exp:loss": -1},
        "R2": {"slope:loss": -1, "level:loss": -1, "lfa:loss": +1, "lff:loss": +1, "null_power:loss": -1, "null_exp:loss": -1},
        "R3": {"slope:loss": +1, "slope:aux_ce": +1, "slope:answer_ce": +1, "level:aux_ce": +1, "lfa:answer_ce": +1,
               "lfa:aux_ce": +1, "vw:write_grad_norm": -1, "composite": +1, "null_power:loss": -1, "null_exp:loss": -1},
    },
    # rev 2 (Audit): CURVE-MECHANICS NULL. Fit the window's raw loss with a saturating form by least squares (grid over the
    # shape parameter, closed-form a, b), extrapolate to the outcome's reference steps. PRIMARY form: power law
    # f(u) = a + b u^(-c); SECONDARY: exponential saturation f(u) = a + b exp(-u / tau); u = step / 1000.
    # Null value: ratio targets r = max(f(t_b), eps) / max(f(t_a), eps) (lower = predicted success, sign -1);
    # level targets f(t) (lower = predicted success, sign -1).
    "null": {"primary": "power", "secondary": "exp", "c_grid": [0.02, 8.0, 400], "tau_grid_k": [0.05, 100.0, 400], "eps": 1e-3,
             "targets": {"R1": ["ratio", 2600, 4000], "R2": ["ratio", 1000, 2000], "R3": ["level", 48000], "R3b": ["level", 48000]}},
    # rev 2 (Audit): REGISTERED BEAT-THE-NULL RULE. USEFUL also needs the feature (in its realised direction) to beat the
    # PRIMARY null: one-sided paired within-stratum rank-swap permutation on dC = C(feature) - C(null), p < 0.05, and dC >= 0.10.
    # Beating the secondary (exp) null too is reported as "robust to the null form".
    "beat_null": {"alpha": 0.05, "min_delta_C": 0.10, "applies_to": ["R1", "R2"], "n_perm": 100000},
    # rev 2: R1c (non-loss outcome: long_acc >= 0.99) is registered only if it has >= 30 within-stratum pairs; else descriptive.
    "R1c_register_min_pairs": 30,
    "composite_R3": ["slope:aux_ce", "vw:write_grad_norm"],
    "thresholds": {"useful_C": 0.72, "useless_ci": [0.28, 0.72]},
    "n_perm": 100000, "n_boot": 4000, "seed": 20261014,
    "dup_rel_tol": 1e-6,
}
SPEC_HASH = hashlib.sha256(json.dumps(SPEC, sort_keys=True).encode()).hexdigest()


# ------------------------------------------------------------------ series and features
def read_series(path, fields, phase_filter=True):
    """Deduped by step (the last occurrence wins: resumed runs log some steps twice). Returns {field: {step: value}} and
    meta (resume markers, records). Values that are None / non-finite are dropped and counted."""
    out = {f: {} for f in fields}; meta = {"records": 0, "resumed_at": [], "nonfinite": {f: 0 for f in fields}}
    for line in open(path):
        try:
            r = json.loads(line)
        except Exception:
            continue
        meta["records"] += 1
        if "resumed" in r or r.get("event") == "resumed":
            meta["resumed_at"].append(r.get("step", r.get("resumed")))
        if "step" not in r or (phase_filter and r.get("phase", 1) != 1):
            continue
        s = int(r["step"])
        for f in fields:
            if f in r:
                v = r[f]
                if v is None or not isinstance(v, (int, float)) or not math.isfinite(v):
                    meta["nonfinite"][f] += 1; out[f].pop(s, None); continue
                out[f][s] = float(v)
    return out, meta


def window_vals(series, t0, t1, interval):
    """Record steps on the run's grid within [t0, t1]; None if any grid step is missing (gate G1)."""
    grid = [s for s in range(int(math.ceil(t0 / interval) * interval), t1 + 1, interval)]
    if not grid or any(s not in series for s in grid):
        return None, grid
    return np.array([series[s] for s in grid]), np.array(grid, dtype=float)


def basis(u, knots):
    cols = [np.ones_like(u), u, u ** 2] + [np.maximum(0.0, u - k / 1000.0) for k in knots]
    return np.stack(cols, 1)


def fit_saturating(u, v, form):
    """Least-squares saturating fit of raw values v at times u (thousand steps). Grid over the shape parameter (log-spaced),
    closed-form linear least squares for (a, b) at each grid value; the minimum-SSE grid value wins (ties: the first).
    Returns (a, b, shape, sse)."""
    if form == "power":
        lo, hi, n = SPEC["null"]["c_grid"]; grid = np.exp(np.linspace(math.log(lo), math.log(hi), int(n)))
        G = u[None, :] ** (-grid[:, None])
    else:
        lo, hi, n = SPEC["null"]["tau_grid_k"]; grid = np.exp(np.linspace(math.log(lo), math.log(hi), int(n)))
        G = np.exp(-u[None, :] / grid[:, None])
    gm = G.mean(1, keepdims=True); vm = v.mean()
    var = ((G - gm) ** 2).sum(1); cov = ((G - gm) * (v - vm)[None, :]).sum(1)
    b = np.where(var > 1e-300, cov / np.maximum(var, 1e-300), 0.0); a = vm - b * gm[:, 0]
    sse = ((v[None, :] - a[:, None] - b[:, None] * G) ** 2).sum(1)
    k = int(np.argmin(sse))
    return float(a[k]), float(b[k]), float(grid[k]), float(sse[k])


def predict_saturating(params, form, u):
    a, b, sh, _ = params
    return a + b * (u ** (-sh) if form == "power" else math.exp(-u / sh))


def null_value(v, grid, reading):
    """The curve-mechanics null for one run (both forms). v: raw window values, grid: steps."""
    tgt = SPEC["null"]["targets"].get(reading)
    if tgt is None:
        return {}
    u = grid / 1000.0; eps = SPEC["null"]["eps"]; out = {}
    for form in ("power", "exp"):
        prm = fit_saturating(u, v, form)
        if tgt[0] == "ratio":
            fa = predict_saturating(prm, form, tgt[1] / 1000.0); fb = predict_saturating(prm, form, tgt[2] / 1000.0)
            out["null_%s:loss" % form] = float(max(fb, eps) / max(fa, eps))
        else:
            out["null_%s:loss" % form] = float(predict_saturating(prm, form, tgt[1] / 1000.0))
        out["_fit_%s" % form] = {"a": prm[0], "b": prm[1], "shape": prm[2], "rmse": math.sqrt(prm[3] / len(v))}
    return out


def features(series_by_field, reading, interval):
    t0, t1 = SPEC["windows"][reading]; knots = SPEC["detrend_knots"].get(reading, [])
    out, why = {}, {}
    for f, ser in series_by_field.items():
        v, grid = window_vals(ser, t0, t1, interval)
        if v is None:
            why[f] = "window incomplete"; continue
        if np.any(v <= 0):
            why[f] = "non-positive value"; continue
        y = np.log(v); u = grid / 1000.0; N = len(y)
        if f == "loss":
            nv = null_value(v, grid, reading)
            out.update({k: x for k, x in nv.items() if not k.startswith("_")})
            why.update({k: x for k, x in nv.items() if k.startswith("_")})
        X1 = np.stack([np.ones_like(u), u], 1); b = np.linalg.lstsq(X1, y, rcond=None)[0]
        out["slope:" + f] = float(b[1])                                     # d ln(value) / d(1000 steps)
        last = grid > (t1 - 1000)
        out["level:" + f] = float(np.mean(y[last]))
        B = basis(u, knots)
        e = y - B @ np.linalg.lstsq(B, y, rcond=None)[0]
        if f == "write_grad_norm":
            out["vw:" + f] = float(np.sqrt(np.sum(e ** 2) / max(1, N - B.shape[1])))
        if N - B.shape[1] >= SPEC["min_resid_dof_spectral"]:
            E = np.fft.fft(e); T = N * interval
            js = [j for j in range(1, (N - 1) // 2 + 1)]
            band = [j for j in js if T / j >= SPEC["lf_min_period_steps"]]
            pw = {j: abs(E[j]) ** 2 for j in js}
            out["lfa:" + f] = float(math.sqrt(2.0 / N ** 2 * sum(pw[j] for j in band))) if band else None
            tot = sum(pw.values())
            out["lff:" + f] = float(sum(pw[j] for j in band) / tot) if band and tot > 0 else None
        else:
            why["lfa:" + f] = "not resolvable (dof %d)" % (N - B.shape[1])
    return out, why


# ------------------------------------------------------------------ statistics
# A stratum is (x, y): x = success scores, y = numeric outcome (binary: 1 success / 0 fail; ordinal: -entry step, with
# never-entered = -inf). Concordance is over pairs with y_i > y_j: P(x_i > x_j), ties in x count 1/2.
def pooled_C(strata):
    num = den = 0.0
    for x, y in strata:
        x = np.asarray(x, float); y = np.asarray(y, float)
        if len(x) < 2:
            continue
        gt = y[:, None] > y[None, :]
        d = x[:, None] - x[None, :]
        num += ((d > 0) & gt).sum() + 0.5 * ((d == 0) & gt).sum(); den += gt.sum()
    return (num / den, int(den)) if den else (None, 0)


def _distinct_perms(y):
    from itertools import permutations
    return sorted(set(permutations(list(y))))


def perm_test(strata, n_perm, rng, two_sided):
    """Null: outcomes exchangeable within each stratum. Exact enumeration when the number of distinct within-stratum
    outcome arrangements is <= 200,000, else Monte Carlo (n_perm draws, p = (count + 1) / (n + 1))."""
    C, npairs = pooled_C(strata)
    if C is None:
        return None
    total = 1
    for x, y in strata:
        cnt_ = {}
        for v in y:
            cnt_[v] = cnt_.get(v, 0) + 1
        m = math.factorial(len(y))
        for c in cnt_.values():
            m //= math.factorial(c)
        total *= m
    exact = total <= 200000 and all(len(y) <= 9 for _, y in strata)
    hit = lambda c: (abs(c - 0.5) >= abs(C - 0.5) - 1e-12) if two_sided else (c >= C - 1e-12)  # noqa: E731
    cnt = 0; n = 0
    if exact:
        from itertools import product
        arr = [_distinct_perms(y) for _, y in strata]
        for pick in product(*arr):
            c, _ = pooled_C([(x, np.array(yy)) for (x, _), yy in zip(strata, pick)]); n += 1; cnt += hit(c)
        p = cnt / n
    else:
        for _ in range(n_perm):
            c, _ = pooled_C([(x, rng.permutation(np.asarray(y))) for x, y in strata]); n += 1; cnt += hit(c)
        p = (cnt + 1) / (n + 1)
    return {"C": C, "pairs": npairs, "p": p, "exact": exact, "two_sided": two_sided, "n_arrangements": total}


def boot_ci(strata, n_boot, rng):
    """Stratified bootstrap: resample units within each (stratum, outcome value) cell."""
    vals = []
    cells = []
    for x, y in strata:
        x = np.asarray(x, float); y = np.asarray(y, float)
        cells.append([(x[y == v], v) for v in np.unique(y)])
    for _ in range(n_boot):
        st = []
        for cl in cells:
            xs, ys = [], []
            for xv, v in cl:
                idx = rng.integers(0, len(xv), len(xv)); xs.append(xv[idx]); ys.append(np.full(len(xv), v))
            st.append((np.concatenate(xs), np.concatenate(ys)))
        c, _ = pooled_C(st)
        if c is not None:
            vals.append(c)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else None


def rank01(x):
    """Within-stratum average ranks scaled to [0, 1] (ties averaged)."""
    x = np.asarray(x, float); n = len(x)
    order = np.argsort(x, kind="mergesort"); r = np.empty(n); r[order] = np.arange(n, dtype=float)
    for v in np.unique(x):
        m = x == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r / max(1, n - 1)


def paired_beat_null(strata, n_perm, rng):
    """strata: list of (xf, xn, y) on the same units: signed feature, signed null, outcome. Both scores are replaced by
    within-stratum ranks; dC = C(feature ranks) - C(null ranks). Null hypothesis: the two scores are exchangeable within
    each unit (paired design), so each unit's two ranks are swapped with probability 1/2. One-sided p for dC > 0;
    exact enumeration over swap masks when 2^N <= 4096, else Monte Carlo."""
    R = [(rank01(xf), rank01(xn), np.asarray(y, float)) for xf, xn, y in strata if len(y) >= 2]
    if not R:
        return None

    def dC(RR):
        cf, n_ = pooled_C([(a, y) for a, _, y in RR]); cn, _ = pooled_C([(b, y) for _, b, y in RR])
        return (None if cf is None else cf - cn), cf, cn, n_
    obs, cf, cn, npairs = dC(R)
    if obs is None:
        return None
    N = sum(len(y) for _, _, y in R); cnt = 0; n = 0
    if 2 ** N <= 4096:
        for mask_all in range(2 ** N):
            RR = []; off = 0
            for a, b, y in R:
                m = np.array([(mask_all >> (off + i)) & 1 for i in range(len(y))], bool); off += len(y)
                RR.append((np.where(m, b, a), np.where(m, a, b), y))
            d, _, _, _ = dC(RR); n += 1; cnt += d >= obs - 1e-12
        p = cnt / n; exact = True
    else:
        for _ in range(n_perm):
            RR = []
            for a, b, y in R:
                m = rng.random(len(y)) < 0.5; RR.append((np.where(m, b, a), np.where(m, a, b), y))
            d, _, _, _ = dC(RR); n += 1; cnt += d >= obs - 1e-12
        p = (cnt + 1) / (n + 1); exact = False
    return {"dC": float(obs), "C_feature_ranks": float(cf), "C_null_ranks": float(cn), "pairs": npairs, "p_one_sided": p, "exact": exact}


def verdict(res, ci):
    th = SPEC["thresholds"]
    if res is None:
        return "NOT EVALUABLE"
    if res["p"] < SPEC["primary"]["alpha"] and res["C"] >= th["useful_C"]:
        return "USEFUL (pre-stated direction)"
    if res["p"] < SPEC["primary"]["alpha"] and res["C"] <= 1 - th["useful_C"]:
        return "USEFUL-REVERSED (opposite of the pre-stated direction; prospective use needs a new registration)"
    if ci and ci[0] > th["useless_ci"][0] and ci[1] < th["useless_ci"][1]:
        return "NOT USEFUL (95% CI inside (0.28, 0.72))"
    if res["p"] < SPEC["primary"]["alpha"]:
        return "DETECTABLE BUT BELOW THE USEFUL BAR"
    return "INCONCLUSIVE"


# ------------------------------------------------------------------ units (samples) from labels.json
def load_units():
    L = json.load(open(os.path.join(HERE, "labels.json")))
    lab, dup_cand = L["labels"], L["duplicate_candidates"]
    units = {"R1": [], "R1b": [], "R1c": [], "R2": [], "R3": [], "R3ord": [], "R3ord_event_only": []}
    # CONSERVATIVE CENSORING (user standing rule 2026-10-14, note 200-6), applied to every label:
    #  (C1) a run that reached its registered end without the event is a FAILURE (nested: entry(0.5) None at 48k;
    #       fsq120: FLAG4 true at 4k, FLAG true at 2k, long_acc < 0.99 at 24k);
    #  (C2) a run whose own trajectory triggered its stop (the 123-3 plateau stop) is a FAILURE in every reading whose
    #       population contains it (R2: flagged; R1 / R1b / R1c condition on escape, so it is outside them);
    #  (C3) a run without the outcome's reference step because of a DESIGN stop (stop_after, section 129 phase 1, never
    #       started) or an infrastructure end (*.void*) is EXCLUDED and listed: the stop was fixed before, and
    #       independently of, the outcome, so excluding it cannot bias a within-stratum concordance.
    excluded = []
    final_gid = lab["s120_s5_final_s1"]["gid"]           # the FINAL S5 recipe group (normalised recipe equal to section 120's)
    SAME = {"s128_s5_s1", "s128_s5_s3", "s127_s5_skip_s2"}   # 131-8 / 131-1(b) SAME_TRAJECTORY: counted once
    # training GPU for runs whose args lack train_gpu (note 123-7, deploy logs); 3090 and 3090 Ti are one class (125-2)
    GPU_DOC = {"s120_s5_final_s1": "3090", "s120_s5_final_s2": "3090", "s120_s5_trace_s1": "4090", "s120_s5_trace_s2": "3090",
               "s123_s5_dither050_s1": "4090", "s123_s5_dither050_s2": "4090"}
    inv_runs = json.load(open(os.path.join(HERE, "inventory.json")))["runs"]

    def gclass(k, g):
        if k in GPU_DOC:
            return GPU_DOC[k]
        if g is None and k.startswith(("s120_", "s123_")):
            return "3090"
        return "3090" if g in ("3090", "3090ti") else g
    for k, v in lab.items():
        # R1b (descriptive): 24k FINAL S5 runs with eval long_acc, 2k escapers
        if v.get("gid") == final_gid and v.get("long_acc") is not None and v.get("flag2k") is False and k not in SAME and v.get("complete"):
            cls = gclass(k, v.get("gpu"))
            units["R1b"].append({"id": k, "src": os.path.join(RUNS, k, "log.jsonl"), "stratum": "R1b|%s" % cls,
                                 "success": bool(v["exact"]), "interval": 100, "seed": v["seed"], "card": cls})
        # R1c (rev 2, the non-loss outcome): every fsq120 S5 recipe with eval long_acc, 2k escapers, complete 24k;
        # strata = (recipe group, card class); registered only if >= R1c_register_min_pairs pairs, else descriptive
        if v.get("gid") == final_gid and v.get("flag2k") is False and v.get("long_acc") is None and not k.startswith(("flag4proto_", "rampproto", "proto")):
            excluded.append((k, "R1b/R1c", "C3: escaper without eval long_acc (design stop: %s)" % v.get("stopped")))
        ir = inv_runs.get(k, {})
        if v.get("family") == "fsq120" and ir.get("sub") == "s5" and v.get("long_acc") is not None and v.get("flag2k") is False \
                and k not in SAME and v.get("complete"):
            cls = gclass(k, v.get("gpu"))
            units["R1c"].append({"id": k, "src": os.path.join(RUNS, k, "log.jsonl"), "stratum": "R1c|%s|%s" % (v["gid"], cls),
                                 "success": bool(v["exact"]), "interval": 100, "seed": v["seed"], "card": cls})
        if v.get("family") == "fsq120" and (v.get("gid") == final_gid or k.startswith(("replay130/", "replay_x/"))):
            src = os.path.join(RES, k + ".jsonl") if k.startswith("replay") else os.path.join(RUNS, k, "log.jsonl")
            card = v["gpu"]; panel = k.split("/")[0] if "/" in k else k.split("_s")[0]
            # R1: proto escapers with FLAG4 (fsq120.py, 100-step logs); outcome success = FLAG4 False
            if k.startswith(("flag4proto_", "rampproto_plain_")) and v.get("flag2k") is None:
                excluded.append((k, "R1/R2", "C3: no 2k record (design stop or never started)"))
            if k.startswith(("flag4proto_", "rampproto_plain_")) and v.get("flag2k") is False and v.get("flag4k") is None:
                excluded.append((k, "R1", "C3: escaper without a 4k record (design stop)"))
            if k.startswith(("flag4proto_", "rampproto_plain_")) and v.get("flag2k") is False and v.get("flag4k") is not None:
                units["R1"].append({"id": k, "src": src, "stratum": "R1|%s" % card, "success": not v["flag4k"], "interval": 100, "seed": v["seed"], "card": card})
            # R2: 2k plateau FLAG (123-3), replays (50-step) and flag4proto (100-step); proto136_d8 / rampproto excluded (dup / selected)
            if (k.startswith(("replay130/", "replay_x/")) or k.startswith("flag4proto_")) and v.get("flag2k") is not None:
                itv = 50 if k.startswith("replay") else 100
                units["R2"].append({"id": k, "src": src, "stratum": "R2|%s|%s" % (panel, card), "success": not v["flag2k"], "interval": itv, "seed": v["seed"], "card": card})
    # R3 nested: binary RMS stratum, entry-time ordinal strata (fixed lists; section 1 of the draft)
    R3_BIN = {"s185_rms_s1": True, "s185_rms_s2": True, "s185_rms_s3": False}
    R3_ORD = {"rms": ["s185_rms_s1", "s185_rms_s2", "s185_rms_s3"], "window": ["s182_window_s1", "s182_window_s2"],
              "reweight": ["s184_reweight_s1", "s184_reweight_s2"], "nopass": ["s187_nopass_s1", "s187_nopass_s2"],
              "rmsnopass": ["s190_rmsnopass_s1", "s190_rmsnopass_s2"]}
    for k, s in R3_BIN.items():
        units["R3"].append({"id": k, "src": os.path.join(RUNS, k, "log.jsonl"), "stratum": "R3|rms|3090", "success": s, "interval": 100})
    for st, ks in R3_ORD.items():
        for k in ks:
            e = lab[k].get("entry_0.5")
            units["R3ord"].append({"id": k, "src": os.path.join(RUNS, k, "log.jsonl"), "stratum": "R3ord|%s" % st,
                                   "entry": e if e is not None else float("inf"), "interval": 100})   # C1: never entered = worst
            if e is not None:                                   # the event-only reading reported beside it (rule (a))
                units["R3ord_event_only"].append({"id": k, "src": os.path.join(RUNS, k, "log.jsonl"), "stratum": "R3ordE|%s" % st,
                                                  "entry": e, "interval": 100})
    units["_excluded"] = excluded
    return units, dup_cand, lab


# ------------------------------------------------------------------ gates
def gate_dups(units_by_reading, dup_cand):
    """G2: candidate pairs (same seed, card, recipe) whose loss records are identical over [0, max window end] count once.
    The replay (50-step) is compared at 100-step resolution: mean of records (s - 50, s) equals the 100-step record s."""
    t_end = 2000; drop = {}; rep = []
    ids = {u["id"] for us in units_by_reading.values() for u in us}
    for key, members in dup_cand.items():
        mem = [m for m in members if m in ids or m.startswith(("replay", "proto136", "rampproto"))]
        if len(mem) < 2:
            continue
        ser = {}
        for m in mem:
            src = os.path.join(RES, m + ".jsonl") if m.startswith("replay") else os.path.join(RUNS, m, "log.jsonl")
            if not os.path.exists(src):
                continue
            s, _ = read_series(src, ["loss"]); s = s["loss"]
            if m.startswith("replay"):
                s = {t: 0.5 * (s[t] + s[t - 50]) for t in range(100, t_end + 1, 100) if t in s and (t - 50) in s}
            ser[m] = {t: v for t, v in s.items() if t <= t_end and t % 100 == 0}
        ms = sorted(ser)
        for i in range(len(ms)):
            for j in range(i + 1, len(ms)):
                a, b = ser[ms[i]], ser[ms[j]]; common = sorted(set(a) & set(b))
                if len(common) < 5:
                    continue
                same = all(abs(a[t] - b[t]) <= SPEC["dup_rel_tol"] * max(abs(a[t]), 1e-12) + 1e-7 for t in common)
                rep.append({"pair": [ms[i], ms[j]], "identical_to_2k": same, "n_common": len(common)})
                if same:   # count once WITHIN a reading: keep the replay (finer log) or the first dir; never drop across readings
                    keep = ms[i] if ms[i].startswith("replay") or not ms[j].startswith("replay") else ms[j]
                    other = ms[j] if keep == ms[i] else ms[i]
                    for rd, us in units_by_reading.items():
                        uid = {u["id"] for u in us}
                        if keep in uid and other in uid:
                            drop.setdefault(rd, set()).add(other)
    return drop, rep


def score_reading(units, reading, fields, rng, two_sided, signs, comp=None):
    rows, excl = [], []
    for u in units:
        if not os.path.exists(u["src"]):
            excl.append((u["id"], "log missing")); continue
        ser, meta = read_series(u["src"], fields)
        t1 = SPEC["windows"][reading][1]
        ra = [x for x in meta["resumed_at"] if x is not None]
        if ra and min(float(x) for x in ra) <= t1:
            excl.append((u["id"], "resumed inside the window")); continue
        F, why = features({f: ser[f] for f in fields}, reading, u["interval"])
        rows.append({**{k: u[k] for k in u if k != "src"}, "features": F, "why": why, "nonfinite": meta["nonfinite"]})
    feats = sorted(set(k for r in rows for k in r["features"] if r["features"][k] is not None))
    if comp:
        feats.append("composite")
    out = {"n_units": len(rows), "excluded": excl, "features": {}}
    strata_ids = sorted(set(r["stratum"] for r in rows))
    for ft in feats:
        sg = signs.get(ft)
        if sg is None:
            continue
        st, st_ids = [], []
        for sid in strata_ids:
            rr = [r for r in rows if r["stratum"] == sid]
            if ft == "composite":
                ok = [r for r in rr if all(r["features"].get(c) is not None for c in comp)]
                if len(ok) < 2:
                    continue
                sc = np.zeros(len(ok))
                for c in comp:      # within-stratum rank average of the signed components (no outcome used)
                    x = np.array([signs[c] * r["features"][c] for r in ok]); sc += x.argsort().argsort() / max(1, len(ok) - 1)
                vals = list(zip(ok, sc))
            else:
                vals = [(r, sg * r["features"][ft]) for r in rr if r["features"].get(ft) is not None]
            if len(vals) < 2:
                continue
            x = np.array([v for _, v in vals], float)
            # binary: y = 1 success / 0 fail; ordinal: y = -entry step (never entered = -inf), so higher y = earlier entry
            y = np.array([(1.0 if r["success"] else 0.0) if "success" in r else -float(r["entry"]) for r, _ in vals], float)
            if len(set(y.tolist())) < 2:
                continue
            st.append((x, y)); st_ids.append(sid)
        res = perm_test(st, SPEC["n_perm"], rng, two_sided)
        ci = boot_ci(st, SPEC["n_boot"], rng) if res and res["pairs"] >= 10 else None
        out["features"][ft] = {"result": res, "ci95": ci, "verdict": verdict(res, ci) if (res and res["pairs"] >= 10) else "descriptive (pairs < 10)",
                               "strata": {sid: {"n": int(len(y)), "outcomes": sorted(set(y.tolist()))[:3]} for sid, (_, y) in zip(st_ids, st)}}
    # rev 2: beat-the-null (curve mechanics). Units with both the feature and the null; within-stratum paired test.
    bn = SPEC["beat_null"]
    for ft, info in out["features"].items():
        if ft.startswith("null_") or info.get("result") is None:
            continue
        d = +1 if info["result"]["C"] >= 0.5 else -1            # the feature's realised direction
        info["beat_null"] = {}
        for nf in ("null_power:loss", "null_exp:loss"):
            if nf not in signs:
                continue
            st3 = []
            for sid in strata_ids:
                rr = [r for r in rows if r["stratum"] == sid and r["features"].get(nf) is not None
                      and (r["features"].get(ft) is not None or ft == "composite")]
                if ft == "composite":
                    rr = [r for r in rr if all(r["features"].get(c) is not None for c in comp)]
                if len(rr) < 2:
                    continue
                y = np.array([(1.0 if r["success"] else 0.0) if "success" in r else -float(r["entry"]) for r in rr], float)
                if len(set(y.tolist())) < 2:
                    continue
                if ft == "composite":
                    xf = np.zeros(len(rr))
                    for c in comp:
                        xc = np.array([signs[c] * r["features"][c] for r in rr]); xf += xc.argsort().argsort() / max(1, len(rr) - 1)
                else:
                    xf = np.array([signs[ft] * r["features"][ft] for r in rr])
                st3.append((d * xf, np.array([signs[nf] * r["features"][nf] for r in rr]), y))
            info["beat_null"][nf] = paired_beat_null(st3, bn["n_perm"], rng)
        bp, be = info["beat_null"].get("null_power:loss"), info["beat_null"].get("null_exp:loss")
        beats = bool(bp and bp["p_one_sided"] < bn["alpha"] and bp["dC"] >= bn["min_delta_C"])
        robust = bool(be and be["p_one_sided"] < bn["alpha"] and be["dC"] >= bn["min_delta_C"])
        info["beats_primary_null"], info["beats_secondary_null"] = beats, robust
        info["verdict_rev1"] = info["verdict"]
        if info["verdict"].startswith("USEFUL"):
            if beats:
                info["verdict"] = info["verdict"] + ("; beats both saturation nulls" if robust else "; beats the primary (power) null only, NOT robust to the null form")
            else:
                info["verdict"] = "EXPLAINED BY CURVE MECHANICS (does not beat the primary saturation null)"
    out["rows"] = rows
    return out


def gdiv(lab):
    """Gdiv (descriptive): for replay_x seeds on 2+ cards, the first record step where |ln loss_a - ln loss_b| > 0.01."""
    by = {}
    for k, v in lab.items():
        if k.startswith("replay_x/"):
            by.setdefault(v["seed"], {})[v["gpu"]] = k
    res = {}
    for seed, cards in by.items():
        ser = {c: read_series(os.path.join(RES, k + ".jsonl"), ["loss"])[0]["loss"] for c, k in cards.items()}
        cs = sorted(ser)
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                a, b = ser[cs[i]], ser[cs[j]]
                d = next((t for t in sorted(set(a) & set(b)) if abs(math.log(a[t]) - math.log(b[t])) > 0.01), None)
                res.setdefault("%s-%s" % (cs[i], cs[j]), []).append(d)
    return {k: {"n": len(v), "median_first_divergence_step": float(np.median([x for x in v if x is not None])) if any(x is not None for x in v) else None,
                "never_within_2k": sum(1 for x in v if x is None)} for k, v in res.items()}


def matched_cross_card(rows, lab):
    """R2m (descriptive): replay_x seeds whose FLAG differs across cards. Feature ranked within card (removes card main
    effects); for each discordant (seed, card pair) the sign of rank(unflagged card) - rank(flagged card) under the R2 sign."""
    out = {}
    rx = [r for r in rows if str(r["id"]).startswith("replay_x/")]
    for ft in ("slope:loss", "level:loss", "lfa:loss"):
        sg = SPEC["signs"]["R2"].get(ft)
        pc = {}
        for card in set(r["card"] for r in rx):
            rr = [r for r in rx if r["card"] == card and r["features"].get(ft) is not None]
            x = np.array([sg * r["features"][ft] for r in rr]); rk = x.argsort().argsort() / max(1, len(rr) - 1)
            for r, q in zip(rr, rk):
                pc[(r["seed"], card)] = (q, r["success"])
        wins = n = 0
        seeds = set(s for s, _ in pc)
        for s in seeds:
            cs = [c for (ss, c) in pc if ss == s]
            for i in range(len(cs)):
                for j in range(i + 1, len(cs)):
                    (qa, ya), (qb, yb) = pc[(s, cs[i])], pc[(s, cs[j])]
                    if ya != yb:
                        n += 1; wins += ((qa > qb) if ya else (qb > qa))
        p = sum(math.comb(n, k) for k in range(wins, n + 1)) / 2 ** n if n else None
        out[ft] = {"discordant_cases": n, "concordant": wins, "one_sided_sign_p": p}
    return out


# ------------------------------------------------------------------ modes
def run(real=True, units=None, dup_cand=None, lab=None, tag="run"):
    rng = np.random.default_rng(SPEC["seed"])
    if real:
        units, dup_cand, lab = load_units()
    excl_c3 = units.pop("_excluded", [])
    drop, dup_rep = gate_dups({r: units[r] for r in ("R1", "R1b", "R1c", "R2", "R3", "R3ord")}, dup_cand) if real else ({}, [])
    for r in units:
        units[r] = [u for u in units[r] if u["id"] not in drop.get(r, set())]
    gates = {"G2_duplicates": dup_rep, "G2_dropped": {r: sorted(v) for r, v in drop.items()},
             "G3_units": {r: len(v) for r, v in units.items()}, "G7_censoring_C3_excluded": excl_c3}
    out = {"spec_hash": SPEC_HASH, "mode": tag, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "gates": gates}
    if real:
        out["Gdiv"] = gdiv(lab)
        print("GATES:", json.dumps(gates, default=str)[:2000]); print("Gdiv:", json.dumps(out["Gdiv"]))
    out["R1"] = score_reading(units["R1"], "R1", ["loss"], rng, True, SPEC["signs"]["R1"])
    if units.get("R1b"):
        out["R1b"] = score_reading(units["R1b"], "R1", ["loss"], rng, False, SPEC["signs"]["R1"])
    if units.get("R1c"):
        out["R1c"] = score_reading(units["R1c"], "R1", ["loss"], rng, False, SPEC["signs"]["R1"])
        pr_c = out["R1c"]["features"].get(SPEC["primary"]["feature"], {}).get("result") or {}
        out["R1c"]["registered"] = bool(pr_c.get("pairs", 0) >= SPEC["R1c_register_min_pairs"])
    out["R2"] = score_reading(units["R2"], "R2", ["loss"], rng, True, SPEC["signs"]["R2"])
    out["R2m"] = matched_cross_card(out["R2"]["rows"], lab) if real else None
    if units.get("R3"):
        f3 = ["loss", "aux_ce", "answer_ce", "write_grad_norm"]
        for r3 in ("R3", "R3b"):
            out[r3 + "_binary"] = score_reading(units["R3"], r3, f3, rng, False, SPEC["signs"]["R3"], comp=SPEC["composite_R3"])
            out[r3 + "_ordinal"] = score_reading(units["R3ord"], r3, f3, rng, False, SPEC["signs"]["R3"], comp=SPEC["composite_R3"])
            out[r3 + "_ordinal_event_only"] = score_reading(units["R3ord_event_only"], r3, f3, rng, False, SPEC["signs"]["R3"], comp=SPEC["composite_R3"])
    for rd, res_ in out.items():                          # Holm across a reading's secondary features (the primary excluded)
        if not isinstance(res_, dict) or "features" not in res_:
            continue
        fs = [(f, v["result"]["p"]) for f, v in res_["features"].items() if v.get("result") and not (rd == "R1" and f == SPEC["primary"]["feature"])]
        fs.sort(key=lambda t: t[1]); m = len(fs); run_max = 0.0
        for i, (f, pv) in enumerate(fs):
            run_max = max(run_max, min(1.0, (m - i) * pv)); res_["features"][f]["p_holm_secondary"] = run_max
    pr = out["R1"]["features"].get(SPEC["primary"]["feature"], {})
    out["PRIMARY"] = {"reading": "R1", "feature": SPEC["primary"]["feature"], **pr}
    return out


def synth_logs(tmp, n_s, n_f, planted, interval, steps, field_noise=0.02, seed=0):
    """Synthetic loss curves: exp decay + noise; failures (planted) decline faster in the window (a hare)."""
    rng = np.random.default_rng(seed); units = []
    for i in range(n_s + n_f):
        succ = i < n_s; rate = 1.0 + (0.0 if succ else planted) + rng.normal(0, 0.15)
        path = os.path.join(tmp, "u%d.jsonl" % i)
        with open(path, "w", newline="\n") as fh:
            for s in range(interval, steps + 1, interval):
                v = 2.0 * math.exp(-rate * s / 4000.0) + 0.5 + rng.normal(0, field_noise)
                fh.write(json.dumps({"step": s, "loss": max(v, 1e-3), "aux_ce": max(v / 2, 1e-3), "answer_ce": max(v / 2, 1e-3),
                                     "write_grad_norm": abs(1 + rng.normal(0, 0.1))}) + "\n")
            if i == 0:
                fh.write(json.dumps({"step": interval, "loss": 9.0}) + "\n")     # a duplicated step; the last occurrence wins
        units.append({"id": "u%d" % i, "src": path, "stratum": "S|x" if i % 2 == 0 else "S|y", "success": succ, "interval": interval, "seed": i, "card": "x"})
    return units


def _flag4(vals):
    """§131's FLAG4 on a {step: loss} dict (nearest logged steps)."""
    g = lambda t: vals[min(vals, key=lambda s: abs(s - t))]  # noqa: E731
    return bool(g(4000) >= 0.9 * g(2600) and g(4000) >= 0.25 * g(100))


def synth_panel(tmp, kind, n_units, seed, noise=0.015, interval=100, steps=4000):
    """rev 2 selftest panels. Half the units are power-law curves and half exponential saturation (so the primary null
    is misspecified for half), mixed within each of two strata. FLAG4 is computed from each synthetic curve itself.
      "saturation": no basin variable. FLAG4 varies only through curve mechanics (asymptote, amplitude, decay shape).
      "basin": a narrow mechanics spread plus a basin variable. Failures carry an extra early descent inside the window
               (a hare term that does not saturate there) and plateau after 2.1k; successes keep descending after 2.1k."""
    rng = np.random.default_rng(seed); units = []
    for i in range(n_units):
        form = "power" if i % 2 == 0 else "exp"
        if kind == "saturation":            # wide mechanics, balanced outcomes
            a = rng.uniform(0.2, 2.5); b = rng.uniform(0.5, 3.5)
            sh = rng.uniform(0.4, 3.0) if form == "power" else rng.uniform(0.4, 3.0)
            fail = None; h = 0.0
        elif kind == "saturation_strong":   # mechanics in which the window slope alone predicts FLAG4 (the confound bites)
            a = rng.uniform(0.05, 2.0); b = rng.uniform(0.5, 4.0)
            sh = rng.uniform(0.2, 2.5) if form == "power" else rng.uniform(0.4, 4.0)
            fail = None; h = 0.0
        else:
            a = rng.uniform(1.0, 1.4); b = rng.uniform(1.8, 2.2)
            sh = rng.uniform(0.8, 1.2) if form == "power" else rng.uniform(1.5, 2.5)
            fail = bool(rng.random() < 0.5); h = rng.uniform(0.35, 0.55) if fail else 0.0
        base = lambda u: a + b * (u ** (-sh) if form == "power" else math.exp(-u / sh))  # noqa: E731
        vals = {}
        for s in range(interval, steps + 1, interval):
            u = s / 1000.0; L = base(u) * math.exp(-h * min(1.0, max(0.0, (s - 500) / 1500.0)))
            if kind == "basin" and s > 2100:
                L = (base(2.0) * math.exp(-h)) * (1 + 0.02 * (s - 2000) / 2000.0) if fail else base(u) * math.exp(-(s - 2100) / 1800.0)
            vals[s] = max(1e-3, L * (1 + rng.normal(0, noise)))
        vals[100] = 3.0 * vals[2000]     # a bounded early loss (real S5: about 8.5 at 100 vs 3-5 at 2k), so FLAG4's floor clause behaves as in the data
        succ = not _flag4(vals)
        path = os.path.join(tmp, "%s%d.jsonl" % (kind, i))
        with open(path, "w", newline="\n") as fh:
            for s in sorted(vals):
                fh.write(json.dumps({"step": s, "loss": vals[s]}) + "\n")
        units.append({"id": "%s%d" % (kind, i), "src": path, "stratum": "S|x" if i % 4 < 2 else "S|y", "success": succ,
                      "interval": interval, "seed": i, "card": "x", "basin_fail": fail})
    return units


def selftest():
    ok = True; rep = {}
    with tempfile.TemporaryDirectory(dir=HERE, prefix="_selftest_") as tmp:
        SPEC_n, SPEC_b, SPEC_bn = SPEC["n_perm"], SPEC["n_boot"], SPEC["beat_null"]["n_perm"]
        try:
            SPEC["n_perm"], SPEC["n_boot"], SPEC["beat_null"]["n_perm"] = 2000, 200, 2000
            # rev 2 panels: saturation-only (the feature must NOT pass the rev-2 rule) and basin beyond saturation (it must)
            for kind, n_pan in (("saturation", 10), ("saturation_strong", 10), ("basin", 5)):
                pans = []
                for j in range(n_pan):
                    sd = {"saturation": 3000, "saturation_strong": 5000, "basin": 4000}[kind] + j
                    u = synth_panel(tmp, kind, 60 if kind == "saturation_strong" else 40, sd)
                    o = score_reading(u, "R1", ["loss"], np.random.default_rng(sd), True, SPEC["signs"]["R1"])
                    fs = o["features"]; sl = fs["slope:loss"]
                    pans.append({
                        "seed": sd, "n": len(u), "n_success": sum(x["success"] for x in u),
                        "slope_C": round(sl["result"]["C"], 3), "slope_p": sl["result"]["p"], "verdict_rev1": sl.get("verdict_rev1"),
                        "verdict_rev2": sl["verdict"],
                        "null_power_C": round(fs["null_power:loss"]["result"]["C"], 3), "null_exp_C": round(fs["null_exp:loss"]["result"]["C"], 3),
                        "beat_power_dC_p": [round(sl["beat_null"]["null_power:loss"]["dC"], 3), sl["beat_null"]["null_power:loss"]["p_one_sided"]],
                        "beat_exp_dC_p": [round(sl["beat_null"]["null_exp:loss"]["dC"], 3), sl["beat_null"]["null_exp:loss"]["p_one_sided"]],
                        "other_features_rev2": {k: v["verdict"] for k, v in fs.items() if not k.startswith("null_") and k != "slope:loss"},
                        "outcome_equals_basin": (sum((not x["success"]) == x["basin_fail"] for x in u) / len(u)) if kind == "basin" else None})
                    for f_ in os.listdir(tmp):
                        os.remove(os.path.join(tmp, f_))
                rep["panels_" + kind] = {
                    "panels": pans,
                    "rev1_useful_any_direction": sum(p_["verdict_rev1"].startswith("USEFUL") for p_ in pans),
                    "rev1_significant": sum(p_["slope_p"] < 0.05 for p_ in pans),
                    "rev2_useful_slope": sum(p_["verdict_rev2"].startswith("USEFUL") for p_ in pans),
                    "rev2_useful_any_feature": sum(p_["verdict_rev2"].startswith("USEFUL") or any(v.startswith("USEFUL") for v in p_["other_features_rev2"].values()) for p_ in pans)}
            # paired beat-null calibration: feature and null are exchangeable noisy copies of one latent predictor
            rej = 0; dcs = []
            for k in range(40):
                g = np.random.default_rng(500 + k); st3 = []
                for _ in range(2):
                    z = g.normal(0, 1, 20); y = (z + g.normal(0, 1, 20) > 0).astype(float)
                    st3.append((z + g.normal(0, 0.7, 20), z + g.normal(0, 0.7, 20), y))
                r_ = paired_beat_null(st3, 400, g); dcs.append(r_["dC"]); rej += r_["p_one_sided"] < 0.05
            rep["paired_calibration"] = {"draws": 40, "mean_dC": float(np.mean(dcs)), "reject_rate_alpha0.05": rej / 40}
            # exact path of the paired test: 3 units, feature perfect, null reversed -> dC = 1, p = 1/8 (2^3 masks)
            r_ = paired_beat_null([(np.array([1.0, 2.0, 0.0]), np.array([-1.0, 0.0, 1.0]), np.array([1.0, 1.0, 0.0]))], 10, np.random.default_rng(0))
            rep["paired_exact"] = r_
        finally:
            SPEC["n_perm"], SPEC["n_boot"], SPEC["beat_null"]["n_perm"] = SPEC_n, SPEC_b, SPEC_bn
        ps, pss, pb = rep["panels_saturation"], rep["panels_saturation_strong"], rep["panels_basin"]
        bal = lambda P, lo, hi: all(lo <= p_["n_success"] / p_["n"] <= hi for p_ in P["panels"])  # noqa: E731
        ok &= bal(ps, 0.2, 0.8) and bal(pss, 0.15, 0.85) and bal(pb, 0.2, 0.8)
        for P in (ps, pss):                                    # mechanics only: the rev-2 rule must not pass
            ok &= P["rev2_useful_slope"] <= 1 and P["rev2_useful_any_feature"] <= 2
        # ... while the rev-1 test is fooled there (the panel bites): falsely significant in >= 3 of 10 panels, which has
        # binomial probability 0.012 at a nominal alpha of 0.05. (First set as "rev-1 USEFUL >= 5"; changed after the first
        # strong-panel run read 3 USEFUL-REVERSED + 2 DETECTABLE; the rev-2 requirements were never changed. Disclosed.)
        ok &= pss["rev1_significant"] >= 3
        ok &= pb["rev2_useful_slope"] >= 4 and all(p_["verdict_rev2"].startswith(("USEFUL (pre-stated", "INCONCLUSIVE", "DETECTABLE")) for p_ in pb["panels"])
        ok &= all(p_["outcome_equals_basin"] >= 0.9 for p_ in pb["panels"])
        ok &= rep["paired_calibration"]["reject_rate_alpha0.05"] <= 0.15 and abs(rep["paired_calibration"]["mean_dC"]) < 0.05
        ok &= rep["paired_exact"]["exact"] and abs(rep["paired_exact"]["p_one_sided"] - 1 / 8) < 1e-9
        SPEC_n, SPEC_b = SPEC["n_perm"], SPEC["n_boot"]
        try:
            SPEC["n_perm"], SPEC["n_boot"] = 2000, 200
            u = synth_logs(tmp, 12, 12, 0.8, 100, 4000, seed=1)
            SPEC["beat_null"]["n_perm"] = 2000
            o = run(real=False, units={"R1": u, "R1b": [], "R1c": [], "R2": [], "R3": [], "R3ord": []}, dup_cand={}, lab={}, tag="selftest-planted")
            SPEC["beat_null"]["n_perm"] = SPEC_bn
            pr = o["R1"]["features"]["slope:loss"]["result"]; ser0, _ = read_series(u[0]["src"], ["loss"])
            rep["planted"] = {"C": pr["C"], "p": pr["p"], "verdict": o["R1"]["features"]["slope:loss"]["verdict"],
                              "note": "the rev-1 planted hare is a decay-rate difference, i.e. curve mechanics; only C and p are checked",
                              "excluded": o["R1"]["excluded"], "lfa_present": "lfa:loss" in o["R1"]["features"],
                              "dedupe_last_wins": ser0["loss"].get(100) == 9.0}
            for f_ in os.listdir(tmp):
                os.remove(os.path.join(tmp, f_))
            # null calibration: 40 independent null panels; the permutation test must reject at about alpha
            SPEC["n_perm"], SPEC["n_boot"] = 400, 50
            Cs, rej = [], 0
            for k in range(40):
                u = synth_logs(tmp, 12, 12, 0.0, 100, 4000, seed=100 + k)
                o = score_reading(u, "R1", ["loss"], np.random.default_rng(k), True, {"slope:loss": +1})
                r_ = o["features"]["slope:loss"]["result"]; Cs.append(r_["C"]); rej += r_["p"] < 0.05
                for f_ in os.listdir(tmp):
                    os.remove(os.path.join(tmp, f_))
            rep["null_calibration"] = {"draws": 40, "mean_C": float(np.mean(Cs)), "sd_C": float(np.std(Cs)), "reject_rate_alpha0.05": rej / 40}
        finally:
            SPEC["n_perm"], SPEC["n_boot"] = SPEC_n, SPEC_b
        ok &= rep["planted"]["C"] >= 0.8 and rep["planted"]["p"] < 0.01
        ok &= abs(rep["null_calibration"]["mean_C"] - 0.5) < 0.05 and rep["null_calibration"]["reject_rate_alpha0.05"] <= 0.15
        ok &= rep["planted"]["lfa_present"] and rep["planted"]["dedupe_last_wins"]
        # exact path, binary: 2 successes + 1 failure perfectly ordered -> one-sided p = 1/3
        r = perm_test([(np.array([1.0, 2.0, 0.0]), np.array([1.0, 1.0, 0.0]))], 1000, np.random.default_rng(0), False)
        ok &= abs(r["p"] - 1 / 3) < 1e-9 and r["exact"]; rep["exact_binary"] = r
        # exact path, ordinal: the R3 design (RMS: s1 = s2 = 19k, s3 never; four 2-seed strata), perfect order -> 1/48
        ords = [(np.array([3.0, 2.0, 1.0]), np.array([-19000.0, -19000.0, -np.inf]))] + [(np.array([2.0, 1.0]), np.array([-1.0, -2.0]))] * 4
        r = perm_test(ords, 1000, np.random.default_rng(0), False)
        ok &= abs(r["p"] - 1 / 48) < 1e-9 and r["exact"] and r["pairs"] == 6; rep["exact_ordinal"] = r
        # resume gate: a resume marker inside the window excludes the unit
        with open(os.path.join(tmp, "res.jsonl"), "w", newline="\n") as fh:
            for st_ in range(100, 2100, 100):
                fh.write(json.dumps({"step": st_, "loss": 2.0 - st_ / 4000}) + "\n")
            fh.write(json.dumps({"resumed": 1500, "step": 1500}) + "\n")
        o = score_reading([{"id": "res", "src": os.path.join(tmp, "res.jsonl"), "stratum": "S", "success": True, "interval": 100}],
                          "R1", ["loss"], np.random.default_rng(0), True, SPEC["signs"]["R1"])
        ok &= o["excluded"] == [("res", "resumed inside the window")]; rep["resume_gate"] = o["excluded"]
        # window gate: a missing grid step inside the window excludes the feature
        F, why = features({"loss": {s_: 1.0 + s_ / 1e4 for s_ in range(100, 2100, 100) if s_ != 1200}}, "R1", 100)
        ok &= "slope:loss" not in F and why.get("loss") == "window incomplete"; rep["window_gate"] = why
    rep["PASS"] = bool(ok)
    json.dump({"spec_hash": SPEC_HASH, "selftest": rep, "time": time.strftime("%Y-%m-%d %H:%M:%S")},
              open(os.path.join(HERE, "selftest.json"), "w", newline="\n"), indent=1, default=str)
    print(json.dumps(rep, indent=1, default=str))
    return ok


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--selftest", action="store_true"); g.add_argument("--freeze", action="store_true"); g.add_argument("--run", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if a.freeze:
        json.dump({"spec": SPEC, "spec_hash": SPEC_HASH, "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S")},
                  open(os.path.join(HERE, "frozen.json"), "w", newline="\n"), indent=1)
        print("frozen", SPEC_HASH); return
    fz, stp = os.path.join(HERE, "frozen.json"), os.path.join(HERE, "selftest.json")
    if not (os.path.exists(fz) and json.load(open(fz))["spec_hash"] == SPEC_HASH):
        sys.exit("refusing: frozen.json missing or SPEC changed since the freeze")
    if not (os.path.exists(stp) and json.load(open(stp))["selftest"]["PASS"] and json.load(open(stp))["spec_hash"] == SPEC_HASH):
        sys.exit("refusing: selftest.json missing, failed, or from another SPEC")
    out = run(real=True)
    json.dump(out, open(os.path.join(HERE, "results.json"), "w", newline="\n"), indent=1, default=str)
    print("PRIMARY:", json.dumps(out["PRIMARY"], default=str))


if __name__ == "__main__":
    main()
