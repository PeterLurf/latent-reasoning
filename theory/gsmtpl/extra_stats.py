import ast
"""GSM-TEMPLATE draft, measurement 4 (CPU, read-only): (i) length -> depth regressions (E[depth | L]) on ProsQA p8 and GSM8K-Aug, and the size-keyed depth
over-statement of p8_ctl / GSM8K test; (ii) functional-fingerprint structure overlap (skeleton evaluated on fixed pseudo-random literal values);
(iii) the numeral composition of targets in recomputable items.  Reuses audit_aug's parsers (imported from this folder).   python -I extra_stats.py"""
import collections
import json
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import audit_aug as AA  # noqa: E402  (own script; sets idle priority)

ENC = AA.ENC


def corr(x, y):
    n = len(x); mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y)); sxx = sum((a - mx) ** 2 for a in x); syy = sum((b - my) ** 2 for b in y)
    return sxy / math.sqrt(sxx * syy), sxy / sxx


def cond_mean(L, d, w):
    b = collections.defaultdict(list)
    for a, c in zip(L, d):
        b[a // w].append(c)
    return {k: sum(v) / len(v) for k, v in b.items()}, b


def est(cm, L, w, lo, hi):
    k = min(max(L // w, lo), hi)
    while k not in cm:
        k += 1 if k < hi else -1
    return cm[k]


def fingerprint(skel, vals):
    if "u" in skel or "?" in skel or not skel:
        return None
    res = []
    for eq in skel.split(";"):
        s = eq
        for j in range(len(res), 0, -1):
            s = s.replace("r%d" % j, repr(res[-j]))
        import re
        s = re.sub(r"q(\d+)", lambda m: repr(vals[int(m.group(1))]) if int(m.group(1)) < len(vals) else "1.0", s)
        s = re.sub(r"k([\d.e+-]+)", lambda m: m.group(1), s)
        try:
            res.append(AA.safe_eval(s))
        except AA.Bad:
            return None
    return round(res[-1], 6) if res else None


def main():
    out = {}
    P = json.load(open("D:/Latent/prosqa/p8data/p8_na_train.json")); C = json.load(open("D:/Latent/prosqa/p8data/p8_ctl.json"))
    T = json.load(open("D:/Latent/prosqa/p8data/p8_na_test.json"))
    st = lambda x: x["steps"] if isinstance(x["steps"], list) else ast.literal_eval(x["steps"])  # noqa: E731
    Lp = [1 + len(t) for t in ENC.encode_ordinary_batch([x["question"] for x in P], num_threads=3)]; dp = [len(st(x)) for x in P]
    Lc = [1 + len(t) for t in ENC.encode_ordinary_batch([x["question"] for x in C], num_threads=3)]; dc = [len(st(x)) for x in C]
    Lt = [1 + len(t) for t in ENC.encode_ordinary_batch([x["question"] for x in T], num_threads=3)]; dt = [len(st(x)) for x in T]
    r, b = corr(Lp, dp); cm, _ = cond_mean(Lp, dp, 20); lo, hi = min(cm), max(cm)
    over_c = [est(cm, L, 20, lo, hi) - d for L, d in zip(Lc, dc)]; over_t = [est(cm, L, 20, lo, hi) - d for L, d in zip(Lt, dt) if d <= 10]
    ext = [str(x.get("ext")) for x in C]
    out["prosqa"] = {"corr_L_depth_train": round(r, 4), "R2": round(r * r, 4), "slope_depth_per_token": round(b, 5),
                     "ctl_mean_overstatement_hops": round(sum(over_c) / len(over_c), 3),
                     "ctl_overstatement_by_ext": {e: round(sum(o for o, x in zip(over_c, ext) if x == e) / ext.count(e), 3) for e in sorted(set(ext))},
                     "test_d3to10_mean_overstatement_hops": round(sum(over_t) / len(over_t), 3), "bin_tokens": 20}
    Q, E, A = AA.load_aug("train"); La = [1 + len(t) for t in ENC.encode_ordinary_batch(Q, num_threads=3)]; ma = [len(e) for e in E]
    r, b = corr(La, ma); cm, _ = cond_mean(La, ma, 4); lo, hi = min(cm), max(cm)
    Qt, Et, At = AA.load_openai("gsm8k_test"); Lt = [1 + len(t) for t in ENC.encode_ordinary_batch(Qt, num_threads=3)]; mt = [len(e) for e in Et]
    ov = [est(cm, L, 4, lo, hi) - m for L, m in zip(Lt, mt)]
    resid = []
    byM = collections.defaultdict(list)
    for L, m in zip(La, ma):
        byM[m].append(L)
    mu = {m: sum(v) / len(v) for m, v in byM.items()}
    resid = [L - mu.get(m, 43.3) for L, m in zip(Lt, mt)]; order = sorted(range(len(resid)), key=lambda i: resid[i]); n = len(order)
    terc = [order[:n // 3], order[n // 3:2 * n // 3], order[2 * n // 3:]]
    Qv, Ev, Av = AA.load_aug("valid"); Lv = [1 + len(t) for t in ENC.encode_ordinary_batch(Qv, num_threads=3)]; mv = [len(e) for e in Ev]
    ovv = [est(cm, L, 4, lo, hi) - m for L, m in zip(Lv, mv)]
    rt, _ = corr(Lt, mt)
    out["gsm8k"] = {"corr_L_m_aug_train": round(r, 4), "R2": round(r * r, 4), "slope_m_per_token": round(b, 5), "corr_L_m_test": round(rt, 4),
                    "test_mean_size_keyed_depth_error_eq": round(sum(ov) / len(ov), 3), "aug_valid_mean_error_eq": round(sum(ovv) / len(ovv), 3),
                    "test_error_by_length_residual_tercile_eq": [round(sum(ov[i] for i in t) / len(t), 3) for t in terc],
                    "test_m_mean_by_tercile": [round(sum(mt[i] for i in t) / len(t), 3) for t in terc],
                    "test_mean_m": round(sum(mt) / len(mt), 3), "aug_mean_m": round(sum(ma) / len(ma), 3), "bin_tokens": 4,
                    "note": "size-keyed depth error = E_aug[m | L-bin] - m (negative = length UNDER-states depth)"}
    # fingerprints
    rng = random.Random(7); V1 = [rng.choice([3, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43]) * 1.0 + rng.random() for _ in range(40)]
    V2 = [rng.uniform(2, 50) for _ in range(40)]
    Ia = [AA.analyse(q, e, a) for q, e, a in zip(Q, E, A)]
    def fp(it):
        a, b2 = fingerprint(it["skel"], V1), fingerprint(it["skel"], V2)
        return None if a is None or b2 is None else (a, b2, len(it["st"]))
    FA = collections.Counter(fp(x) for x in Ia); FA.pop(None, None)
    It = [AA.analyse(q, e, a) for q, e, a in zip(Qt, Et, At)]; ft = [fp(x) for x in It]
    Qg, Eg, Ag = AA.load_openai("gsm8k_train"); Ig = [AA.analyse(q, e, a) for q, e, a in zip(Qg, Eg, Ag)]; FG = collections.Counter(fp(x) for x in Ig); FG.pop(None, None)
    seen = [f is not None and f in FA for f in ft]
    by_m = collections.defaultdict(lambda: [0, 0])
    for s, m in zip(seen, mt):
        by_m[m][0] += s; by_m[m][1] += 1
    out["fingerprint"] = {"def": "skeleton evaluated with question literal q_i := fixed pseudo-random values (2 assignments), constants kept; key = (value1, value2, n_eq); "
                                 "commutation / step-order variants of one function collide (intended); items with an ungrounded or unparsed operand get no key",
                          "aug_train_distinct": len(FA), "gsm8k_train_distinct": len(FG), "test_keyed": sum(f is not None for f in ft),
                          "test_seen_in_aug_train": round(sum(seen) / len(seen), 4),
                          "test_seen_in_gsm8k_train": round(sum(f is not None and f in FG for f in ft) / len(ft), 4),
                          "test_seen_by_m": {m: [v[0], v[1], round(v[0] / v[1], 3)] for m, v in sorted(by_m.items())}}
    # numeral composition of targets in recomputable items
    var = fix = 0; nm_tok = q_tok = 0
    for it in Ia:
        rec = len(it["st"]) > 0 and all(s == "exact" for s in it["st"]) and it["ungrounded"] == 0 and it["ans_last"] and it["q_unique"]
        if rec:
            c = it["cats"]; var += c["Q"] + c["R"] + len(it["st"]); fix += c["W"] + c["K"]
    out["numerals_recomputable_items"] = {"variable_share": round(var / (var + fix), 4), "def": "operands Q + R + one result per equation vary under literal resampling; W + K stay"}
    json.dump(out, open(os.path.join(HERE, "extra_stats.json"), "w", newline="\n"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
