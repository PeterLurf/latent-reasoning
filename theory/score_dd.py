"""THEORY §265 DENSE-DISTRACT scorer (DD_v1 in THEORY §265; Theory is the only scorer; committed before output).
Arms (warm from bun_sX; the only change is --distract_nmax: inserts n ~ U{0..nmax}, bun = 2):
  pilot 2k: ddp_s1 / ddp_s2 (nmax 8) vs ddc_s1 / ddc_s2 (nmax 2, the extra-steps CTRL);   full 8k: dd_s1 / dd_s2 (nmax 8).
PILOT GO (mechanism-gated, Note 264-1): mean over seeds of [acc_D(d 8) - acc_C(d 8)] >= 0.10 AND mean over seeds of [rel7_D - rel7_C] >= 0.10, where
  rel7 = P(pass-7 WS ok | passes 1-6 WS ok) on TEST2 d 8 (eval_ws_deep items), AND every D cell TEST2 d 1-4 acc >= 0.98. Else PARK.
FULL binding per seed (vs bun_sX): C2-EXT d 8 acc >= 0.75 AND d 7 >= bun d 7; RETAIN TEST2 d 1-4 >= 0.98, d 6 >= bun d 6 - 0.02, NOT-DECODER d 4 K'2 <= 0.20, SWAP4,
  LP5-D4; C6 eval_long_k128 d 1-6 acc[64], acc[128] >= acc[8] - 0.02; CONTENT H-CHAINSRC-OP acc8 >= 0.85 and min(by_pos) >= 0.80; STABLE (score_ao).
  Label DENSE-C2 if all pass, else 'FAIL: <criteria>'; per arm k/2.
D1a ATTENTION-FADING (diagnostic, descriptive; on bun_s1 / bun_s2 eval_attn_fade.json; layout conds[c][t] = per item [12][12] target mass (the ':' query's
  attention on S_t's op-sentence tokens, layer x head), ws_ok_d8_t7 = per item 0/1 (the d 8 items' pass-7 WS ok given 1-6 ok, None otherwise)):
  selector heads = top-5 (layer, head) by mean target mass on d4k0 passes 2-4, even-index items; on odd items: drop_d8 = 1 - m(d8, t4) / m(d4k0, t4),
  drop_k6 = 1 - m(d4k6, t4) / m(d4k0, t4); AUC of m(d8, t7) for ws_ok (1 vs 0). FADING if max(drop_d8, drop_k6) >= 0.30 AND AUC >= 0.65 in both seeds.
    python -B theory/score_dd.py [--fixtures]
"""
import argparse, json, os, sys, tempfile
_HR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_HR, "theory"))
import score_ao as AO  # noqa: E402

R = os.path.join(_HR, "boxes", "runs_fsq")
OUT = os.path.join(_HR, "theory", "results", "score_dd.json")
PILOT = {"1": ("ddp_s1", "ddc_s1"), "2": ("ddp_s2", "ddc_s2")}
FULL = {"dd_s1": "bun_s1", "dd_s2": "bun_s2"}
BASE = ("bun_s1", "bun_s2")


def jl(p):
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return None


def rel7(ws):
    it = ws["items"]["8"]; n = len(it["sig"]); num = den = 0
    for i in range(n):
        ok = [it["ws_pred"][str(t)][i] == it["v"][i][t - 1] for t in range(1, 8)]
        if all(ok[:6]):
            den += 1; num += ok[6]
    return num / den if den else 0.0


def pilot(runs):
    rows = {}
    for s, (dn, cn) in PILOT.items():
        f = {k: jl(os.path.join(runs, r, k2 + ".json")) for k, r in (("D", dn), ("C", cn)) for k2 in ("eval_acc",)}
        w = {k: jl(os.path.join(runs, r, "eval_ws_deep.json")) for k, r in (("D", dn), ("C", cn))}
        if any(v is None for v in list(f.values()) + list(w.values())):
            return {"verdict": "pending"}
        aD, aC = f["D"]["by_d"], f["C"]["by_d"]
        rows[s] = {"d8_D": aD["8"]["acc"], "d8_C": aC["8"]["acc"], "rel7_D": rel7(w["D"]), "rel7_C": rel7(w["C"]),
                   "retain_D": all(aD[d]["acc"] >= 0.98 for d in ("1", "2", "3", "4"))}
    gd8 = sum(r["d8_D"] - r["d8_C"] for r in rows.values()) / len(rows); g7 = sum(r["rel7_D"] - r["rel7_C"] for r in rows.values()) / len(rows)
    go = gd8 >= 0.10 and g7 >= 0.10 and all(r["retain_D"] for r in rows.values())
    return {"verdict": "GO" if go else "PARK", "gain_d8": gd8, "gain_rel7": g7, "seeds": rows}


def full_seed(runs, run, ref):
    f = {x: jl(os.path.join(runs, run, x + ".json")) for x in ("eval_acc", "eval_swap_kd", "eval_lp5h", "eval_long_k128", "eval_insert_ho", "eval_ws_deep")}
    rf = jl(os.path.join(runs, ref, "eval_acc.json"))
    if any(v is None for v in list(f.values()) + [rf]):
        return {"label": "pending"}
    a = f["eval_acc"]["by_d"]; r = rf["by_d"]; sw = f["eval_swap_kd"]["by_d"]["4"]["swap_mid"]["4"]; h = f["eval_lp5h"]["by_d"]["4"]; L = f["eval_long_k128"]["by_d"]
    c2 = a["8"]["acc"] >= 0.75 and a["7"]["acc"] >= r["7"]["acc"]
    retain = (all(a[d]["acc"] >= 0.98 for d in ("1", "2", "3", "4")) and a["6"]["acc"] >= r["6"]["acc"] - 0.02 and a["4"]["acc_kprime"]["2"] <= 0.20
              and sw["follow_U"] >= 0.50 and sw["follow_U"] >= sw["floor"] + 0.30 and h["intact"]["4"] - h["mean_at_t"]["4"] >= 0.30)
    c6 = all(L[d]["acc"][k] >= L[d]["acc"]["8"] - 0.02 for d in ("1", "2", "3", "4", "5", "6") for k in ("64", "128"))
    co = f["eval_insert_ho"]["fam"]["H-CHAINSRC-OP"]; content = co["acc8"] >= 0.85 and min(co["by_pos"].values()) >= 0.80
    sok, smsg = AO.stable(runs, run)
    crit = {"C2-EXT": c2, "RETAIN": retain, "C6": c6, "CONTENT": content, "STABLE": sok is not False}; fail = [k for k, v in crit.items() if not v]
    return {"label": "DENSE-C2" if not fail else "FAIL: " + ",".join(fail), "crit": crit, "d7": a["7"]["acc"], "d8": a["8"]["acc"], "bun_d7": r["7"]["acc"],
            "bun_d8": r["8"]["acc"], "rel7": rel7(f["eval_ws_deep"]), "d9_12": {d: a[d]["acc"] for d in ("9", "10", "11", "12") if d in a}, "stable": (sok, smsg)}


def auc(pos, neg):
    if not pos or not neg:
        return None
    return sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg))


def fade_seed(runs, run):
    f = jl(os.path.join(runs, run, "eval_attn_fade.json"))
    if f is None:
        return None
    C = f["conds"]; n = len(C["d4k0"]["2"]); ev = range(0, n, 2)
    mean = lambda arr, idx, l, h: sum(arr[i][l][h] for i in idx) / len(idx)
    sc = sorted(((sum(mean(C["d4k0"][t], ev, l, h) for t in ("2", "3", "4")), l, h) for l in range(12) for h in range(12)), reverse=True)[:5]
    heads = [(l, h) for _, l, h in sc]
    m = lambda c, t, idx: sum(sum(C[c][t][i][l][h] for l, h in heads) / len(heads) for i in idx) / len(idx)
    od = list(range(1, n, 2)); od8 = list(range(1, len(C["d8"]["4"]), 2)); odk = list(range(1, len(C["d4k6"]["4"]), 2))
    base = m("d4k0", "4", od); d8 = m("d8", "4", od8); k6 = m("d4k6", "4", odk)
    per = [sum(C["d8"]["7"][i][l][h] for l, h in heads) / len(heads) for i in range(len(C["d8"]["7"]))]; ok = f["ws_ok_d8_t7"]
    a = auc([p for p, o in zip(per, ok) if o == 1], [p for p, o in zip(per, ok) if o == 0])
    drop_d8, drop_k6 = 1 - d8 / base, 1 - k6 / base
    return {"heads": heads, "m_d4k0_t4": base, "m_d8_t4": d8, "m_d4k6_t4": k6, "drop_d8": drop_d8, "drop_k6": drop_k6, "auc_t7": a,
            "fading": max(drop_d8, drop_k6) >= 0.30 and a is not None and a >= 0.65}


def score(runs=R):
    o = {"section": 265, "pilot": pilot(runs), "full": {r: full_seed(runs, r, ref) for r, ref in FULL.items()}, "fade": {r: fade_seed(runs, r) for r in BASE}}
    labs = [v["label"] for v in o["full"].values()]
    o["full_verdict"] = "pending" if "pending" in labs else "DENSE-C2 %d/2 (%s)" % (labs.count("DENSE-C2"), " / ".join(labs))
    fd = list(o["fade"].values())
    o["fade_verdict"] = "pending" if any(x is None for x in fd) else ("FADING" if all(x["fading"] for x in fd) else "NOT-FADING" if not any(x["fading"] for x in fd) else "MIXED")
    o["verdict"] = "pilot %s; full %s; D1a %s" % (o["pilot"]["verdict"], o["full_verdict"], o["fade_verdict"]); return o


def fixtures():
    res = []

    def ws(n, p7_ok):
        v = [[(t * 3) % 11 for t in range(1, 9)] for _ in range(n)]
        pred = {str(t): [v[i][t - 1] if (t != 7 or i < int(p7_ok * n)) else (v[i][t - 1] + 1) % 11 for i in range(n)] for t in range(1, 9)}
        return {"items": {"8": {"sig": list(range(n)), "ws_pred": pred, "v": v}}}

    def world(d8D=0.75, d8C=0.55, r7D=0.95, r7C=0.7, ret=1.0, full_d8=0.8, d7=0.9, k2=0.05, hold=1.0, co=0.95, fade=True):
        dd = tempfile.mkdtemp()

        def put(run, name, obj):
            os.makedirs(os.path.join(dd, run), exist_ok=True); json.dump(obj, open(os.path.join(dd, run, name), "w"))

        def acc(d8, d7_, t2=1.0):
            return {"by_d": {**{str(x): {"acc": t2, "acc_kprime": {"2": k2 if x == 4 else 1.0}} for x in range(1, 5)}, "5": {"acc": 0.99}, "6": {"acc": 0.97},
                             "7": {"acc": d7_}, "8": {"acc": d8}}}
        for s in ("1", "2"):
            put("ddp_s" + s, "eval_acc.json", acc(d8D, 0.9, ret)); put("ddc_s" + s, "eval_acc.json", acc(d8C, 0.86))
            put("ddp_s" + s, "eval_ws_deep.json", ws(100, r7D)); put("ddc_s" + s, "eval_ws_deep.json", ws(100, r7C))
            put("bun_s" + s, "eval_acc.json", {"by_d": {"6": {"acc": 0.97}, "7": {"acc": 0.86}, "8": {"acc": 0.5}}})
            run = "dd_s" + s; put(run, "eval_acc.json", acc(full_d8, d7)); put(run, "eval_ws_deep.json", ws(100, 0.95))
            put(run, "eval_swap_kd.json", {"by_d": {"4": {"swap_mid": {"4": {"follow_U": 0.9, "floor": 0.1}}}}})
            put(run, "eval_lp5h.json", {"by_d": {"4": {"intact": {"4": 1.0}, "mean_at_t": {"4": 0.1}}}})
            put(run, "eval_long_k128.json", {"by_d": {str(x): {"acc": {"8": 1.0, "64": 1.0, "128": hold}} for x in range(1, 7)}})
            put(run, "eval_insert_ho.json", {"fam": {"H-CHAINSRC-OP": {"acc8": co, "by_pos": {"1": 0.9}}}})
            put(run, "train_summary.json", {"n_nonfinite": 0})
            with open(os.path.join(dd, run, "log.jsonl"), "w") as fh:
                for i in range(1, 81):
                    fh.write(json.dumps({"step": i * 100, "loss": 0.01}) + chr(10))
            z = [[0.01] * 12 for _ in range(12)]

            def att(val):
                a_ = [r[:] for r in z]; a_[3][4] = val; return a_
            n = 40; lo = 0.2 if fade else 0.5
            conds = {"d4k0": {t: [att(0.5) for _ in range(n)] for t in ("2", "3", "4")}, "d4k6": {"4": [att(lo) for _ in range(n)]},
                     "d8": {"4": [att(lo) for _ in range(n)], "7": [att(0.6 if i % 2 == 0 else 0.1) for i in range(n)]}}
            okv = [1 if (i % 2 == 0 or not fade) else 0 for i in range(n)]
            if not fade:
                okv = [i % 2 for i in range(n)]; conds["d8"]["7"] = [att(0.3) for _ in range(n)]
            put("bun_s" + s, "eval_attn_fade.json", {"conds": conds, "ws_ok_d8_t7": okv})
        return dd
    for name, kw, want in (("all", {}, "pilot GO; full DENSE-C2 2/2 (DENSE-C2 / DENSE-C2); D1a FADING"),
                           ("pilot d8 gain small", {"d8D": 0.6}, "pilot PARK"), ("pilot rel7 gain small", {"r7D": 0.75}, "pilot PARK"),
                           ("pilot retain", {"ret": 0.95}, "pilot PARK"), ("full d8", {"full_d8": 0.7}, "FAIL: C2-EXT"), ("full d7", {"d7": 0.8}, "FAIL: C2-EXT"),
                           ("decoder", {"k2": 0.5}, "FAIL: RETAIN"), ("c6", {"hold": 0.5}, "FAIL: C6"), ("content", {"co": 0.5}, "FAIL: CONTENT"),
                           ("not fading", {"fade": False}, "D1a NOT-FADING")):
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
