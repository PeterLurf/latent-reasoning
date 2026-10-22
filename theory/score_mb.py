"""THEORY §225 MATCHED-BUDGET scorer (registered text docs/research_matchedbudget_2026-10-22/MB_v1.md at e9096291; Theory is the only scorer).

Inputs: boxes/runs_fsq/s216_r2_s1/eval_deep.json (items {n: {idx, y, T, ans [[argmax at K' 0..28]]}}, acc_k {K': {n}}) and
boxes/runs_fsq/s219_t2_s1/eval_text.json (items [{idx, n, y, P, answer, ntok, overflow}], acc_by_n {n}); items matched on (n, idx), y must agree.
Per item cost / correctness:
  text   correct = answer == y and not overflow; S = ntok + 1 (prefill counted as one step); F = 12 * (P + ntok).
  latent k* = the smallest K' in 0..28 with ans[k] == y for every k in k*..28 (None: not correct at 28); S = k* + 1; F = T * (6 + 4 k* + 2).
  correct at budget B iff correct and cost <= B (truncation = the step cut; latent never extrapolated past K' 28).
Gates (else VOID): text acc at B = inf == eval_text acc_by_n[n]; latent share with k* != None == acc_k['28'][n]; y agree on matched items; 1,000 per n.
Labels per axis X in {S, F} over n 13..20 that reach 0.5 in either arm by the largest grid B:
  C3-MATCHED-X: latent >= text - 0.02 at every grid B; TEXT-AHEAD-X: text >= latent + 0.05 at some B for >= 3 n; MIXED-X otherwise.
Also B*(n, q) for q 0.5 / 0.9 per arm and axis ("> grid" if never), n 13..24. Descriptive only: an approximate attention-inclusive FLOPs row (formula in
ATT_DEF). "C3 met" only if both axes read C3-MATCHED.
    python -B theory/score_mb.py [--fixtures]
"""
import argparse, json, os, sys, tempfile
import numpy as np

_HR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(_HR, "boxes", "runs_fsq")
OUT = os.path.join(_HR, "theory", "results", "score_mb.json")
LAT, TXT = "s216_r2_s1", "s219_t2_s1"
GRID = {"S": list(range(1, 30)) + [33, 49, 65, 129, 257, 513, 1025, 2049], "F": [2 ** e for e in range(5, 18)]}
NS, NLAB = range(13, 25), range(13, 21)
KCAP = 28
D = 768
ATT_DEF = ("approx FLOPs per item: dense 24 D^2 per block-position + attention 4 D L per block-position (L = attended length). text: 12 blocks, "
           "prefill L ~ P/2 per position, decode position i attends P + i. latent: prelude 6 blocks and coda 2 blocks at L = T; each core pass 4 blocks at "
           "L = T + t (slots plus the slot's t bank versions, an upper bound). Descriptive only.")


def jl(p):
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return None


def kstar(ans, y):
    k = None
    for j in range(len(ans) - 1, -1, -1):
        if ans[j] == y:
            k = j
        else:
            break
    return k


def att_text(P, nt):
    dense = 24 * D * D * 12 * (P + nt); att = 4 * D * 12 * (P * P / 2 + sum(P + i for i in range(nt)))
    return dense + att


def att_lat(T, k):
    dense = 24 * D * D * T * (8 + 4 * k); att = 4 * D * T * (8 * T + 4 * sum(T + t for t in range(1, k + 1)))
    return dense + att


def build(runs):
    ld, td = jl(os.path.join(runs, LAT, "eval_deep.json")), jl(os.path.join(runs, TXT, "eval_text.json"))
    if ld is None or td is None:
        return None, "pending (missing %s)" % [n for n, v in ((LAT, ld), (TXT, td)) if v is None]
    tx = {}
    for it in td["items"]:
        if 13 <= it["n"] <= 24:
            tx[(it["n"], it["idx"])] = it
    rows, bad = {}, []
    for n in NS:
        L = ld["items"][str(n)]; R_ = []
        for q in range(len(L["idx"])):
            t = tx.get((n, L["idx"][q]))
            if t is None or int(t["y"]) != int(L["y"][q]):
                bad.append((n, L["idx"][q])); continue
            k = kstar(L["ans"][q], L["y"][q]); nt = int(t["ntok"]) if t.get("ntok") is not None else 0
            tc = (t["answer"] == t["y"]) and not t.get("overflow")
            R_.append({"lat_ok": k is not None, "lat_S": (k + 1) if k is not None else None, "lat_F": L["T"][q] * (8 + 4 * k) if k is not None else None,
                       "lat_A": att_lat(L["T"][q], k) if k is not None else None,
                       "txt_ok": bool(tc), "txt_S": nt + 1, "txt_F": 12 * (int(t["P"]) + nt), "txt_A": att_text(int(t["P"]), nt)})
        rows[n] = R_
        if len(R_) != 1000:
            bad.append((n, "count %d" % len(R_)))
        tacc = float(np.mean([r["txt_ok"] for r in R_])); lacc = float(np.mean([r["lat_ok"] for r in R_]))
        if abs(tacc - td["acc_by_n"][str(n)]) > 1e-12:
            bad.append((n, "text acc %s vs %s" % (tacc, td["acc_by_n"][str(n)])))
        if abs(lacc - ld["acc_k"][str(KCAP)][str(n)]) > 1e-12:
            bad.append((n, "latent stay-acc %s vs acc_k[28] %s" % (lacc, ld["acc_k"][str(KCAP)][str(n)])))
    return rows, bad


def curve(R_, arm, ax, B):
    return float(np.mean([r[arm + "_ok"] and r[arm + "_" + ax] is not None and r[arm + "_" + ax] <= B for r in R_]))


def score(runs=R):
    rows, bad = build(runs)
    if rows is None:
        return {"verdict": bad}
    out = {"section": 225, "registered": "MB_v1.md @ e9096291", "gate_problems": bad[:20]}
    if bad:
        out["verdict"] = "VOID (gates: %d problems)" % len(bad); return out
    for ax in ("S", "F"):
        cur = {n: {B: {"lat": curve(rows[n], "lat", ax, B), "txt": curve(rows[n], "txt", ax, B)} for B in GRID[ax]} for n in NS}
        bstar = {n: {a: {q: next((B for B in GRID[ax] if cur[n][B][a] >= q), "> grid") for q in (0.5, 0.9)} for a in ("lat", "txt")} for n in NS}
        lab_ns = [n for n in NLAB if max(cur[n][GRID[ax][-1]].values()) >= 0.5]
        matched = all(cur[n][B]["lat"] >= cur[n][B]["txt"] - 0.02 for n in lab_ns for B in GRID[ax]) if lab_ns else False
        ahead_n = [n for n in lab_ns if any(cur[n][B]["txt"] >= cur[n][B]["lat"] + 0.05 for B in GRID[ax])]
        lab = "C3-MATCHED-%s" % ax if matched else ("TEXT-AHEAD-%s" % ax if len(ahead_n) >= 3 else "MIXED-%s" % ax)
        out[ax] = {"label": lab, "label_ns": lab_ns, "text_ahead_ns": ahead_n, "bstar": bstar, "curves": {n: cur[n] for n in NS}}
    out["attention_flops_def"] = ATT_DEF
    out["attention_flops_per_correct"] = {n: {"lat": float(np.sum([r["lat_A"] for r in rows[n] if r["lat_ok"]]) / max(1, sum(r["lat_ok"] for r in rows[n]))),
                                              "txt": float(np.sum([r["txt_A"] for r in rows[n] if r["txt_ok"]]) / max(1, sum(r["txt_ok"] for r in rows[n])))} for n in NS}
    out["verdict"] = "%s | %s | C3 met: %s" % (out["S"]["label"], out["F"]["label"], out["S"]["label"].startswith("C3") and out["F"]["label"].startswith("C3"))
    return out


# ------------------------------------------------------------------ fixtures
def _write(d, lat_k, txt_nt, txt_ok=1.0, lat_ok=1.0, T=50, P=40, mismatch=False, flip=False, seed=0):
    rng = np.random.default_rng(seed); os.makedirs(os.path.join(d, LAT), exist_ok=True); os.makedirs(os.path.join(d, TXT), exist_ok=True)
    li, ti, acck, accn = {}, [], {str(k): {} for k in range(KCAP + 1)}, {}
    for n in NS:
        ans, ys, okl = [], [], 0
        for q in range(1000):
            y = int(rng.integers(0, 11)); k = lat_k(n); good = rng.random() < lat_ok
            row = [((y if (j >= k and good) else (y + 1) % 11)) for j in range(KCAP + 1)]
            if flip and q == 0:
                row = [y] * (KCAP + 1); row[KCAP] = (y + 1) % 11            # correct early, wrong at the cap -> k* None
            ans.append(row); ys.append(y); okl += row[KCAP] == y and kstar(row, y) is not None
            tg = rng.random() < txt_ok
            ti.append({"idx": q, "n": n, "y": (y + 1) % 11 if (mismatch and q == 5) else y, "P": P, "answer": y if tg else -1, "ntok": txt_nt(n), "overflow": False})
        li[str(n)] = {"idx": list(range(1000)), "y": ys, "T": [T] * 1000, "height": [0] * 1000, "ans": ans}
        for j in range(KCAP + 1):
            acck[str(j)][str(n)] = float(np.mean([a[j] == yy for a, yy in zip(ans, ys)]))
        acck[str(KCAP)][str(n)] = okl / 1000
        accn[str(n)] = float(np.mean([t["answer"] == t["y"] and not t["overflow"] for t in ti if t["n"] == n]))
    json.dump({"items": li, "acc_k": acck}, open(os.path.join(d, LAT, "eval_deep.json"), "w"))
    json.dump({"items": ti, "acc_by_n": accn}, open(os.path.join(d, TXT, "eval_text.json"), "w"))


def fixtures():
    res = []
    d = tempfile.mkdtemp(); _write(d, lambda n: n + 1, lambda n: 30 * n)        # latent: K' n+1, F = 50 * (8 + 4(n+1)) ~ 3-4k; text S ~400, F = 12 * (40 + 30n) ~ 5-9k
    o = score(d); res.append(("latent cheaper on both axes -> MATCHED both, C3 met", o["S"]["label"] == "C3-MATCHED-S" and o["F"]["label"] == "C3-MATCHED-F" and o["verdict"].endswith("True"), o["verdict"]))
    d = tempfile.mkdtemp(); _write(d, lambda n: n + 1, lambda n: 2 * n, T=200)   # text: 2n tokens (cheap); latent F = 200 * (8 + 4(n+1)) ~ 12k vs text 12 * (40 + 2n) ~ 0.8k
    o = score(d); res.append(("text far cheaper in F, latent wins S? -> F TEXT-AHEAD", o["F"]["label"] == "TEXT-AHEAD-F", o["verdict"]))
    d = tempfile.mkdtemp(); _write(d, lambda n: n + 1, lambda n: 30 * n, lat_ok=0.6, txt_ok=0.95)
    o = score(d); res.append(("text more accurate at large B -> not MATCHED in S", o["S"]["label"] != "C3-MATCHED-S", o["verdict"]))
    d = tempfile.mkdtemp(); _write(d, lambda n: n + 1, lambda n: 30 * n, mismatch=True)
    o = score(d); res.append(("y mismatch -> VOID", o["verdict"].startswith("VOID"), o["verdict"]))
    d = tempfile.mkdtemp(); _write(d, lambda n: n + 1, lambda n: 30 * n, flip=True)
    rows, bad = build(d); r0 = rows[13][0]
    res.append(("late flip -> k* None (not counted)", r0["lat_ok"] is False and not bad, (r0["lat_ok"], bad[:2])))
    d = tempfile.mkdtemp(); _write(d, lambda n: n + 1, lambda n: 30 * n); os.remove(os.path.join(d, TXT, "eval_text.json"))
    o = score(d); res.append(("missing -> pending", str(o["verdict"]).startswith("pending"), o["verdict"]))
    d = tempfile.mkdtemp(); _write(d, lambda n: n + 1, lambda n: 30 * n)
    o = score(d); b = o["S"]["bstar"][13]; res.append(("B* S: latent 15 (k* 14 + 1), text 391", b["lat"][0.5] == 15 and b["txt"][0.5] == 513, b))
    for r in res:
        print("FIXTURE", "PASS" if r[1] else "FAIL", r[0], r[2])
    ok = sum(bool(r[1]) for r in res); print("fixtures %d/%d" % (ok, len(res))); return ok == len(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--fixtures", action="store_true"); a = ap.parse_args()
    if a.fixtures:
        sys.exit(0 if fixtures() else 1)
    o = score(); print(o["verdict"])
    os.makedirs(os.path.dirname(OUT), exist_ok=True); json.dump(o, open(OUT, "w", newline=chr(10)), indent=1, default=str)
