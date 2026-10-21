"""§220 GSM-CACHE-SWAP scorer (GCS_v1.md at 3a88762e, v1 + Addendum 1). Written and fixture-checked before any §220 output.

File contract, one per model: <run dir>/eval_gcs.json =
  {"meta": {"model", "stage": 0|1, "card", "dtype", "bs": 1, "test_sha256", "transformers"},
   "gate": {"agree": float, "acc_bs1": float, "acc_official": float},           # G-R: bs-1 INTACT vs official-semantics virtual bs 128
   "asserts": {"cache_rows_ok": bool},                                          # G-C: every non-written cache row bitwise the recipient's
   "golds": {"<idx>": number},                                                  # all 1,319 test golds (for the permutation floor)
   "acc": {"intact": float, "mean_z": float, ...},                              # over all 1,319 items
   "intact_sha": {"<idx>": str},                                                # sha of INTACT's generated token ids per item
   "pairs": [{"r": idx, "d": idx, "out": {"cache": num|None, "state": num|None, "leak": num|None},
              "sha": {"self_cache": str, "self_state": str}}],                  # primary exact-length pairs (Addendum 1: <= 3 uses per donor)
   ...descriptive keys (k_sweep, variants, prefix, qdrop, qmask, logit_lens, scit, graph_matched, pairs_210_4) are passed through}
Labels (Addendum 1 precedence): UNREAD > UNNECESSARY > CARRIER-PORTS > CARRIER-LEAK > SCAFFOLD > CONTEXT-BOUND > MIXED.
    python theory/score_s220.py [--fixtures]
"""
import argparse, json, math, os, sys
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "theory", "results", "score_s220.json")
FILES = {"s0_codi20": os.path.join(ROOT, "boxes", "runs_fsq", "gsm_codi_20ep_s1", "eval_gcs.json"),
         "s0_zen_gpt2": os.path.join(ROOT, "scratch", "s220", "zen_codi_gpt2", "eval_gcs.json"),
         "s1_zen_1b": os.path.join(ROOT, "scratch", "s220", "zen_codi_llama1b", "eval_gcs.json")}
PRIMARY = {0: "s0_codi20", 1: "s1_zen_1b"}
ACC_CODI20 = 0.379                      # 210-4's committed INTACT (G-R extra check for CODI20)
ACC_ZEN_G1P = 0.4245640636846095        # Note 220-2 (b): §209 G1' acc of zen-E/CODI-gpt2 on our harness (scratch/gate_s209/gate.json g1.acc)
NPERM, SEED_PERM, NBOOT, SEED_BOOT, MAX_REUSE, MIN_PAIRS = 1000, 22024, 10000, 22025, 3, 300
EPS = 1e-12


def jl(p):
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return None


def eq(a, b):
    return a is not None and b is not None and float(a) == float(b)


def perm_floor(pairs, arm, golds, seed=SEED_PERM, nperm=NPERM):
    """mean over nperm re-pairings of P(arm output == gold of a random other item whose gold differs from the recipient's)."""
    keys = sorted(golds, key=int); G = np.array([float(golds[k]) for k in keys]); idx = {int(k): i for i, k in enumerate(keys)}
    rng = np.random.default_rng(seed); hits = np.zeros(nperm)
    for p in pairs:
        o = p["out"][arm]
        if o is None:
            continue
        gr = float(golds[str(p["r"])]); cand = np.nonzero((G != gr) & (np.arange(len(G)) != idx[int(p["r"])]))[0]
        js = cand[rng.integers(0, len(cand), nperm)]; hits += (G[js] == float(o))
    return float(hits.mean() / len(pairs)) if pairs else None


def rates(pairs, arm, golds):
    f = [eq(p["out"][arm], golds[str(p["d"])]) for p in pairs]; o = [eq(p["out"][arm], golds[str(p["r"])]) for p in pairs]
    n = len(pairs)
    return {"follow": sum(f) / n, "own": sum(o) / n, "neither": 1 - (sum(f) + sum(o)) / n, "_f": f}


def cluster_ci(pairs, fl, seed=SEED_BOOT, nboot=NBOOT):
    """donor-cluster bootstrap CI of follow (Addendum 1 item 2)."""
    by = {}
    for p, x in zip(pairs, fl):
        by.setdefault(p["d"], []).append(x)
    cl = list(by.values()); rng = np.random.default_rng(seed); R = np.empty(nboot)
    for b in range(nboot):
        pick = rng.integers(0, len(cl), len(cl)); s = sum(sum(cl[i]) for i in pick); n = sum(len(cl[i]) for i in pick); R[b] = s / n
    return [float(np.quantile(R, 0.025)), float(np.quantile(R, 0.975))]


def mcnemar_exact(a, b):
    """two-sided exact McNemar on paired booleans a (cache follow), b (state follow)."""
    n10 = sum(1 for x, y in zip(a, b) if x and not y); n01 = sum(1 for x, y in zip(a, b) if y and not x); n = n10 + n01
    if n == 0:
        return {"n10": 0, "n01": 0, "p": 1.0}
    k = min(n10, n01); p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return {"n10": n10, "n01": n01, "p": min(1.0, 2 * p)}


def score_model(key, f):
    if f is None:
        return {"status": "pending (missing eval_gcs.json)"}
    o = {"meta": f.get("meta"), "status": "ok"}; why = []
    g = f["gate"]; oo = g.get("official_self")                            # Note 220-3 + addendum: [A1, A2, A3] official-vs-official agreements
    C = min(oo) if oo else None; o_gate = {"C": C, "official_self": oo}
    dacc_ok = abs(g["acc_bs1"] - g["acc_official"]) <= 0.025 + EPS
    if C is not None and C < 0.97 - EPS:                                  # amended G-R (post-failure gate amendment, known-answer, no arm read)
        o_gate["rule"] = "amended (C < 0.97): agree >= C - 0.03"
        if not (dacc_ok and g["agree"] >= C - 0.03 - EPS):
            why.append("G-R amended (agree %.3f vs C - 0.03 = %.3f, |dacc| %.3f)" % (g["agree"], C - 0.03, abs(g["acc_bs1"] - g["acc_official"])))
    else:
        o_gate["rule"] = "registered (agree >= 0.97)" + ("" if C is None else "; C %.3f >= 0.97" % C)
        if not (g["agree"] >= 0.97 - EPS and dacc_ok):
            why.append("G-R (agree %.3f, |dacc| %.3f)" % (g["agree"], abs(g["acc_bs1"] - g["acc_official"])))
    o["G_R"] = o_gate
    if key == "s0_codi20" and abs(g["acc_bs1"] - ACC_CODI20) > 0.025 + EPS:
        why.append("G-R CODI20 acc %.3f vs 0.379" % g["acc_bs1"])
    if key == "s0_zen_gpt2" and abs(g["acc_bs1"] - ACC_ZEN_G1P) > 0.025 + EPS:     # Note 220-2 (b): format gate vs §209 G1'
        why.append("G-R zen format acc %.3f vs G1' 0.4246" % g["acc_bs1"])
    if not f["asserts"].get("cache_rows_ok"):
        why.append("cache-row assertion")
    pairs = f["pairs"]; ish = f["intact_sha"]
    self_ok = all(p["sha"]["self_cache"] == ish[str(p["r"])] and p["sha"]["self_state"] == ish[str(p["r"])] for p in pairs)
    if not self_ok:
        why.append("SELF-SWAP not bitwise")
    reuse = {}
    for p in pairs:
        reuse[p["d"]] = reuse.get(p["d"], 0) + 1
    mx = max(reuse.values()) if reuse else 0
    if mx > MAX_REUSE:
        why.append("donor reuse %d > %d" % (mx, MAX_REUSE))
    if len(pairs) < MIN_PAIRS:
        why.append("n_pairs %d < %d (UNREADABLE)" % (len(pairs), MIN_PAIRS))
    golds = f["golds"]; acc = f["acc"]
    nec = acc["intact"] - acc["mean_z"] >= 0.10 - EPS
    arms = {}
    if pairs:
        for arm in ("cache", "state", "leak"):
            r = rates(pairs, arm, golds); r["floor"] = perm_floor(pairs, arm, golds); r["ci_cluster"] = cluster_ci(pairs, r["_f"]); arms[arm] = r
    o.update({"n_pairs": len(pairs), "max_reuse": mx, "NECESSARY": nec, "nec_gap": acc["intact"] - acc["mean_z"]})
    if why:
        o["label"] = "UNREAD (%s)" % "; ".join(why)
    else:
        c, lk = arms["cache"], arms["leak"]
        ports_follow = c["follow"] >= c["floor"] + 0.30 - EPS
        if not nec:
            lab = "UNNECESSARY"
        elif ports_follow and lk["follow"] <= 0.25 + EPS:
            lab = "CARRIER-PORTS"
        elif ports_follow:
            lab = "CARRIER-LEAK"
        elif c["own"] >= 0.70 - EPS:
            lab = "SCAFFOLD"
        elif c["follow"] < c["floor"] + 0.10:
            lab = "CONTEXT-BOUND"
        else:
            lab = "MIXED"
        o["label"] = lab
        o["GRANULARITY"] = {"cache_minus_state": c["follow"] - arms["state"]["follow"], "mcnemar": mcnemar_exact(c["_f"], arms["state"]["_f"])}
    for r in arms.values():
        r.pop("_f", None)
    o["arms"] = arms
    for k in ("k_sweep", "variants", "prefix", "qdrop", "qmask", "logit_lens", "scit", "graph_matched", "pairs_210_4"):
        if k in f:
            o.setdefault("descriptive", {})[k] = f[k]
    return o


def _with_sidecar(p):
    """eval_gcs.json + the Note 220-3 sidecar gate_official_self.json beside it ({"official_self": [A1, A2, A3]}), if present."""
    f = jl(p)
    sc = jl(os.path.join(os.path.dirname(p), "gate_official_self.json"))
    if f is not None and sc is not None and sc.get("official_self") is not None:
        f["gate"]["official_self"] = sc["official_self"]
    return f


def score(files=FILES):
    out = {"section": 220, "registered": "GCS_v1.md at 3a88762e", "models": {}}
    for k, p in files.items():
        if k == "s1_zen_1b":
            continue
        out["models"][k] = score_model(k, _with_sidecar(p))
    trig = [out["models"][k].get("label") for k in ("s0_codi20", "s0_zen_gpt2")]
    if any(t == "CARRIER-PORTS" for t in trig):
        out["stage1_trigger"] = "met"; out["models"]["s1_zen_1b"] = score_model("s1_zen_1b", _with_sidecar(files["s1_zen_1b"]))
    elif all(t is not None for t in trig):
        out["stage1_trigger"] = "not met"; out["models"]["s1_zen_1b"] = {"status": "UNREAD (stage 1 not triggered; v1 §6: a new Note needed)", "label": "UNREAD"}
    else:
        out["stage1_trigger"] = "pending"; out["models"]["s1_zen_1b"] = {"status": "pending (stage 0 incomplete)"}
    out["primary"] = {st: out["models"][k].get("label", out["models"][k].get("status")) for st, k in PRIMARY.items()}
    return out


# ------------------------------------------------------------------ fixtures
def _mk(cache_f, cache_o, state_f, leak_f, nec_gap=0.2, n=480, agree=0.99, rows_ok=True, self_bad=False, reuse=1, seed=0, acc=0.38):
    rng = np.random.default_rng(seed); golds = {str(i): float(i % 300 + 1) for i in range(1319)}
    pairs = []; ish = {str(i): "s%d" % i for i in range(1319)}
    for k in range(n):
        r = k; d = 1000 + (k // reuse) % 319 if reuse > 1 else (k + 600) % 1319
        if golds[str(d)] == golds[str(r)]:
            d = (d + 1) % 1319
        gd, gr = golds[str(d)], golds[str(r)]
        def pick(pf, po=0.0):
            u = rng.random(); return gd if u < pf else (gr if u < pf + po else -1.0)
        pairs.append({"r": r, "d": d, "out": {"cache": pick(cache_f, cache_o), "state": pick(state_f, 0.5), "leak": pick(leak_f, 0.5)},
                      "sha": {"self_cache": "bad" if (self_bad and k == 0) else ish[str(r)], "self_state": ish[str(r)]}})
    return {"meta": {"stage": 0}, "gate": {"agree": agree, "acc_bs1": acc, "acc_official": acc}, "asserts": {"cache_rows_ok": rows_ok},
            "golds": golds, "acc": {"intact": acc, "mean_z": acc - nec_gap}, "intact_sha": ish, "pairs": pairs}


def fixtures():
    cases = [("ports", dict(cache_f=0.6, cache_o=0.1, state_f=0.04, leak_f=0.05), "CARRIER-PORTS"),
             ("leak", dict(cache_f=0.6, cache_o=0.1, state_f=0.04, leak_f=0.5), "CARRIER-LEAK"),
             ("ports beats high own", dict(cache_f=0.5, cache_o=0.45, state_f=0.04, leak_f=0.05), "CARRIER-PORTS"),
             ("leak beats scaffold", dict(cache_f=0.4, cache_o=0.75 - 0.4 + 0.35, state_f=0.0, leak_f=0.6), "CARRIER-LEAK"),
             ("scaffold over context-bound", dict(cache_f=0.02, cache_o=0.85, state_f=0.04, leak_f=0.02), "SCAFFOLD"),
             ("context-bound", dict(cache_f=0.03, cache_o=0.3, state_f=0.04, leak_f=0.02), "CONTEXT-BOUND"),
             ("mixed", dict(cache_f=0.2, cache_o=0.3, state_f=0.04, leak_f=0.02), "MIXED"),
             ("unnecessary beats high follow", dict(cache_f=0.7, cache_o=0.1, state_f=0.04, leak_f=0.02, nec_gap=0.05), "UNNECESSARY"),
             ("G-R fail", dict(cache_f=0.7, cache_o=0.1, state_f=0.04, leak_f=0.02, agree=0.95), "UNREAD"),
             ("self not bitwise", dict(cache_f=0.7, cache_o=0.1, state_f=0.04, leak_f=0.02, self_bad=True), "UNREAD"),
             ("cache rows", dict(cache_f=0.7, cache_o=0.1, state_f=0.04, leak_f=0.02, rows_ok=False), "UNREAD"),
             ("too few pairs", dict(cache_f=0.7, cache_o=0.1, state_f=0.04, leak_f=0.02, n=250), "UNREAD"),
             ("donor reuse > 3", dict(cache_f=0.7, cache_o=0.1, state_f=0.04, leak_f=0.02, reuse=5), "UNREAD")]
    res = []
    for name, kw, exp in cases:
        got = score_model("fixture", _mk(**kw))["label"]; res.append((name, got.startswith(exp), got))
    a_ = _mk(0.6, 0.1, 0.04, 0.05, agree=0.667); a_["gate"]["official_self"] = [0.70, 0.68, 0.72]
    res.append(("amended G-R pass (C 0.68, agree 0.667 >= 0.65)", score_model("fixture", a_)["label"] == "CARRIER-PORTS", score_model("fixture", a_)["label"]))
    a_["gate"]["official_self"] = [0.80, 0.85, 0.82]
    res.append(("amended G-R fail (C 0.80, agree 0.667 < 0.77)", score_model("fixture", a_)["label"].startswith("UNREAD (G-R amended"), score_model("fixture", a_)["label"]))
    a_["gate"]["official_self"] = [0.98, 0.99, 0.975]
    res.append(("C >= 0.97 -> registered rule, UNREAD", score_model("fixture", a_)["label"].startswith("UNREAD (G-R (agree"), score_model("fixture", a_)["label"]))
    z = _mk(0.6, 0.1, 0.04, 0.05); z["gate"]["acc_bs1"] = z["gate"]["acc_official"] = 0.38
    res.append(("zen format gate fail -> UNREAD", score_model("s0_zen_gpt2", z)["label"].startswith("UNREAD (G-R zen"), score_model("s0_zen_gpt2", z)["label"]))
    z["gate"]["acc_bs1"] = z["gate"]["acc_official"] = 0.43
    res.append(("zen format gate pass", score_model("s0_zen_gpt2", z)["label"] == "CARRIER-PORTS", score_model("s0_zen_gpt2", z)["label"]))
    c = _mk(0.6, 0.1, 0.04, 0.05); c["gate"]["acc_bs1"] = c["gate"]["acc_official"] = 0.43
    res.append(("codi20 0.379 gate fail -> UNREAD", score_model("s0_codi20", c)["label"].startswith("UNREAD (G-R CODI20"), score_model("s0_codi20", c)["label"]))
    import tempfile
    tmp = tempfile.mkdtemp(); fs = {k: os.path.join(tmp, k + ".json") for k in FILES}
    json.dump(_mk(0.03, 0.3, 0.04, 0.02), open(fs["s0_codi20"], "w")); json.dump(_mk(0.03, 0.3, 0.04, 0.02, acc=0.43), open(fs["s0_zen_gpt2"], "w"))
    o = score(fs); res.append(("stage 1 not triggered -> UNREAD", o["models"]["s1_zen_1b"]["label"] == "UNREAD", o["stage1_trigger"]))
    json.dump(_mk(0.6, 0.1, 0.04, 0.05, acc=0.43), open(fs["s0_zen_gpt2"], "w"))
    o = score(fs); res.append(("stage 1 triggered, missing -> pending", o["models"]["s1_zen_1b"]["status"].startswith("pending"), o["stage1_trigger"]))
    os.remove(fs["s0_codi20"]); o = score(fs)
    res.append(("stage 0 incomplete -> pending", o["models"]["s0_codi20"]["status"].startswith("pending") and o["stage1_trigger"] == "met", o["stage1_trigger"]))
    m = mcnemar_exact([True] * 10 + [False] * 10, [False] * 20); res.append(("mcnemar 10/0", abs(m["p"] - 2 / 1024) < 1e-12, m["p"]))
    fl = perm_floor([{"r": 0, "d": 5, "out": {"cache": 1.0}}], "cache", {"0": 2.0, "1": 1.0, "2": 3.0, "3": 1.0, "5": 4.0})
    res.append(("perm floor analytic 2/4", abs(fl - 0.5) < 0.05, fl))
    for r in res:
        print("FIXTURE", "PASS" if r[1] else "FAIL", r[0], r[2])
    ok = sum(r[1] for r in res); print("fixtures %d/%d" % (ok, len(res))); return ok == len(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--fixtures", action="store_true"); a = ap.parse_args()
    if a.fixtures:
        sys.exit(0 if fixtures() else 1)
    o = score(); print(json.dumps(o["primary"]), o["stage1_trigger"])
    json.dump(o, open(OUT, "w", newline=chr(10)), indent=1, default=str)
