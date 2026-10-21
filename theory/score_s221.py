"""§221 GSM-CACHE-SWAP-2 scorer (GCS2_v1.md at b7c66d8f; inherits GCS_v1 at 3a88762e + Addendum 1 + Notes 220-1 / 220-2). G-R is replaced by G-FID:
  (a) zen / CODI-1B: official test.py at bs 1 preds vs our port's bs-1 INTACT preds, agree >= 0.97 and |dacc| <= 0.01; CODI20: bitwise determinism
      (a full bs-1 INTACT re-run's token shas == eval_gcs.intact_sha for every item);
  (a') pairs[0:50]: INTACT (recipient), CACHE-SWAP and LEAK-SEG generated-token shas re-run bitwise; any mismatch -> reuse VOID (UNREAD here; full re-run);
  (b) established accuracy (CODI20 0.379, zen 0.4246, +-0.025; recorded fact);
  pinned eval_gcs.json sha256 asserted. G-FID file contract, <run dir>/gate_fid.json:
  {"kind": "official_bs1"|"determinism", "ref": {"pred": {idx: num|None}, "sha": {idx: str} (determinism), "acc": float, "repo_commit", "ckpt_sha256"},
   "subset": {"intact_sha": {r: str}, "cache_sha": {"0".."49": str}, "leak_sha": {"0".."49": str}}, "meta": {...}}
Readings, labels, floors, sealing: as score_s220 (copied).
--- score_s220 docstring follows ---

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
OUT = os.path.join(ROOT, "theory", "results", "score_s221.json")
SEALED_OUT = os.path.join(ROOT, "scratch", "s220", "SEALED_score_s221_arms.json")   # uncommitted; never opened before the seal lifts
FILES = {"s0_codi20": os.path.join(ROOT, "boxes", "runs_fsq", "gsm_codi_20ep_s1", "eval_gcs.json"),
         "s0_zen_gpt2": os.path.join(ROOT, "scratch", "s220", "zen_codi_gpt2", "eval_gcs.json"),
         "s1_zen_1b": os.path.join(ROOT, "scratch", "s220", "zen_codi_llama1b", "eval_gcs.json")}
PRIMARY = {0: "s0_codi20", 1: "s1_zen_1b"}
PIN = {"s0_codi20": "82aa84138eea9a390d713cdb0f09c4f7e78deac295a09519ce2f8d716b6bef36",
       "s0_zen_gpt2": "fec1f9ec1bda121d33485b8a325fb92e963756b3b0ef1f1fd195ba6b512de769"}       # GCS2_v1 §3 (stage 1: re-run, no pin)
FID_KIND = {"s0_codi20": "determinism", "s0_zen_gpt2": "official_bs1", "s1_zen_1b": "official_bs1"}
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
    g = f["gate"]; fid = f.get("_fid"); gf = {}
    if key in PIN and f.get("_sha") != PIN[key]:
        why.append("pinned eval_gcs.json sha mismatch")
    if fid is None:
        return {"status": "pending (gate_fid.json missing; G-FID not run)", "meta": f.get("meta")}
    else:
        kind = FID_KIND.get(key, "official_bs1"); gf["kind"] = kind
        port = f["per_item"]["intact"]["pred"]
        if isinstance(port, list):                                        # s220_gcs.py stores per_item.*.pred as a list in golds-key order (Experiments, dcb189ad)
            gk = list(f["golds"].keys()); assert len(gk) == len(port), "per_item.intact.pred length != golds"
            port = dict(zip(gk, port))
        if kind != fid.get("kind"):
            why.append("G-FID kind %s != %s" % (fid.get("kind"), kind))
        elif kind == "official_bs1":
            ref = fid["ref"]["pred"]; ks = sorted(port, key=int)
            agree = sum(1 for k in ks if (port[k] is None and ref.get(k) is None) or eq(port[k], ref.get(k))) / len(ks)
            gf.update(agree=agree, acc_ref=fid["ref"]["acc"], dacc=abs(g["acc_bs1"] - fid["ref"]["acc"]))
            if fid.get("ref_bf16"):                                       # Note 221-1 item 6: descriptive, never a gate
                rb = fid["ref_bf16"]["pred"]
                gf["bf16_descriptive"] = {"agree_port_fp32": sum(1 for k in ks if (port[k] is None and rb.get(k) is None) or eq(port[k], rb.get(k))) / len(ks),
                                          "acc": fid["ref_bf16"].get("acc")}
            if not (agree >= 0.97 - EPS and gf["dacc"] <= 0.01 + EPS):
                why.append("G-FID(a) (agree %.3f, |dacc| %.3f)" % (agree, gf["dacc"]))
        else:
            ok = all(fid["ref"]["sha"].get(k) == v for k, v in f["intact_sha"].items()); gf["determinism_bitwise"] = ok
            if not ok:
                why.append("G-FID(a) determinism not bitwise")
        sub = fid.get("subset", {}); bad = []
        for i_ in range(50):
            p_ = f["pairs"][i_]
            if sub.get("intact_sha", {}).get(str(p_["r"])) != f["intact_sha"][str(p_["r"])]: bad.append(("intact", i_))
            if sub.get("cache_sha", {}).get(str(i_)) != p_["sha"]["cache"]: bad.append(("cache", i_))
            if sub.get("leak_sha", {}).get(str(i_)) != p_["sha"]["leak"]: bad.append(("leak", i_))
        gf["subset_bitwise"] = not bad; gf["subset_mismatch"] = bad[:10]
        if bad:
            why.append("G-FID(a') reuse VOID (%d mismatches; full re-run required)" % len(bad))
    o["G_FID"] = gf
    if key == "s0_codi20" and abs(g["acc_bs1"] - ACC_CODI20) > 0.025 + EPS:
        why.append("G-FID(b) CODI20 acc %.3f vs 0.379" % g["acc_bs1"])
    if key == "s0_zen_gpt2" and abs(g["acc_bs1"] - ACC_ZEN_G1P) > 0.025 + EPS:     # Note 220-2 (b): format gate vs §209 G1'
        why.append("G-FID(b) zen format acc %.3f vs G1' 0.4246" % g["acc_bs1"])
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
    """eval_gcs.json + its sha256 + the G-FID file gate_fid.json beside it (None if absent)."""
    import hashlib
    f = jl(p)
    if f is not None:
        f["_sha"] = hashlib.sha256(open(p, "rb").read()).hexdigest(); f["_fid"] = jl(os.path.join(os.path.dirname(p), "gate_fid.json"))
    return f


def score(files=FILES):
    out = {"section": 221, "registered": "GCS2_v1.md at b7c66d8f", "models": {}, "provenance": "generated under §220's run, sealed, unread until §221's freeze"}
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
def _mk(cache_f, cache_o, state_f, leak_f, nec_gap=0.2, n=480, rows_ok=True, self_bad=False, reuse=1, seed=0, acc=0.38,
        fid_kind="official_bs1", fid_agree=1.0, fid_dacc=0.0, det_ok=True, sub_bad=0, no_fid=False):
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
                      "sha": {"self_cache": "bad" if (self_bad and k == 0) else ish[str(r)], "self_state": ish[str(r)], "cache": "c%d" % k, "leak": "l%d" % k}})
    port = {str(i): float(i % 7) for i in range(1319)}
    f = {"meta": {"stage": 0}, "gate": {"agree": 0.5, "acc_bs1": acc, "acc_official": acc - 0.1}, "asserts": {"cache_rows_ok": rows_ok},
         "golds": golds, "acc": {"intact": acc, "mean_z": acc - nec_gap}, "intact_sha": ish, "pairs": pairs, "per_item": {"intact": {"pred": port}}, "_sha": "x"}
    nd = int(round((1 - fid_agree) * 1319)); ref = {k: (v if int(k) >= nd else v + 100.0) for k, v in port.items()}
    sub = {"intact_sha": {str(p["r"]): ish[str(p["r"])] for p in pairs[:50]}, "cache_sha": {str(i): "c%d" % i for i in range(50)}, "leak_sha": {str(i): "l%d" % i for i in range(50)}}
    for i in range(sub_bad):
        sub["cache_sha"][str(i)] = "zz"
    f["_fid"] = None if no_fid else {"kind": fid_kind, "ref": {"pred": ref, "acc": acc + fid_dacc, "sha": {k: (v if det_ok else "q") for k, v in ish.items()}}, "subset": sub}
    return f


def fixtures():
    cases = [("ports", dict(cache_f=0.6, cache_o=0.1, state_f=0.04, leak_f=0.05), "CARRIER-PORTS"),
             ("leak", dict(cache_f=0.6, cache_o=0.1, state_f=0.04, leak_f=0.5), "CARRIER-LEAK"),
             ("ports beats high own", dict(cache_f=0.5, cache_o=0.45, state_f=0.04, leak_f=0.05), "CARRIER-PORTS"),
             ("leak beats scaffold", dict(cache_f=0.4, cache_o=0.7, state_f=0.0, leak_f=0.6), "CARRIER-LEAK"),
             ("scaffold over context-bound", dict(cache_f=0.02, cache_o=0.85, state_f=0.04, leak_f=0.02), "SCAFFOLD"),
             ("context-bound", dict(cache_f=0.03, cache_o=0.3, state_f=0.04, leak_f=0.02), "CONTEXT-BOUND"),
             ("mixed", dict(cache_f=0.2, cache_o=0.3, state_f=0.04, leak_f=0.02), "MIXED"),
             ("unnecessary beats high follow", dict(cache_f=0.7, cache_o=0.1, state_f=0.04, leak_f=0.02, nec_gap=0.05), "UNNECESSARY"),
             ("G-FID(a) agree < 0.97", dict(cache_f=0.7, cache_o=0.1, state_f=0.04, leak_f=0.02, fid_agree=0.95), "UNREAD (G-FID(a)"),
             ("G-FID(a) dacc > 0.01", dict(cache_f=0.7, cache_o=0.1, state_f=0.04, leak_f=0.02, fid_dacc=0.02), "UNREAD (G-FID(a)"),
             ("G-FID(a') one mismatch voids reuse", dict(cache_f=0.7, cache_o=0.1, state_f=0.04, leak_f=0.02, sub_bad=1), "UNREAD (G-FID(a')"),
                          ("self not bitwise", dict(cache_f=0.7, cache_o=0.1, state_f=0.04, leak_f=0.02, self_bad=True), "UNREAD"),
             ("cache rows", dict(cache_f=0.7, cache_o=0.1, state_f=0.04, leak_f=0.02, rows_ok=False), "UNREAD"),
             ("too few pairs", dict(cache_f=0.7, cache_o=0.1, state_f=0.04, leak_f=0.02, n=250), "UNREAD"),
             ("donor reuse > 3", dict(cache_f=0.7, cache_o=0.1, state_f=0.04, leak_f=0.02, reuse=5), "UNREAD")]
    res = []
    for name, kw, exp in cases:
        got = score_model("fixture", _mk(**kw))["label"]; res.append((name, got.startswith(exp), got))
    c = _mk(0.6, 0.1, 0.04, 0.05, fid_kind="determinism"); c["_sha"] = PIN["s0_codi20"]; c["gate"]["acc_bs1"] = 0.38
    res.append(("CODI20 determinism pass", score_model("s0_codi20", c)["label"] == "CARRIER-PORTS", score_model("s0_codi20", c)["label"]))
    c2 = _mk(0.6, 0.1, 0.04, 0.05, fid_kind="determinism", det_ok=False); c2["_sha"] = PIN["s0_codi20"]
    res.append(("CODI20 determinism fail", "determinism not bitwise" in score_model("s0_codi20", c2)["label"], score_model("s0_codi20", c2)["label"]))
    c3 = _mk(0.6, 0.1, 0.04, 0.05, fid_kind="determinism")
    res.append(("CODI20 pin mismatch", "pinned eval_gcs.json sha mismatch" in score_model("s0_codi20", c3)["label"], score_model("s0_codi20", c3)["label"]))
    z = _mk(0.6, 0.1, 0.04, 0.05, acc=0.43); z["_sha"] = PIN["s0_zen_gpt2"]
    res.append(("zen pass (official_bs1, format 0.43)", score_model("s0_zen_gpt2", z)["label"] == "CARRIER-PORTS", score_model("s0_zen_gpt2", z)["label"]))
    z2 = _mk(0.6, 0.1, 0.04, 0.05, acc=0.38); z2["_sha"] = PIN["s0_zen_gpt2"]
    res.append(("zen format gate fail", "G-FID(b) zen format" in score_model("s0_zen_gpt2", z2)["label"], score_model("s0_zen_gpt2", z2)["label"]))
    z3 = _mk(0.6, 0.1, 0.04, 0.05, acc=0.43, fid_kind="determinism"); z3["_sha"] = PIN["s0_zen_gpt2"]
    res.append(("zen wrong G-FID kind", "G-FID kind" in score_model("s0_zen_gpt2", z3)["label"], score_model("s0_zen_gpt2", z3)["label"]))
    L = _mk(0.6, 0.1, 0.04, 0.05); L["per_item"]["intact"]["pred"] = [L["per_item"]["intact"]["pred"][k] for k in L["golds"]]
    res.append(("per_item pred as list (real layout) parses + passes", score_model("fixture", L)["label"] == "CARRIER-PORTS", score_model("fixture", L)["label"]))
    L2 = _mk(0.6, 0.1, 0.04, 0.05, fid_agree=0.95); L2["per_item"]["intact"]["pred"] = [L2["per_item"]["intact"]["pred"][k] for k in L2["golds"]]
    res.append(("list layout, agree 0.95 -> UNREAD", score_model("fixture", L2)["label"].startswith("UNREAD (G-FID(a)"), score_model("fixture", L2)["label"]))
    pm = score_model("fixture", _mk(0.7, 0.1, 0.04, 0.02, no_fid=True)); res.append(("G-FID missing -> pending", pm.get("status", "").startswith("pending"), pm.get("status")))
    m = mcnemar_exact([True] * 10 + [False] * 10, [False] * 20); res.append(("mcnemar 10/0", abs(m["p"] - 2 / 1024) < 1e-12, m["p"]))
    for r in res:
        print("FIXTURE", "PASS" if r[1] else "FAIL", r[0], r[2])
    ok = sum(r[1] for r in res); print("fixtures %d/%d" % (ok, len(res))); return ok == len(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--fixtures", action="store_true"); a = ap.parse_args()
    if a.fixtures:
        sys.exit(0 if fixtures() else 1)
    o = score(); print(json.dumps(o["primary"]), o["stage1_trigger"])
    # Audit ruling (10-21): an UNREAD model's arms / descriptive values are SEALED - written only to an uncommitted sidecar (sha printed), never to OUT
    sealed = {}
    for k, m in o["models"].items():
        if str(m.get("label", "")).startswith("UNREAD"):
            sealed[k] = {x: m.pop(x) for x in ("arms", "GRANULARITY", "descriptive", "NECESSARY", "nec_gap") if x in m}
            m["sealed"] = "arms / descriptive values sealed (Audit ruling 10-21; THEORY §220 notes): see SEALED_OUT sha"
    json.dump(o, open(OUT, "w", newline=chr(10)), indent=1, default=str)
    if sealed:
        import hashlib
        b = json.dumps(sealed, indent=1, default=str).encode(); open(SEALED_OUT, "wb").write(b)
        print("SEALED", os.path.relpath(SEALED_OUT, ROOT), hashlib.sha256(b).hexdigest())
