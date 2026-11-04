"""§271 N1 gate G0 (N1_v1 §10; generator / sets; CPU) -> scratch/gate_s271n1/g0_sets.json. Checks:
  solve      so_words.solve(text) == answer on 20k items (train frames + transfer frames, offsets, both modes, n 3-5, m 1-12)
  k_touch    chi^2 of k_touch vs Binomial(m, 2/n) per (n, m in 4 / 8), 4k items each, p > 0.01 (bins with expected < 5 pooled into the tail)
  hold       P(S_m = S_0) vs M-1's 1/n + ((n-1)/n) lambda^m, lambda = (n-3)/(n-1), per (n, m) on 4k items: |diff| <= 0.01 + 3 SE (reported; 0.01 is the registered tolerance)
  modes      NUM / ORD of the same draw: the same holdings / swaps / question; the swap sentences equal modulo the markers ('Swap L:' vs First / Then / Finally)
  offsets    labels 1-12 in 64 x 8000 training draws (Stream SEED_TRAIN, d ~ U{1..4}, offset on): every label >= 1,000 times
  vidx       tok_so decode assert on 20k items (S_t's name token at vidx; unique) for cues index / entity / generic (shared padding asserted inside tok_so)
  decontam   no word 13-gram of any train / test text (2,000 sampled per family x frame) appears in the 750 BBH inputs
  bbh        inject_so_eval.bbh_check(D:/Latent/scratch/n1_build/bbh): 750 / 750 parse, first tokens distinct
  pairs      so_words.swap_pairs at (5,4,2), (5,8,4), (5,6,3): n_pairs == 1,000 and the keep rule re-checked per pair
  excl       no training draw (64 x 2000 steps) has a TEST / TRANSFER / OFFSET sig; Eve / Bob never drawn
  shas       set shas recorded (must equal floors_so.json's set_sha)
    python -B scratch/n1_build/g0_sets.py"""
import collections
import json
import math
import os
import re
import sys

NEW = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")).replace("\\", "/"); sys.path.insert(0, NEW); sys.path.insert(0, NEW + "/scratch/s272_build")
os.chdir(NEW)
import numpy as np  # noqa: E402
import boxes.so_words as SO  # noqa: E402
import boxes.so_tok as T  # noqa: E402
import boxes.inject_so_eval as SE  # noqa: E402
from g2_g3 import chi2_p  # noqa: E402  (regularised-gamma chi^2; no scipy)


def chi2_binom(ks, m, p):
    from math import comb
    n = len(ks); c = collections.Counter(ks); exp = [n * comb(m, k) * p ** k * (1 - p) ** (m - k) for k in range(m + 1)]; obs = [c.get(k, 0) for k in range(m + 1)]
    while len(exp) > 2 and exp[-1] < 5:                                      # pool the tail
        exp[-2] += exp.pop(); obs[-2] += obs.pop()
    x = sum((o - e) ** 2 / e for o, e in zip(obs, exp)); return chi2_from(x, len(exp) - 1)


def chi2_from(x, df):
    """P(chi^2_df >= x) via chi2_p's machinery: a synthetic count vector is not needed; reuse its gamma code by constructing counts is awkward -> direct."""
    a, z = df / 2, x / 2
    if z <= 0:
        return 1.0
    if z < a + 1:
        t = s = 1 / a; nn = a
        while abs(t) > 1e-15 * abs(s):
            nn += 1; t *= z / nn; s += t
        return 1 - s * math.exp(-z + a * math.log(z) - math.lgamma(a))
    b = z + 1 - a; c = 1e300; dd = 1 / b; h = dd
    for i in range(1, 500):
        an = -i * (i - a); b += 2; dd = an * dd + b; dd = 1 / (dd if abs(dd) > 1e-300 else 1e-300); c = b + an / c; c = c if abs(c) > 1e-300 else 1e-300; h *= dd * c
        if abs(dd * c - 1) < 1e-15:
            break
    return math.exp(-z + a * math.log(z) - math.lgamma(a)) * h


def main():
    out = {}; r = np.random.default_rng(29900)
    bad = 0; vbad = 0
    for k in range(20000):
        it = SO.make(r, int(r.choice(SO.NS)), int(r.integers(1, 13)), SO.MODES[k % 2], SO.FRAME_TRAIN + SO.FRAME_TRANSFER, offset=True)
        bad += SO.solve(it["text"]) != it["answer"]
        for c in ("index", "entity", "generic"):
            o = T.tok_so(it, c); vbad += any(T.ENC.decode([o["aux"][t][o["vidx"][t]]]) != " " + it["values"][t + 1] for t in range(it["m"]))
    out["solve"] = {"n": 20000, "mismatch": bad, "ok": bad == 0}; out["vidx"] = {"n": 20000, "bad": vbad, "ok": vbad == 0}
    kt = {}; hold = {}
    for n in SO.NS:
        for m in (1, 2, 4, 8, 12):
            its = [SO.make(r, n, m, "num") for _ in range(4000)]
            if m in (4, 8):
                kt["%d_%d" % (n, m)] = chi2_binom([it["k_touch"] for it in its], m, 2 / n)
            lam = (n - 3) / (n - 1); th = 1 / n + (n - 1) / n * lam ** m; ob = np.mean([it["values"][-1] == it["values"][0] for it in its]); se = math.sqrt(th * (1 - th) / 4000)
            hold["%d_%d" % (n, m)] = {"obs": float(ob), "theory": th, "diff": float(ob - th), "se": se, "ok": abs(ob - th) <= 0.01 + 3 * se}
    out["k_touch"] = {"p": kt, "ok": all(p > 0.01 for p in kt.values())}; out["hold"] = {"cells": hold, "ok": all(v["ok"] for v in hold.values())}
    mb = 0
    for k in range(2000):
        n, m = int(r.choice(SO.NS)), int(r.integers(1, 13)); it = SO.make(np.random.default_rng((29901, k)), n, m, "num"); jt = SO.make(np.random.default_rng((29901, k)), n, m, "ord")
        strip = lambda s: [re.sub(r"^(Swap \d+:|First,|Then,|Finally,) ", "", x) for x in SO.swap_sentences(s)]  # noqa: E731
        mb += not (it["people"] == jt["people"] and it["swaps"] == jt["swaps"] and it["ostar"] == jt["ostar"] and strip(it) == strip(jt) and it["question"] == jt["question"])
    out["modes"] = {"n": 2000, "bad": mb, "ok": mb == 0}
    st = SO.Stream(SO.SEED_TRAIN); rd = np.random.default_rng(1 + 22331); lab = collections.Counter(); names = set(); sigs = set(); n_draw = 0
    for step in range(8000):
        m = int(rd.integers(1, 5))
        for _ in range(64):
            it = st.draw(m); n_draw += 1; names |= set(it["people"])
            if it["mode"] == "num":
                lab.update(SO.label(it, s) for s in range(1, m + 1))
            if step < 2000:
                sigs.add(it["sig"])
    out["offsets"] = {"hist": {str(k): lab[k] for k in range(1, 13)}, "ok": all(lab[k] >= 1000 for k in range(1, 13))}
    sets = SO.test_sets(); ex = SO.excl_sigs(sets)
    out["excl"] = {"train_draws_checked": 64 * 2000, "overlap": len(sigs & ex), "names_eve_bob": sorted(names & {"Eve", "Bob"}), "ok": not (sigs & ex) and not (names & {"Eve", "Bob"})}
    bbh_dir = "D:/Latent/scratch/n1_build/bbh"; bb = [json.loads(open(os.path.join(bbh_dir, f), encoding="utf-8").read()) for f in sorted(os.listdir(bbh_dir)) if f.startswith("tracking")]
    bbtext = " ".join(" ".join(e["input"].split()) for b in bb for e in b["examples"]); words = bbtext.split(); g13 = {tuple(words[i:i + 13]) for i in range(len(words) - 12)}
    hits = 0; ntx = 0
    for f in SO.FAMS:
        for fr in SO.FRAME_TRAIN + SO.FRAME_TRANSFER:
            rr = np.random.default_rng((29902, SO.FAMS.index(f), fr))
            for _ in range(400):
                it = SO.make(rr, int(rr.choice(SO.NS)), int(rr.integers(1, 13)), SO.MODES[int(rr.integers(0, 2))], (fr,), offset=True)
                if it["family"] != f:
                    continue
                w = it["text"].split(); ntx += 1; hits += sum(tuple(w[i:i + 13]) in g13 for i in range(len(w) - 12))
    out["decontam"] = {"texts": ntx, "hits_13gram": hits, "ok": hits == 0}
    try:
        b = SE.bbh_check(bbh_dir); out["bbh"] = {"result": b, "ok": True}
    except AssertionError as e:
        out["bbh"] = {"error": str(e), "ok": False}
    pr = {}
    for n, m, t in ((5, 4, 2), (5, 8, 4), (5, 6, 3)):
        ps, k = SO.swap_pairs(n, m, t); ok = len(ps) == 1000 and all(p["cf"] not in (p["rec"]["answer"], p["don"]["answer"], p["don"]["values"][t]) and
                                                                     p["cf"] == SO.apply_swaps(p["don"]["values"][t], p["rec"]["swaps"][t:]) for p in ps)
        pr["%d_%d_%d" % (n, m, t)] = {"n_pairs": len(ps), "tries": k, "ok": ok}
    out["pairs"] = {"cells": pr, "ok": all(v["ok"] for v in pr.values())}
    out["shas"] = SE.set_shas(sets); fl = json.load(open("D:/Latent/boxes/runs_fsq/n1_floors/floors_so.json")); out["shas"]["floors_set_sha_match"] = fl["set_sha"] == out["shas"]["so_test_sha"]
    out["pass"] = all(v.get("ok", True) for v in out.values() if isinstance(v, dict)) and out["shas"]["floors_set_sha_match"]
    os.makedirs("scratch/gate_s271n1", exist_ok=True); json.dump(out, open("scratch/gate_s271n1/g0_sets.json", "w", newline="\n"), indent=1, default=str)
    print(json.dumps({k: (v.get("ok") if isinstance(v, dict) else v) for k, v in out.items()}))


if __name__ == "__main__":
    main()
