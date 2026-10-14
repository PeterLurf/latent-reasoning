"""§200 GSM8K-AUG PILOT prep (CPU, local, read-only on the data; idle priority): the small data files the harness ships with.
  bench/data/gsm8k_aug/s200_names.json  : R_nu's whitelist pool - capitalised non-initial words of Aug train (audit_aug's CAP regex, STOP removed), kept if
                                          (a) >= 20 single-name items, (b) pronoun share >= 0.85 for one gender over those items (he / him / his / himself
                                          vs she / her / hers / herself), (c) lower-case occurrences <= 2% of capitalised (drops Bill / Mark / Rose ...);
                                          the top 250 per gender by item count.
  bench/data/gsm8k_aug/s200_test_meta.json: per GSM8K test item (gsm8k_aug/test.txt order): m, struct group (seen / unseen / nokey: theory/gsmtpl
                                          seen_split.py's fingerprint vs Aug train), dedupe (gsmlen/dedupe.py's 8-gram method vs ALL Aug train: max score,
                                          flagged >= 0.5); the FILLER-N strings (fixed neutral prose, no digits, no capitalised names, exactly 16 / 32 GPT-2
                                          tokens) and a sample of resampled TRAIN draws scanned against test (Audit).
    python -I scratch/s200_prep.py"""
import collections
import ctypes
import json
import os
import re
import sys

ROOT = "D:/Latent"; sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/theory/gsmtpl")
try:
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x40)
except Exception:
    pass
import numpy as np  # noqa: E402
import tiktoken  # noqa: E402

import audit_aug as AA  # noqa: E402
import extra_stats as ES  # noqa: E402
import boxes.gsm8k_data as G  # noqa: E402

ENC = tiktoken.get_encoding("gpt2"); OUT = G.AUG
MALE = re.compile(r"\b(he|him|his|himself)\b", re.I); FEMALE = re.compile(r"\b(she|her|hers|herself)\b", re.I)
FILL16 = "The morning was calm and quiet, and the road ran along the river."
FILL32 = FILL16 + " Birds sang in the tall trees while a soft wind moved slowly over the hills."   # k32 extends k16


def names(R):
    cap_items = collections.defaultdict(set); low = collections.Counter(); capc = collections.Counter()
    for i, r in enumerate(R):
        q = r["q"]
        for w in AA.CAP.findall(q):
            if w not in AA.STOP:
                cap_items[w].add(i)
    words = set(cap_items)
    for r in R:
        for w in re.findall(r"\b[A-Za-z]{3,}\b", r["q"]):
            if w[0].islower() and w.capitalize() in words:
                low[w.capitalize()] += 1
            elif w in words:
                capc[w] += 1
    item_names = collections.defaultdict(set)
    for w, its in cap_items.items():
        for i in its:
            item_names[i].add(w)
    stats = {}
    for w, its in cap_items.items():
        single = [i for i in its if len(item_names[i]) == 1]
        if len(single) < 20:
            continue
        mm = sum(bool(MALE.search(R[i]["q"])) and not FEMALE.search(R[i]["q"]) for i in single); ff = sum(bool(FEMALE.search(R[i]["q"])) and not MALE.search(R[i]["q"]) for i in single)
        if mm + ff == 0:
            continue
        stats[w] = {"items": len(its), "single": len(single), "male": mm, "female": ff, "lower_ratio": low[w] / max(1, capc[w])}
    pool = {}
    for gd, k in (("male", "male"), ("female", "female")):
        c = [w for w, s in stats.items() if s[k] / (s["male"] + s["female"]) >= 0.85 and s["lower_ratio"] <= 0.02]
        pool[gd] = sorted(c, key=lambda w: -stats[w]["items"])[:250]
    return pool, stats


def main():
    R, cnt = G.load_aug("train"); pool, stats = names(R)
    json.dump({"male": pool["male"], "female": pool["female"], "stats": {w: stats[w] for w in pool["male"] + pool["female"]},
               "rule": "capitalised non-initial word (audit_aug.CAP, STOP removed); >= 20 single-name items; one gender's pronoun share >= 0.85; lower-case "
                       "occurrences <= 2% of capitalised; top 250 per gender by item count", "filter_counts": cnt},
              open(os.path.join(OUT, "s200_names.json"), "w", newline="\n", encoding="utf-8"), indent=1)
    print("names", len(pool["male"]), len(pool["female"]), pool["male"][:15], pool["female"][:15], flush=True)
    T, _ = G.load_aug("test", filter_bad=False); assert len(T) == 1319
    rng = __import__("random").Random(7); V1 = [rng.choice([3, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43]) * 1.0 + rng.random() for _ in range(40)]
    V2 = [rng.uniform(2, 50) for _ in range(40)]                                    # seen_split.py's draws

    def fp(q, e, a):
        it = AA.analyse(q, e, a); x, y = ES.fingerprint(it["skel"], V1), ES.fingerprint(it["skel"], V2)
        return None if x is None or y is None else (x, y, len(it["st"]))
    Qa, Ea, Aa = AA.load_aug("train"); FA = set(fp(q, e, a) for q, e, a in zip(Qa, Ea, Aa)); FA.discard(None)
    Qt, Et, At = AA.load_openai("gsm8k_test"); assert [q.strip() for q in Qt] == [t["q"] for t in T], "§200: Aug test == GSM8K test order"
    grp = []
    for q, e, a in zip(Qt, Et, At):
        f = fp(q, e, a); grp.append("nokey" if f is None else ("seen" if f in FA else "unseen"))
    sh = lambda q: {" ".join(w[i:i + 8]) for w in [re.sub(r"[^a-z# ]", " ", re.sub(r"\d+(\.\d+)?", "#", q.lower())).split()] for i in range(max(1, len(w) - 7))}   # noqa: E731
    idx = collections.defaultdict(list)
    for i, q in enumerate(Qa):
        for s in sh(q):
            idx[s].append(i)
    ded = []
    for q in Qt:
        S = sh(q); c = collections.Counter()
        for s in S:
            l_ = idx.get(s, ())
            if len(l_) < 5000:
                c.update(l_)
        best = max(c.values(), default=0); ded.append(round(best / len(S), 4))
    f16, f32 = len(ENC.encode(FILL16 + " ")), len(ENC.encode(FILL32 + " "))
    assert f16 == 16 and f32 == 32, ("filler token counts", f16, f32)
    assert not re.search(r"\d", FILL16 + FILL32) and not [w for w in AA.CAP.findall(FILL32) if w not in AA.STOP], "filler: digits / names"
    meta = {"m": [len(t["eqs"]) for t in T], "struct": grp, "dedupe_score": ded, "dedupe_flag": [s >= 0.5 for s in ded],
            "filler": {"k16": FILL16 + " ", "k32": FILL32 + " "},
            "def": {"struct": "theory/gsmtpl/seen_split.py's functional fingerprint (extra_stats.fingerprint at two random value vectors + m) seen in Aug train (unfiltered)",
                    "dedupe": "gsmlen/dedupe.py: lower-case, numbers -> #, word 8-grams; score = max over Aug train items of shared / test shingles (shingles in >= 5000 "
                              "items skipped); flagged >= 0.5", "filler": "prepended to the question; exactly 16 / 32 GPT-2 tokens incl. the trailing space"}}
    # Audit: resampled TRAIN draws scanned against test too (a sample)
    names_pool = G.load_names(os.path.join(OUT, "s200_names.json")); rr = np.random.default_rng(20001); tq = set(Qt)
    samp = [R[i] for i in rr.permutation(len(R))[:20000]]; drawn = []
    for r in samp:
        y, _ = G.relabel_names(r, names_pool, rr); tp, _ = G.parse_recomputable(y)
        if tp is not None:
            y, _ = G.resample(y, tp, rr, test_q=tq)
        drawn.append(y["q"])
    hit = 0; mx = []
    tidx = collections.defaultdict(list)
    for j, q in enumerate(Qt):
        for s in sh(q):
            tidx[s].append(j)
    for q in drawn:
        S = sh(q); c = collections.Counter()
        for s in S:
            c.update(tidx.get(s, ()))
        b = max(c.values(), default=0) / len(S); mx.append(b); hit += b >= 0.5
    meta["resampled_draw_scan"] = {"n_draws": len(drawn), "n_ge_0.5_vs_test": hit, "max_score": max(mx), "exact_test_question": sum(q in tq for q in drawn),
                                   "def": "20k train items (rng 20001), each relabelled + resampled once; score = shared 8-grams / draw 8-grams vs the best test item"}
    json.dump(meta, open(os.path.join(OUT, "s200_test_meta.json"), "w", newline="\n", encoding="utf-8"), indent=1)
    print("struct", collections.Counter(grp), "dedupe flagged", sum(meta["dedupe_flag"]), "draw scan", meta["resampled_draw_scan"], flush=True)


if __name__ == "__main__":
    main()
