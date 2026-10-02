"""THEORY §78 data (75ec955): the step-count split, frozen by the registration.
    python gsmlen/data78.py      -> gsmlen/d78/{train_short,valid_short,test,aug_long}.json + gsmlen/d78/summary.json
* n = number of non-empty "<<...>>" steps (gsmlen/steps.py). Items keep: question, steps (non-empty), answer, n, depth.
* train_short: GSM8K-Aug train with n in {1, 2}. valid_short: GSM8K-Aug valid with n in {1, 2} (checkpoint selection).
* test: the GSM8K test (GSM8K-Aug's test file = the human test, identical questions and calculator annotations; checked
  1,319/1,319 in gsmlen prep), with bin = "n0" (n = 0), "id" (1-2), "mid" (3-4), "long" (>= 5); n >= 6 flagged.
* aug_long (secondary): GSM8K-Aug train items with n >= 5 (none of them in train_short by construction), dropped if
  (a) any equation's arithmetic does not recompute (|eval(expr) - value| > 1e-6 * max(1, |value|), or unparsable), or
  (b) question 8-gram Jaccard > 0.5 with any train_short item (word 8-grams, lower-case, numbers masked).
* depth (DAG depth): equation k depends on an earlier equation j if expr_k contains value_j as a number token; depth =
  1 + the longest dependency chain (an equation with no dependency has depth 1; the final answer is one more level, so
  n = 0 has depth 1). Fixed with Theory 2026-10-02 11:45.
"""
import collections
import json
import os
import re

D = "D:/ext/coconut/data"; OUT = "gsmlen/d78"
EQ = re.compile(r"<<([^<>=]*)=([^<>]*)>>")
NUM = re.compile(r"\d+(?:\.\d+)?")


def clean(x):
    st = [s.strip() for s in x["steps"] if s.strip()]
    return {"question": x["question"], "steps": st, "answer": x["answer"].strip(), "n": len(st)}


def numkey(s):
    try:
        f = float(s.replace(",", "")); return repr(round(f, 6))
    except ValueError:
        return s


def depth(steps):
    vals, dep = [], []
    for s in steps:
        m = EQ.search(s)
        if not m:
            dep.append(1); vals.append(None); continue
        nums = {numkey(t) for t in NUM.findall(m.group(1))}
        d = 1 + max([dep[j] for j, v in enumerate(vals) if v is not None and v in nums], default=0)
        dep.append(d); vals.append(numkey(m.group(2).strip()))
    return 1 + max(dep, default=0)                                 # the final answer is one level


def arith_ok(steps):
    for s in steps:
        m = EQ.fullmatch(s.strip())
        if not m:
            return False
        e, v = m.group(1).replace(",", ""), m.group(2).replace(",", "").strip()
        if not re.fullmatch(r"[0-9+\-*/(). ]+", e):
            return False
        try:
            r, w = eval(e, {"__builtins__": {}}), float(v)          # noqa: S307 (digits and operators only)
        except Exception:
            return False
        if abs(r - w) > 1e-6 * max(1.0, abs(w)):
            return False
    return True


def shingles(q, k=8):
    w = re.sub(r"[^a-z# ]", " ", re.sub(r"\d+(\.\d+)?", "#", q.lower())).split()
    return {" ".join(w[i:i + k]) for i in range(max(1, len(w) - k + 1))}


def main():
    os.makedirs(OUT, exist_ok=True)
    tr = [clean(x) for x in json.load(open(f"{D}/gsm_train.json", encoding="utf-8"))]
    va = [clean(x) for x in json.load(open(f"{D}/gsm_valid.json", encoding="utf-8"))]
    te = [clean(x) for x in json.load(open(f"{D}/gsm_test.json", encoding="utf-8"))]
    for x in tr + va + te:
        x["depth"] = depth(x["steps"])
    short = [x for x in tr if x["n"] in (1, 2)]
    vshort = [x for x in va if x["n"] in (1, 2)]
    for i, x in enumerate(te):
        x["idx"] = i; x["bin"] = "n0" if x["n"] == 0 else "id" if x["n"] <= 2 else "mid" if x["n"] <= 4 else "long"
        x["n_ge6"] = x["n"] >= 6
    cand = [x for x in tr if x["n"] >= 5]
    ok = [x for x in cand if arith_ok(x["steps"])]
    idx = collections.defaultdict(set); sh_short = []
    for i, x in enumerate(short):
        s = shingles(x["question"]); sh_short.append(s)
        for g in s:
            idx[g].add(i)
    keep, dup = [], 0
    for x in ok:
        s = shingles(x["question"]); c = collections.Counter()
        for g in s:
            if len(idx.get(g, ())) < 5000:
                c.update(idx.get(g, ()))
        best = max((len(s & sh_short[i]) / len(s | sh_short[i]) for i, _ in c.most_common(20)), default=0.0)
        if best > 0.5:
            dup += 1
        else:
            keep.append(x)
    for name, d in (("train_short", short), ("valid_short", vshort), ("test", te), ("aug_long", keep)):
        json.dump(d, open(f"{OUT}/{name}.json", "w", encoding="utf-8", newline="\n"))
    summ = {"train_short": len(short), "valid_short": len(vshort),
            "test_bins": dict(collections.Counter(x["bin"] for x in te)), "test_n_ge6": sum(x["n_ge6"] for x in te),
            "test_depth": dict(sorted(collections.Counter(x["depth"] for x in te).items())),
            "aug_long_candidates": len(cand), "aug_long_bad_arith": len(cand) - len(ok), "aug_long_dup_dropped": dup,
            "aug_long": len(keep)}
    json.dump(summ, open(f"{OUT}/summary.json", "w", newline="\n"), indent=1); print(summ)


if __name__ == "__main__":
    main()
