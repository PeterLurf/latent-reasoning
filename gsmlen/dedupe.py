"""GSM8K length split prep (Literature 56d524d): near-duplicates between the short training split (GSM8K-Aug train,
step count 1..--short_max) and the test set (all 1,319, flagged by step count). Questions are lower-cased, numbers
replaced by "#", punctuation stripped; shingles = word 8-grams. A test item's score against a train item = shared
shingles / test shingles; reported: per test item the max score, counts above 0.5 / 0.8, and exact normalised matches.
Writes gsmlen/dedupe.json {test index: [max score, train index]} for scores >= 0.5.
    python gsmlen/dedupe.py [--short_max 2]
"""
import argparse
import collections
import json
import re


def shingles(q, k=8):
    w = re.sub(r"[^a-z# ]", " ", re.sub(r"\d+(\.\d+)?", "#", q.lower())).split()
    return {" ".join(w[i:i + k]) for i in range(max(1, len(w) - k + 1))}, " ".join(w)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--short_max", type=int, default=2); a = ap.parse_args()
    D = "D:/ext/coconut/data"; cnt = json.load(open("gsmlen/step_counts.json"))
    train = json.load(open(f"{D}/gsm_train.json", encoding="utf-8")); test = json.load(open(f"{D}/gsm_test.json", encoding="utf-8"))
    idx = collections.defaultdict(list); exact = {}
    for i, (x, n) in enumerate(zip(train, cnt["train"])):
        if 1 <= n <= a.short_max:
            sh, full = shingles(x["question"]); exact.setdefault(full, i)
            for s in sh:
                idx[s].append(i)
    out = {}; rows = []
    for j, x in enumerate(test):
        sh, full = shingles(x["question"]); c = collections.Counter()
        for s in sh:
            l = idx.get(s, ())
            if len(l) < 5000:                                  # skip boilerplate shingles shared by thousands of items
                c.update(l)
        best, bi = (max(((v, i) for i, v in c.items()), default=(0, -1)))
        score = best / len(sh); rows.append((cnt["test"][j], score, full in exact))
        if score >= 0.5:
            out[j] = [round(score, 3), bi]
    json.dump(out, open("gsmlen/dedupe.json", "w", newline="\n"), indent=1)
    for name, sel in (("all test", lambda n: True), ("test n>=4", lambda n: n >= 4), ("test n>=5", lambda n: n >= 5)):
        r = [x for x in rows if sel(x[0])]
        print(f"{name}: {len(r)} items | max-score >= 0.5: {sum(x[1] >= 0.5 for x in r)} | >= 0.8: {sum(x[1] >= 0.8 for x in r)}"
              f" | exact normalised match: {sum(x[2] for x in r)}")


if __name__ == "__main__":
    main()
