"""GSM8K shortcut check (Audit's request; the GSM8K track, step 1): how far do no-reasoning heuristics get on the test set?

Heuristics read only the numbers written in the question (no reasoning, no model):
  last / first / max number; sum; product; max - min; last - first; round(mean)
  constant: the most frequent test answer (an optimistic upper bound for any constant predictor, since it peeks at the
            test answers)
  one-op oracle (an upper bound on any one-step pattern rule): correct if ANY of +, -, x, / applied to ANY ordered pair of
            question numbers gives the answer
Answers are compared as numbers (commas removed; exact match after float rounding to 6 decimals).
    python theory/gsm8k_shortcuts.py [--data bench/data/gsm8k_test.jsonl] [--out theory/results/gsm8k_shortcuts.json]
"""
import argparse
import collections
import itertools
import json
import math
import re


def nums(text):
    return [float(x.replace(",", "")) for x in re.findall(r"-?\d[\d,]*\.?\d*", text) if re.search(r"\d", x)]


def gold(ans):
    return float(ans.split("####")[-1].strip().replace(",", ""))


def eq(a, b):
    return a is not None and b is not None and math.isfinite(a) and abs(round(a, 6) - round(b, 6)) < 1e-9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="bench/data/gsm8k_test.jsonl")
    ap.add_argument("--out", default="theory/results/gsm8k_shortcuts.json")
    a = ap.parse_args()
    items = [json.loads(l) for l in open(a.data, encoding="utf-8")]
    Y = [gold(x["answer"]) for x in items]
    Q = [nums(x["question"]) for x in items]
    rules = {
        "last": lambda n: n[-1] if n else None, "first": lambda n: n[0] if n else None,
        "max": lambda n: max(n) if n else None, "sum": lambda n: sum(n) if n else None,
        "product": lambda n: math.prod(n) if n and len(n) < 12 else None,
        "max_minus_min": lambda n: max(n) - min(n) if n else None,
        "last_minus_first": lambda n: n[-1] - n[0] if n else None,
        "round_mean": lambda n: round(sum(n) / len(n)) if n else None,
    }
    res = {k: sum(eq(f(n), y) for n, y in zip(Q, Y)) / len(Y) for k, f in rules.items()}
    most = collections.Counter(Y).most_common(1)[0]
    res["constant_most_frequent_test_answer"] = most[1] / len(Y)
    res["constant_value"] = most[0]

    def one_op(n, y):
        for p, q in itertools.permutations(n, 2):
            for v in (p + q, p - q, p * q, p / q if q else None):
                if eq(v, y):
                    return True
        return False
    res["oracle_one_op_any_pair"] = sum(one_op(n, y) for n, y in zip(Q, Y)) / len(Y)
    res["answer_in_question"] = sum(any(eq(v, y) for v in n) for n, y in zip(Q, Y)) / len(Y)
    res["n_items"] = len(Y)
    json.dump(res, open(a.out, "w", newline="\n"), indent=1)
    for k, v in res.items():
        print("%-36s %s" % (k, ("%.3f" % v) if isinstance(v, float) and k != "constant_value" else v))


if __name__ == "__main__":
    main()
