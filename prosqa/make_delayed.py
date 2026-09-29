"""Delayed-query ProsQA (PROTOCOL6 addendum J): the prefix holds the statements and the start entity; the candidate
question moves to a new field "query", placed after <|end-latent|> by patch_delayed.py. As in the toy model's delayed
format, the thoughts see the graph and the start node but not which concepts will be asked about.

    python make_delayed.py <in.json> <out.json>

  question: "<statements> Start: Sally."      (was "<statements> Is Sally a hilpus or sterpus?")
  query:    "Is Sally a hilpus or sterpus?"
  steps, answer and the graph fields are unchanged.
"""
import json
import re
import sys


def convert(x):
    parts = re.split(r"(?<=\.) ", x["question"])
    stmts, q = parts[:-1], parts[-1]
    m = re.match(r"^Is (\w+) an? \w+ or \w+\?$", q) or re.match(r"^Is (\w+) a \w+ or \w+\?$", q)
    assert m, q
    root = x["idx_to_symbol"][x["root"]]
    assert m.group(1) == root, (m.group(1), root)
    return {**x, "question": " ".join(stmts) + f" Start: {root}.", "query": q}


if __name__ == "__main__":
    d = json.load(open(sys.argv[1]))
    out = [convert(x) for x in d]
    json.dump(out, open(sys.argv[2], "w"))
    print(f"{sys.argv[1]} -> {sys.argv[2]}: {len(out)} examples; e.g. query {out[0]['query']!r}")
