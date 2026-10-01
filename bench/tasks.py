"""Task registry for the reasoning benchmark harness.

A task turns a data file into a list of items with one schema, so any model adapter can be run on any task it supports:
    {"id": str, "kind": "choice" | "generate", "depth": int | None, "answer": str, "raw": <original record>}
Choice items are scored by the adapter's margin (log p(correct) - log p(other) > 0); generate items by exact match of the
normalised final answer. Each task names its source, how depth is defined, and its no-reasoning baselines, which every
report prints next to model scores (ProsQA's proximity rule scores 0.91 with no reasoning: THEORY §57).

Add a task: write a loader returning items and register it in TASKS with `Task(...)`.
"""
import collections
import hashlib
import json
import os
import re
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "bench", "data")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fetch(url, name):
    """Download once into bench/data/ (gitignored); return the local path."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, name)
    if not os.path.exists(path):
        tmp = path + ".part"
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, path)
    return path


# ---------------------------------------------------------------- ProsQA family (graph reachability, two options)

def _bfs(adj, srcs):
    d = {s: 0 for s in srcs}; q = deque(srcs)
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in d:
                d[v] = d[u] + 1; q.append(v)
    return d


def prosqa_proximity(x):
    """THEORY §57's no-reasoning rule: answer the option farther from every person named in the problem.
    Returns 1 (right), 0 (wrong) or 0.5 (tie)."""
    persons = [i for i, s in enumerate(x["idx_to_symbol"]) if s[:1].isupper()]
    adj = collections.defaultdict(list)
    for a, b in x["edges"]:
        adj[a].append(b)
    dp = [_bfs(adj, [p]) for p in persons]
    dt = min(d.get(x["target"], 99) for d in dp)
    dn = min(d.get(x["neg_target"], 99) for d in dp)
    return 1.0 if dn < dt else 0.0 if dn > dt else 0.5


def prosqa_loader(rel, depth_min=None, depth_max=None):
    def load():
        path = os.path.join(ROOT, rel)
        items = []
        for i, x in enumerate(json.load(open(path))):
            if isinstance(x.get("edges"), str):
                x = {**x, "edges": json.loads(x["edges"])}
            d = len(x["steps"])
            if (depth_min and d < depth_min) or (depth_max and d > depth_max):
                continue
            items.append({"id": f"{os.path.basename(rel)}:{i}", "kind": "choice", "depth": d, "answer": x["answer"], "raw": x})
        return items, [path]
    return load


# ---------------------------------------------------------------- GSM8K (Cobbe et al. 2110.14168), test split

GSM8K_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"


def gsm8k_load():
    path = fetch(GSM8K_URL, "gsm8k_test.jsonl")
    items = []
    for i, line in enumerate(open(path, encoding="utf-8")):
        x = json.loads(line)
        sol, ans = x["answer"].split("####")
        steps = len(re.findall(r"<<[^>]*>>", sol))                # calculator annotations = equation steps
        items.append({"id": f"gsm8k:{i}", "kind": "generate", "depth": steps, "answer": norm_number(ans),
                      "raw": {"question": x["question"], "solution": sol.strip(), "final": ans.strip()}})
    return items, [path]


# No-reasoning GSM8K baselines (THEORY's shortcut check, theory/gsm8k_shortcuts.py; Audit's request). They read only the
# numbers written in the question; answers compared as numbers (commas removed, rounded to 6 decimals).
def _gsm_nums(text):
    return [float(x.replace(",", "")) for x in re.findall(r"-?\d[\d,]*\.?\d*", text) if re.search(r"\d", x)]


def _gsm_eq(a, b):
    import math
    return a is not None and b is not None and math.isfinite(a) and abs(round(a, 6) - round(b, 6)) < 1e-9


def _gsm_gold(r):
    return float(r["final"].replace(",", ""))


def gsm_last(r):
    n = _gsm_nums(r["question"]); return float(bool(n) and _gsm_eq(n[-1], _gsm_gold(r)))


def gsm_const5(r):
    return float(_gsm_eq(5.0, _gsm_gold(r)))


def gsm_in_question(r):
    return float(any(_gsm_eq(v, _gsm_gold(r)) for v in _gsm_nums(r["question"])))


def gsm_one_op(r):
    import itertools
    y = _gsm_gold(r)
    for p, q in itertools.permutations(_gsm_nums(r["question"]), 2):
        for v in (p + q, p - q, p * q, p / q if q else None):
            if _gsm_eq(v, y):
                return 1.0
    return 0.0


GSM_BASE = {"last number in question": gsm_last, "constant 5 (most frequent test answer; peeks)": gsm_const5,
            "answer appears in question (oracle)": gsm_in_question, "one-op oracle (any pair, + - x /)": gsm_one_op}


def norm_number(s):
    s = s.strip().replace(",", "").replace("$", "")
    m = re.findall(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return s
    v = m[-1]
    return v[:-2] if v.endswith(".0") else v


# ---------------------------------------------------------------- Multi-digit multiplication (Deng et al. 2311.01460)
# Format of the implicit-CoT repo: digits least-significant first, "a * b||<steps> #### <product>".

MULT_URL = "https://raw.githubusercontent.com/da03/implicit_chain_of_thought/main/data/{n}_by_{n}_mult/test_bigbench.txt"


def mult_loader(n):
    def load():
        path = fetch(MULT_URL.format(n=n), f"mult_{n}x{n}_test.txt")
        items = []
        for i, line in enumerate(open(path, encoding="utf-8")):
            line = line.strip()
            if not line:
                continue
            q, rest = line.split("||")
            steps, ans = rest.split("####")
            items.append({"id": f"mult{n}x{n}:{i}", "kind": "generate", "depth": n, "answer": " ".join(ans.split()),
                          "raw": {"question": q.strip(), "steps": steps.strip()}})
        return items, [path]
    return load


def norm_digits(s):
    return " ".join(re.findall(r"\d", s))


# ---------------------------------------------------------------- Boxes state tracking (THEORY §66), generator-built

def boxes_load():
    import random as _r
    import sys as _s
    _s.path.insert(0, ROOT)
    from boxes.core import apply, make_problem, truth
    rng = _r.Random(6600); items = []
    for n in range(1, 33):
        for qt in ("all", "where"):
            for i in range(50):
                p = make_problem(rng, n, qt)
                items.append({"id": f"boxes:{n}:{qt}:{i}", "kind": "generate", "depth": n, "answer": truth(p), "raw": p})
    return items, []


def _boxes_ans(p, arr):
    return (" " + " ".join(arr)) if p["qtype"] == "all" else " " + str(arr.index(p["target"]) + 1)


def boxes_identity(p):
    from boxes.core import truth
    return float(_boxes_ans(p, p["start"]).strip() == truth(p).strip())


def boxes_lastop(p):
    from boxes.core import apply, truth
    return float(_boxes_ans(p, apply(p["start"], p["ops"][-1:])).strip() == truth(p).strip())


def boxes_norm(s):
    """THEORY §66 amendment 66-1: compare only the content. A full arrangement -> its letters A-D in order ("A,C,B,D",
    " A C B D", "ACBD" all -> "A C B D"); a "where" answer -> its box digit 1-4."""
    t = s.strip()
    if re.fullmatch(r"[A-Da-d]{4}", t):
        return " ".join(t.upper())
    letters = re.findall(r"\b[A-Da-d]\b", t)                   # standalone letters only ("box" is not a B)
    if letters:
        return " ".join(c.upper() for c in letters)
    d = re.findall(r"[1-4]", s)
    return d[0] if d else s.strip()


def boxes_chance(p):
    return 1 / 24 if p["qtype"] == "all" else 1 / 4


def boxes_majority(p):
    from boxes.core import truth
    maj = "ABCD" if p["qtype"] == "all" else None                # "all": the identity-ordered arrangement; "where": box 1
    return float(truth(p).strip() == (" ".join(maj) if maj else "1"))


# ---------------------------------------------------------------- registry

@dataclass
class Task:
    name: str
    load: Callable                      # () -> (items, [source files])
    source: str                         # where the data comes from
    depth: str                          # what "depth" means for this task
    normalise: Callable = lambda s: s.strip()
    baselines: dict = field(default_factory=dict)   # name -> per-item score function (raw record -> 0..1)
    chance: float = None


PROSQA_BASE = {"proximity rule (no reasoning)": prosqa_proximity}

TASKS = {t.name: t for t in [
    Task("prosqa", prosqa_loader("prosqa/data/prosqa_test.json"), "Coconut repo (2412.06769) ProsQA test, 500 items",
         "hops on the answer path (3-6)", baselines=PROSQA_BASE, chance=0.5),
    Task("prosqa_na", prosqa_loader("prosqa/p8data/prosqa_na_test.json"),
         "ProsQA test with the proximity shortcut neutralised (THEORY §59); same answers", "hops (3-6)",
         baselines=PROSQA_BASE, chance=0.5),
    Task("prosqa_deep", prosqa_loader("prosqa/p8data/p8_test.json"),
         "ProsQA generator extended to deeper problems (PROTOCOL8 test)", "hops on the answer path",
         baselines=PROSQA_BASE, chance=0.5),
    Task("prosqa_deep_na", prosqa_loader("prosqa/p8data/p8_na_test.json"),
         "PROTOCOL8 deep test with the shortcut neutralised (THEORY §59)", "hops on the answer path",
         baselines=PROSQA_BASE, chance=0.5),
    Task("gsm8k", gsm8k_load, "GSM8K test (2110.14168), 1,319 items, openai/grade-school-math",
         "equation steps in the reference solution", normalise=norm_number, baselines=GSM_BASE),
    Task("mult4x4", mult_loader(4), "4x4-digit multiplication test (implicit-CoT repo, 2311.01460)", "digits",
         normalise=norm_digits),
    Task("boxes", boxes_load, "THEORY §66 generator: 4 boxes, swap ops (S4), N 1-32, 50 per N per question type (seed 6600)",
         "operations N", normalise=boxes_norm,
         baselines={"identity (start)": boxes_identity, "last op only": boxes_lastop,
                    "majority (ABCD / box 1)": boxes_majority, "chance (1/24 all, 1/4 where)": boxes_chance}),
    Task("mult5x5", mult_loader(5), "5x5-digit multiplication test (implicit-CoT repo, 2311.01460)", "digits",
         normalise=norm_digits),
]}
