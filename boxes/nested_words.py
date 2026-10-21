"""§223 PORT-STATE F1: the NESTED-WORDS v1 generator (docs/research_portstate_2026-10-21/PS2_v1.md at 54f63381 §1; REVIEW B3 / M6; Theory 10-21 on
ea7c1fbc: mul_s). NEW module, imported by nothing else yet. A program is a straight-line chain of d steps over named quantities with a forced
dependency SPINE: step t defines spine quantity S_t from S_{t-1} and one more operand, so the DAG depth equals d.
  * ops: add_c / sub_c / mul_c (S_{t-1} with a constant), add_v / sub_v / mul_v (S_{t-1} with a GIVEN quantity G_t stated as a constant sentence), and
    the cross-spine ops mul_s / add_s (S_{t-1} with an EARLIER spine quantity S_j, j <= t-2; S_0 counts). Every program of d >= 2 has exactly one
    forced mul_s at a uniform t in 2..d (a product of two computed / spine quantities: no composition of affine maps x -> a x + b reproduces it, the
    REVIEW B3(b) shortcut); add_s is a regular op; a d = 1 program is a single mul_v (no mul_s is possible; d 1-2 are not in the G-W0 gate range).
    Every quantity (given or computed) lies in 0..99; multiplicands are nonzero (mul_c constants 2..9, mul_v givens 2..9; both factors of mul_s >= 2, so the cross product is never a x1 no-op that
    an affine chain would reproduce - G-W0's affine rule).
    Construction: steps are drawn in order; a non-forced step whose value leaves 0..99 is redrawn (up to 50 tries); the forced mul_s picks j uniformly
    among the j <= t-2 with S_j >= 2, S_{t-1} >= 2 and S_{t-1} x S_j <= 99; if a step cannot be placed the whole program restarts (counted: Stream.restarts).
  * language: 8 paraphrase frames per op (mul_s uses mul_v's frames and add_s add_v's, with the partner's name; frames 6 and 7 are held out of
    training, FRAME_TRAIN = 0..5; test items draw from all 8 and record each op statement's frame); sampled single-token person names; one sampled
    unit noun per program; 0-2 numeric DISTRACTOR sentences (unused names, same unit); statements in a random topological order (a statement comes
    after every statement it reads); the question "How many <u> does <S_d> have?" is the last sentence.
  * R_t (t = 0..d; the re-rooted problem after the first t spine steps): R_0 = the problem. For 1 <= t < d: the statement defining S_t is replaced
    IN PLACE by the constant sentence "<S_t> has <value> <u>."; every S_k (k < t) statement that a later step (> t) still reads (via mul_s / add_s)
    is likewise KEPT IN PLACE as its constant sentence; all other S_k (k < t) statements and the givens of steps <= t are removed; every remaining
    statement keeps its frame, name and position; distractors and the question are kept. R_d = the answer sentence "<S_d> has <answer> <u>." alone.
    The DAG has back-edges but no parallel branches, so the spine order defines R_t uniquely (the leftmost-first tie-break is a no-op in v1).
  * item: {"d", "text", "question", "answer", "R": [R_0..R_d], "values": [S_0..S_d], "ops", "operands", "src_j", "mul_s_t", "givens", "frames",
    "names", "unit", "n_distractors", "order", "sig"}; sig = sha1 of the rendered text (exclusion by rendered match).
  * streams: Stream(seed, excl) draws items of a given d from np.random.default_rng(seed), skipping any whose sig is in excl. Sets: test_set /
    pair seeds are fixed below; set_sha = sha256 over the rendered texts in order.
    python -B -m boxes.nested_words --selfcheck"""
import hashlib
import json
import sys

import numpy as np

NAMES = ["Ann", "Bob", "Jim", "Dan", "Eve", "Finn", "Gus", "Hal", "Ivy", "Jon", "Kim", "Leo", "Mia", "Ned", "Joe", "Pam", "Ray", "Sam", "Tom", "Kate",
         "Val", "Wes", "Zoe", "Amy", "Ben", "Cal", "Dee", "Eli", "Fay", "Gil", "Ian", "Joy", "Ken", "Lou", "Max", "Nan", "Pat", "Ron", "Sue", "Ted"]
UNITS = ["coins", "apples", "stamps", "books", "eggs", "shells", "cards", "pens", "cups", "keys"]
OPS = ("add_c", "sub_c", "mul_c", "add_v", "sub_v", "mul_v", "add_s", "mul_s")
OPS_REG = ("add_c", "sub_c", "mul_c", "add_v", "sub_v", "mul_v", "add_s")   # the regular (non-forced) ops
BASE = {"add_c": "add", "sub_c": "sub", "mul_c": "mul", "add_v": "add", "sub_v": "sub", "mul_v": "mul", "add_s": "add", "mul_s": "mul"}
FRAME_TRAIN = tuple(range(6)); FRAME_ALL = tuple(range(8))
FRAMES = {
    "add_c": ["{B} has {c} more {u} than {A}.", "{B} owns {c} {u} more than {A} does.", "Compared with {A}, {B} has {c} extra {u}.",
              "{B} has the number of {u} {A} has, plus {c}.", "{A} has {c} fewer {u} than {B}.", "If you add {c} to {A}'s {u}, you get {B}'s {u}.",
              "{B} holds {c} {u} beyond what {A} holds.", "{B}'s count of {u} exceeds {A}'s by {c}."],
    "sub_c": ["{B} has {c} fewer {u} than {A}.", "{B} owns {c} {u} less than {A} does.", "Compared with {A}, {B} is short {c} {u}.",
              "{B} has the number of {u} {A} has, minus {c}.", "{A} has {c} more {u} than {B}.", "If you take {c} from {A}'s {u}, you get {B}'s {u}.",
              "{B} holds {c} {u} less than what {A} holds.", "{A}'s count of {u} exceeds {B}'s by {c}."],
    "mul_c": ["{B} has {c} times as many {u} as {A}.", "{B} owns {c} times the {u} that {A} owns.", "{A}'s {u}, multiplied by {c}, give {B}'s {u}.",
              "{B}'s {u} are {c} times {A}'s.", "{B} has {c} times the number of {u} {A} has.", "Multiply {A}'s {u} by {c} to get {B}'s {u}.",
              "{B} holds {c} times as many {u} as {A} holds.", "The {u} of {B} number {c} times those of {A}."],
    "add_v": ["{B} has as many {u} as {A} and {C} together.", "{B} owns the {u} of {A} and {C} combined.", "{A}'s {u} plus {C}'s {u} make {B}'s {u}.",
              "{B}'s {u} equal the sum of {A}'s and {C}'s.", "Together, {A} and {C} have as many {u} as {B}.", "Add {C}'s {u} to {A}'s {u} to get {B}'s {u}.",
              "{B} holds the {u} of {A} plus those of {C}.", "The {u} of {B} total those of {A} and {C}."],
    "sub_v": ["{B} has as many {u} as {A} has more than {C}.", "{B} owns the difference between {A}'s and {C}'s {u}.", "{A}'s {u} minus {C}'s {u} make {B}'s {u}.",
              "{B}'s {u} equal {A}'s minus {C}'s.", "{B} has the {u} {A} has beyond {C}'s.", "Take {C}'s {u} from {A}'s {u} to get {B}'s {u}.",
              "{B} holds the {u} of {A} less those of {C}.", "The {u} of {B} are what {A} has above {C}."],
    "mul_v": ["{B} has as many {u} as {A}'s {u} times {C}'s.", "{B} owns the product of {A}'s and {C}'s {u}.", "{A}'s {u} times {C}'s {u} make {B}'s {u}.",
              "{B}'s {u} equal {A}'s multiplied by {C}'s.", "Multiply {A}'s {u} by {C}'s {u} to get {B}'s {u}.", "{B} holds {A}'s {u} multiplied by {C}'s {u}.",
              "For each of {C}'s {u}, {B} has as many {u} as {A}.", "The {u} of {B} number {A}'s times {C}'s."],
}
FRAMES["mul_s"] = FRAMES["mul_v"]; FRAMES["add_s"] = FRAMES["add_v"]      # the partner {C} is an earlier spine quantity's name
GIVEN = "{A} has {v} {u}."
QUESTION = "How many {u} does {A} have?"
SEED_TEST, SEED_PAIRS = 22300, 22310            # §223 test / pair streams (PS2_v1 §1: test and pair sets carry a sha)


def apply(op, a, b):
    return {"add": a + b, "sub": a - b, "mul": a * b}[BASE[op]]


def _operand(rng, op):
    return int(rng.integers(2, 10)) if BASE[op] == "mul" else int(rng.integers(1, 21))


def _draw_program(rng, d):
    """-> (ops, operands b_t, src_j (None unless *_s), S values, mul_s_t, restarts); every value in 0..99; one forced mul_s at t in 2..d (d >= 2)."""
    restarts = 0
    while True:
        ts = int(rng.integers(2, d + 1)) if d >= 2 else None
        s = [int(rng.integers(1, 21))]; ops, bs, js = [], [], []; ok = True
        for t in range(1, d + 1):
            if t == ts:                                                     # the forced cross-spine product
                cand = [j for j in range(0, t - 1) if s[j] >= 2 and s[t - 1] >= 2 and s[t - 1] * s[j] <= 99]   # both factors >= 2: never a x1 / x0 no-op
                if not cand:
                    ok = False; break
                j = cand[int(rng.integers(0, len(cand)))]; ops.append("mul_s"); bs.append(s[j]); js.append(j); s.append(s[t - 1] * s[j]); continue
            for _ in range(50):
                op = "mul_v" if d == 1 else OPS_REG[int(rng.integers(0, len(OPS_REG)))]
                if op == "add_s":
                    if t < 2:
                        continue
                    j = int(rng.integers(0, t - 1)); b = s[j]
                else:
                    j = None; b = _operand(rng, op)
                if BASE[op] == "mul" and s[t - 1] == 0:
                    continue
                v = apply(op, s[t - 1], b)
                if 0 <= v <= 99:
                    break
            else:
                ok = False; break
            ops.append(op); bs.append(b); js.append(j); s.append(v)
        if ok:
            return ops, bs, js, s, ts, restarts
        restarts += 1


def make(rng, d, frames=FRAME_TRAIN, stats=None):
    """one item of depth d."""
    ops, bs, js, s, ts, rs = _draw_program(rng, d)
    if stats is not None:
        stats["restarts"] = stats.get("restarts", 0) + rs; stats["items"] = stats.get("items", 0) + 1
    n_v = sum(op.endswith("_v") for op in ops); n_dis = int(rng.integers(0, 3))
    pick = rng.choice(len(NAMES), size=d + 1 + n_v + n_dis, replace=False); nm = [NAMES[i] for i in pick]
    spine = nm[:d + 1]; gnames = nm[d + 1:d + 1 + n_v]; dnames = nm[d + 1 + n_v:]; u = UNITS[int(rng.integers(0, len(UNITS)))]
    st = [{"id": "S0", "kind": "given", "deps": [], "name": spine[0], "v": s[0]}]
    gi = 0; givens = []
    for i, op in enumerate(ops, 1):
        f = int(frames[int(rng.integers(0, len(frames)))])
        if op.endswith("_v"):
            g = gnames[gi]; gi += 1; givens.append({"name": g, "v": bs[i - 1], "step": i})
            st.append({"id": "G%d" % i, "kind": "given", "deps": [], "name": g, "v": bs[i - 1]})
            st.append({"id": "S%d" % i, "kind": "op", "op": op, "frame": f, "deps": ["S%d" % (i - 1), "G%d" % i], "B": spine[i], "A": spine[i - 1], "C": g, "c": None})
        elif op.endswith("_s"):
            j = js[i - 1]
            st.append({"id": "S%d" % i, "kind": "op", "op": op, "frame": f, "deps": ["S%d" % (i - 1), "S%d" % j], "B": spine[i], "A": spine[i - 1], "C": spine[j], "c": None})
        else:
            st.append({"id": "S%d" % i, "kind": "op", "op": op, "frame": f, "deps": ["S%d" % (i - 1)], "B": spine[i], "A": spine[i - 1], "C": None, "c": bs[i - 1]})
    for j, x in enumerate(dnames):
        st.append({"id": "D%d" % j, "kind": "distractor", "deps": [], "name": x, "v": int(rng.integers(1, 100))})
    order = topo_shuffle(rng, st); byid = {x["id"]: x for x in st}; seq = [byid[i] for i in order]
    q = QUESTION.format(u=u, A=spine[-1])

    def render(x, val=None):
        if val is not None:
            return GIVEN.format(A=x["B"] if x["kind"] == "op" else x["name"], v=val, u=u)
        if x["kind"] in ("given", "distractor"):
            return GIVEN.format(A=x["name"], v=x["v"], u=u)
        return FRAMES[x["op"]][x["frame"]].format(B=x["B"], A=x["A"], C=x["C"], c=x["c"], u=u)
    text = " ".join([render(x) for x in seq] + [q])
    R = [text]
    for t in range(1, d):                                                   # R_t (Theory 10-21: back-referenced S_k kept as constants)
        later = {js[k - 1] for k in range(t + 1, d + 1) if js[k - 1] is not None}
        keep_const = {"S%d" % k: s[k] for k in later if k < t}; keep_const["S%d" % t] = s[t]
        drop = {"S%d" % k for k in range(t)} - set(keep_const) | {"G%d" % k for k in range(1, t + 1)}
        R.append(" ".join([render(x, keep_const.get(x["id"])) for x in seq if x["id"] not in drop] + [q]))
    R.append(GIVEN.format(A=spine[-1], v=s[-1], u=u))                       # R_d = the answer sentence
    it = {"d": d, "text": text, "question": q, "answer": s[-1], "R": R, "values": s, "ops": ops, "operands": bs, "src_j": js, "mul_s_t": ts, "givens": givens,
          "frames": [x.get("frame") for x in st if x["kind"] == "op"], "names": {"spine": spine, "givens": gnames, "distractors": dnames}, "unit": u,
          "n_distractors": n_dis, "order": order}
    it["sig"] = hashlib.sha1(text.encode()).hexdigest(); return it


def affine_solve(it):
    """the affine-chain shortcut (Theory 10-21 G-W0 rule): S_0 pushed through every op with its stated operand, cross-spine (*_s) ops ignored."""
    x = it["values"][0]
    for op, b in zip(it["ops"], it["operands"]):
        if not op.endswith("_s"):
            x = apply(op, x, b)
    return x


def topo_shuffle(rng, st):
    """a uniformly random linear extension (by repeated random choice among ready statements) of the deps order -> list of ids."""
    left = {x["id"]: set(x["deps"]) for x in st}; out = []
    while left:
        ready = sorted(i for i, dps in left.items() if not dps)
        pick = ready[int(rng.integers(0, len(ready)))]; out.append(pick); del left[pick]
        for dps in left.values():
            dps.discard(pick)
    return out


def solve(text):
    """an INDEPENDENT solver of a rendered problem (regex over the frames; the invariant check): -> the asked quantity's value or None."""
    import re
    sents = [x.strip() for x in re.split(r"(?<=[.?])\s+", text) if x.strip()]
    qm = re.match(r"How many (\w+) does (\w+) have\?", sents[-1]); u, target = qm.group(1), qm.group(2); val = {}; rules = []
    pats = []
    for op, fs in FRAMES.items():
        for f in fs:
            rx = re.escape(f).replace(re.escape("{u}"), re.escape(u))
            for k, g in (("B", r"(?P<B>\w+)"), ("A", r"(?P<A>\w+)"), ("C", r"(?P<C>\w+)"), ("c", r"(?P<c>\d+)")):
                rx = rx.replace(re.escape("{%s}" % k), g, 1)
            pats.append((op, re.compile("^" + rx + "$")))
    gv = re.compile(r"^(\w+) has (\d+) " + re.escape(u) + r"\.$")
    for s_ in sents[:-1]:
        m = gv.match(s_)
        if m:
            val[m.group(1)] = int(m.group(2)); continue
        hit = [(op, mm) for op, p in pats for mm in [p.match(s_)] if mm]
        if not hit:
            return None
        rules.append(hit)                                                   # every reading: mirrored frames (add_c 4 = sub_c 0 reversed, ...) are one equation
    for _ in range(len(rules) + 1):
        for hit in rules:
            for op, mm in hit:
                gd = mm.groupdict(); a = val.get(gd["A"]); b = int(gd["c"]) if gd.get("c") is not None else val.get(gd.get("C"))
                if a is not None and b is not None and gd["B"] not in val:
                    val[gd["B"]] = apply(op, a, b)
    return val.get(target)


class Stream:
    def __init__(self, seed, excl=(), frames=FRAME_TRAIN):
        self.rng = np.random.default_rng(seed); self.excl = set(excl); self.frames = frames; self.stats = {}

    def draw(self, d):
        while True:
            it = make(self.rng, d, self.frames, self.stats)
            if it["sig"] not in self.excl:
                return it


def test_set(per_d=1000, ds=range(1, 13), seed=SEED_TEST):
    """the test set {d: [items]} (d 1-8 registered, 9-12 extrapolation; all 8 frames)."""
    s = Stream(seed, frames=FRAME_ALL); return {d: [s.draw(d) for _ in range(per_d)] for d in ds}


def set_sha(items):
    return hashlib.sha256("\n".join(x["text"] for x in items).encode()).hexdigest()


def selfcheck(n=300):
    rng = np.random.default_rng(1); bad = []
    for d in range(1, 13):
        for _ in range(n // 12 + 1):
            it = make(rng, d, FRAME_ALL)
            if solve(it["text"]) != it["answer"]:
                bad.append(("solve", d, it["text"]))
            for t in range(1, d):
                if solve(it["R"][t]) != it["answer"]:
                    bad.append(("R_t", d, t, it["R"][t]))
            if (it["ops"].count("mul_s") != (1 if d >= 2 else 0)) or not all(0 <= v <= 99 for v in it["values"]):
                bad.append(("constraint", d))
    print("SELFCHECK", "PASS" if not bad else "FAIL", len(bad), bad[:3]); return not bad


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        sys.exit(0 if selfcheck() else 1)
    it = make(np.random.default_rng(0), 4)
    print(json.dumps({k: it[k] for k in ("d", "text", "answer", "R", "values", "ops")}, indent=1))
