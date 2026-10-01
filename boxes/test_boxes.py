"""Unit tests for THEORY §66 scoring: normaliser (amendment 66-1), truth, baselines.  python -m boxes.test_boxes"""
import random

from bench.tasks import TASKS, boxes_chance, boxes_identity, boxes_lastop, boxes_norm
from boxes.core import apply, make_problem, truth

assert boxes_norm("A,C,B,D") == boxes_norm(" A C B D") == boxes_norm("ACBD") == "A C B D"
assert boxes_norm(" 3") == boxes_norm("box 3.") == "3"
assert boxes_norm(" a c b d") == "A C B D"
r = random.Random(1)
for n in range(1, 33):
    for qt in ("all", "where"):
        p = make_problem(r, n, qt)
        t = truth(p)
        assert boxes_norm(t) == boxes_norm(t.replace(" ", ",")), t
        if qt == "all":
            assert sorted(boxes_norm(t).split()) == list("ABCD")
p = {"start": "BADC", "ops": [(0, 1)], "qtype": "all", "target": "A"}
assert apply(p["start"], p["ops"]) == "ABDC" and boxes_lastop(p) == 1.0 and boxes_identity(p) == 0.0
assert abs(boxes_chance({"qtype": "all"}) - 1 / 24) < 1e-12 and boxes_chance({"qtype": "where"}) == 0.25
items, _ = TASKS["boxes"].load()
assert len(items) == 32 * 2 * 50 and len({i["id"] for i in items}) == len(items)
print("BOXES TESTS PASSED")
