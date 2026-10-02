"""THEORY §79: operation phrasings for the S3 boxes latent task (fixed before any §79 run).

Formal spec. Three boxes; an arrangement is the tuple (content of box 1, box 2, box 3). Operations (all of S3):
  id            : unchanged
  swap(i, j)    : the contents of boxes i and j are exchanged (i != j; "swap 2 1" == "swap 1 2")
  rotL          : every item moves one box to the LEFT, the item in box 1 wraps to box 3: new = (old2, old3, old1)
  rotR          : every item moves one box to the RIGHT, the item in box 3 wraps to box 1: new = (old3, old1, old2)
Families: 8 training (T1-T8) and 4 held-out (H1-H4); each family renders every operation. Swap pairs are rendered in a
random order of (i, j) (both orders are the same operation), in training and at test. Minimal pairs (word order only, a
bag of words cannot separate them): T6 (training) and H4 (held-out) express rotL / rotR with the same multiset of words.
Controls: POS (casing / whitespace of T1) and NEG (" copy box i into box j", a non-permutation verb, scored against
swap(i, j)). `python theory/s79_phrasings.py` runs the spec tests and prints the operative-word overlap report.
"""
import itertools
import random
import re

OPS = ["id", "swap12", "swap13", "swap23", "rotL", "rotR"]
PAIRS = {"swap12": (1, 2), "swap13": (1, 3), "swap23": (2, 3)}
NUMW = {1: "one", 2: "two", 3: "three"}
ORD = {1: "first", 2: "second", 3: "third"}


def apply(arr, op):
    a = list(arr)
    if op in PAIRS:
        i, j = PAIRS[op]; a[i - 1], a[j - 1] = a[j - 1], a[i - 1]
    elif op == "rotL":
        a = [a[1], a[2], a[0]]
    elif op == "rotR":
        a = [a[2], a[0], a[1]]
    return tuple(a)


# Each family: (swap template with {i} {j} or {I} {J} (number words) or {oi} {oj} (ordinals), rotL, rotR, id)
TRAIN = {
    "T1": ("swap boxes {i} and {j}", "rotate the boxes to the left", "rotate the boxes to the right", "do nothing"),
    "T2": ("exchange box {i} and box {j}", "shift every item one box left", "shift every item one box right",
           "leave everything as it is"),
    "T3": ("switch the contents of box {i} with box {j}", "move each item to the box on its left",
           "move each item to the box on its right", "keep all boxes unchanged"),
    "T4": ("trade the items in boxes {i} and {j}", "turn the row of boxes leftward", "turn the row of boxes rightward",
           "make no change"),
    "T5": ("box {i} and box {j} swap their items", "cycle the items left by one", "cycle the items right by one",
           "nothing happens"),
    "T6": ("put box {i}'s item in box {j} and box {j}'s item in box {i}",
           "the item in box 2 goes to box 1, box 3's to box 2, and box 1's to box 3",       # minimal pair (order only)
           "the item in box 1 goes to box 2, box 2's to box 3, and box 3's to box 1",
           "every item stays where it is"),
    "T7": ("interchange boxes {i} and {j}", "roll the boxes one place to the left", "roll the boxes one place to the right",
           "skip this step"),
    "T8": ("the items of boxes {i} and {j} change places", "slide all items one position left",
           "slide all items one position right", "hold all items in place"),
}
HELDOUT = {
    "H1": ("transpose boxes {i} and {j}",                                                   # new verbs, digits
           "every item retreats one box, and the first item wraps to the last box",
           "every item advances one box, and the last item wraps to the first box",
           "remain idle"),
    "H2": ("box {i} gives its item to box {j} and receives box {j}'s item in return",       # different frame
           "each box receives the item from the box after it, and the last box receives the first box's item",
           "each box receives the item from the box before it, and the first box receives the last box's item",
           "no box gives or receives anything"),
    "H3": ("flip box {I} with box {J}",                                                     # number words, new verb
           "pass every item back one box, with box one's item going to box three",
           "pass every item forward one box, with box three's item going to box one",
           "pause"),
    "H4": ("reverse the {oi} and {oj} boxes",                                               # ordinals; minimal pair
           "box 2 hands its item to box 1, box 3 hands to box 2, and box 1 hands to box 3",
           "box 1 hands its item to box 2, box 2 hands to box 3, and box 3 hands to box 1",
           "wait a moment"),
}
FAMILIES = {**TRAIN, **HELDOUT}


def render(family, op, rng=None):
    """Text (with a leading space, GPT-2 style) of operation `op` in `family`; POS / NEG are the controls."""
    rng = rng or random
    if family == "NEG":
        assert op in PAIRS
        i, j = PAIRS[op]
        if rng.random() < 0.5:
            i, j = j, i
        return " copy box %d into box %d" % (i, j)
    fam = FAMILIES["T1" if family == "POS" else family]
    if op in PAIRS:
        i, j = PAIRS[op]
        if rng.random() < 0.5:
            i, j = j, i
        s = fam[0].format(i=i, j=j, I=NUMW[i], J=NUMW[j], oi=ORD[i], oj=ORD[j])
    else:
        s = fam[{"rotL": 1, "rotR": 2, "id": 3}[op]]
    if family == "POS":
        s = s[0].upper() + s[1:].replace(" ", "  ", 1)
    return " " + s


STOP = {"the", "a", "of", "to", "in", "and", "its", "it", "is", "as", "on", "by", "with", "from", "their", "all", "each",
        "every", "box", "boxes", "item", "items", "one", "two", "three", "1", "2", "3", "s", "first", "second", "third",
        "last", "this", "in", "where", "after", "before", "return", "row", "place", "position", "contents"}


def words(fam):
    return {w for t in fam for w in re.findall(r"[a-z]+|\d", re.sub(r"\{\w+\}", "", t).lower())} - STOP


def report():
    tr = set().union(*(words(f) for f in TRAIN.values()))
    print("training operative words:", sorted(tr))
    for k, f in HELDOUT.items():
        w = words(f); ov = w & tr
        print("%s operative words %s | overlap with training %s (%d/%d)" % (k, sorted(w), sorted(ov), len(ov), len(w)))
    # minimal pairs: identical bags of words, different operations
    for k in ("T6", "H4"):
        bag = lambda s: sorted(re.findall(r"[a-z']+|\d", s.lower()))          # noqa: E731
        print(k, "rotL/rotR identical bag of words:", bag(FAMILIES[k][1]) == bag(FAMILIES[k][2]))


def tests():
    arrs = list(itertools.permutations("ABC"))
    for a in arrs:
        assert apply(apply(a, "rotL"), "rotR") == a and apply(apply(apply(a, "rotL"), "rotL"), "rotL") == a
        for p in PAIRS:
            assert apply(apply(a, p), p) == a
    assert apply(("A", "B", "C"), "rotL") == ("B", "C", "A") and apply(("A", "B", "C"), "rotR") == ("C", "A", "B")
    assert len({tuple(apply(("A", "B", "C"), o)) for o in OPS}) == 6                 # all six elements distinct
    rng = random.Random(0)
    for f in list(FAMILIES) + ["POS"]:
        for o in OPS:
            assert render(f, o, rng).startswith(" ") and "{" not in render(f, o, rng)
    print("spec tests pass")


if __name__ == "__main__":
    tests(); report()
