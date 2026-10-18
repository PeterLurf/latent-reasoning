"""PROOFS_LANG item 1c: how much of §191 A's p8_na_test accuracy can person-agnostic §59 cues (option + chain-anchor level,
plus the observable chain length m) explain? Bayes-optimal lookup over the cue sign vector, fitted on p8_na_train,
scored on p8_na_test per depth (plus an in-sample test lookup = optimistic ceiling). Compared with §191 A's acc_by_depth.
    python -B cue_ceiling.py <out.json>
"""
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from anchor_check import parse, heur, anchors  # noqa: E402

H = ("proximity", "parents", "mentions", "first", "last", "pindeg")


def feats(x):
    edges, words, _ = parse(x)
    m = int(x.get("ext", 0))
    # orient by the displayed option order so the label is "is the FIRST displayed option the target?"
    q = x["question"].rsplit(" Is ", 1)[1]
    first_name = q.split(" a ", 1)[1].split(" or ")[0]
    t_first = x["idx_to_symbol"][x["target"]] == first_name
    o = heur(x, edges, words, x["target"], x["neg_target"])
    f = [m]
    sgn = lambda v: 0 if v == 0.5 else (1 if v == 1.0 else -1)  # noqa: E731  +1 = cue favours target
    flip = 1 if t_first else -1  # express cue as "favours the first displayed option"
    f += [sgn(o[h]) * flip for h in H]
    if m >= 1:
        A, _B, _amb = anchors(x, edges)
        a = heur(x, edges, words, A[0], A[1])
        f += [sgn(a[h]) * flip for h in H]
    else:
        f += [0] * len(H)
    return tuple(f), t_first


def main():
    tr = json.load(open("D:/Latent/prosqa/p8data/p8_na_train.json")); te = json.load(open("D:/Latent/prosqa/p8data/p8_na_test.json"))
    tab = collections.defaultdict(collections.Counter)
    for x in tr:
        f, y = feats(x); tab[f][y] += 1
    tab_te = collections.defaultdict(collections.Counter); rows = []
    for x in te:
        f, y = feats(x); tab_te[f][y] += 1; rows.append((len(x["steps"]), f, y))

    def pred(t, f):
        c = t.get(f)
        if not c or c[True] == c[False]:
            return None
        return c[True] > c[False]
    out = collections.defaultdict(lambda: [0.0, 0.0, 0])
    for D, f, y in rows:
        p = pred(tab, f); s = 0.5 if p is None else float(p == y)
        q = pred(tab_te, f); s2 = 0.5 if q is None else float(q == y)
        out[D][0] += s; out[D][1] += s2; out[D][2] += 1
    s191 = json.load(open("D:/Latent/boxes/runs_fsq/s191_stepaux_relabel_s1/eval.json"))["acc_by_depth"]
    res = {}
    for D in sorted(out):
        a, b, n = out[D]
        res[D] = {"n": n, "cue_lookup_train_fit": round(a / n, 4), "cue_lookup_test_insample": round(b / n, 4), "s191A_acc": round(s191[str(D)], 4)}
        print(D, res[D])
    json.dump({"def": __doc__, "by_depth": res, "n_cells_train": len(tab)}, open(sys.argv[1], "w", newline="\n"), indent=1)


if __name__ == "__main__":
    main()
