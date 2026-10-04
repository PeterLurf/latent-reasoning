"""THEORY §100 (Theory's request 2026-10-04, descriptive): §104's donor protocol on §100 MASK2 4x4 (mult98 arm mask, k = 2),
masked decodes as trained. n recipient problems A and n donor problems B (fresh random 4-digit a and b, leading digit
non-zero; step texts built in the dataset's format and checked token-exact against the test file).
  chunk1_slot1: chunk 1 (ops 1-2, no start). Slot 1's input <- B's slot-1 input (B's chunk 1). The slot-2 step text
    " + P_2 ( S_2 )" is decoded and S_2 parsed. donor_consistent: S_2 = a_B b_1^B + a_A b_2^A 10; recipient_consistent:
    S_2 = A's true S_2.
  chunk2_slot3: chunk 2 (ops 3-4, start = the true step-2 text: A's for A, B's for B). Slot 3's input <- B's slot-3
    input. The final fork (step 4 ++ " #### product") is decoded and the answer parsed. donor_consistent:
    S_3^B + a_A b_4^A 10^3; recipient_consistent: A's true product.
  intact: the same decodes with no patch (S_2 and answer accuracy).
Unparseable decodes count as neither. Writes <run>/donor100.json.
    python -m boxes.donor100 --run boxes/runs_mult/m100_mask2_4x4_s1 [--n 2000 --bs 200 --device cuda:0]
"""
import os
os.environ["BOXES_NB"] = "5"
import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from boxes import s5  # noqa: E402
from boxes.mult98 import DIGIT_IDS, HASH, TOK, MultGen, digits_lsb, parse  # noqa: E402

NA = NB = 4
TOK2DIG = {int(t): d for d, t in enumerate(DIGIT_IDS)}


def steps_of(a, b):
    """The dataset's step texts (token ids) and answer for a * b (4x4): step 1 " P_1", step j " + P_j ( S_j )", last " + P_4"."""
    bd = digits_lsb(b, NB); out = []; S = 0
    for j in range(NB):
        P = a * bd[j] * 10 ** j; S += P; w = NA + 1 + j
        p = [TOK[str(x)] for x in digits_lsb(P, w)]
        if j == 0:
            out.append(p)
        elif j < NB - 1:
            out.append([TOK["+"]] + p + [TOK["("]] + [TOK[str(x)] for x in digits_lsb(S, w)] + [TOK[")"]])
        else:
            out.append([TOK["+"]] + p)
    ans = list(HASH) + [TOK[str(x)] for x in digits_lsb(a * b, NA + NB)]
    return out, ans


def check_format():
    with open(os.path.join(ROOT, "bench", "data", "mult_4x4_test.txt")) as f:
        for line in list(f)[:200]:
            a, b, steps, prod = parse(line)
            ai, bi = int("".join(a[::-1])), int("".join(b[::-1])); st, ans = steps_of(ai, bi)
            assert st == [[TOK[w] for w in s] for s in steps], line
            assert ans == list(HASH) + [TOK[w] for w in prod], line


def to_num(toks):
    try:
        return int("".join(str(TOK2DIG[int(t)]) for t in toks[::-1]))
    except KeyError:
        return None


def problems(rng, n):
    a = rng.integers(10 ** (NA - 1), 10 ** NA, n); b = rng.integers(10 ** (NB - 1), 10 ** NB, n)
    st = [steps_of(int(x), int(y)) for x, y in zip(a, b)]
    return {"a": a, "b": b, "A": np.array([digits_lsb(int(x), NA) for x in a]), "B": np.array([digits_lsb(int(y), NB) for y in b]),
            "steps": [np.array([s[0][j] for s in st]) for j in range(NB)], "ans": np.array([s[1] for s in st])}


@torch.no_grad()
def run(m, dev, amp, n, bs, seed):
    rng = np.random.default_rng(seed); A = problems(rng, n); B = problems(rng, n)
    T = lambda x: torch.tensor(x, device=dev)  # noqa: E731
    pre = lambda P, s: T(np.concatenate([DIGIT_IDS[P["A"][s]], np.full((len(P["A"][s]), 1), TOK["*"])], 1))  # noqa: E731
    L2, L3, L4, LA = A["steps"][1].shape[1], A["steps"][2].shape[1], A["steps"][3].shape[1], A["ans"].shape[1]
    s2 = np.full((2, n), -1, dtype=object); fa = np.full((2, n), -1, dtype=object)       # [intact, patched]
    for s0 in range(0, n, bs):
        s = slice(s0, min(n, s0 + bs))
        with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
            _, _, _, insB1 = m.chunk(pre(B, s), None, T(B["B"][s, 0:2]))
            _, _, _, insB2 = m.chunk(pre(B, s), T(B["steps"][1][s]), T(B["B"][s, 2:4]))
            for pi, ov1, ov2 in ((0, None, None), (1, (0, insB1[:, 0]), (0, insB2[:, 0]))):
                m.override = ov1
                _, d1, _, _ = m.chunk(pre(A, s), None, T(A["B"][s, 0:2]), step_len=[A["steps"][0].shape[1], L2])
                m.override = ov2
                _, _, an, _ = m.chunk(pre(A, s), T(A["steps"][1][s]), T(A["B"][s, 2:4]), step_len=[L3, L4], ans_len=LA)
                m.override = None
                t2 = d1[1].cpu().numpy(); w = NA + 2                                    # " + P_2(6) ( S_2(6) )"
                ok = (t2[:, 0] == TOK["+"]) & (t2[:, 1 + w] == TOK["("]) & (t2[:, -1] == TOK[")"])
                s2[pi, s] = [to_num(r[2 + w:2 + 2 * w]) if o else None for r, o in zip(t2, ok)]
                ta = an.cpu().numpy(); okh = (ta[:, :len(HASH)] == np.array(HASH)).all(1)
                fa[pi, s] = [to_num(r[len(HASH):]) if o else None for r, o in zip(ta, okh)]
    aA, bA = A["a"].astype(object), A["B"]; aB = B["a"].astype(object); bB = B["B"]
    S2A = aA * bA[:, 0] + aA * bA[:, 1] * 10; S2D = aB * bB[:, 0] + aA * bA[:, 1] * 10
    S3B = aB * (bB[:, 0] + bB[:, 1] * 10 + bB[:, 2] * 100); PA = aA * A["b"].astype(object); PD = S3B + aA * bA[:, 3] * 1000
    eq = lambda x, y: float(np.mean([u is not None and u == v for u, v in zip(x, y)]))  # noqa: E731
    return {"n": n, "seed": seed,
            "intact": {"S2_acc": eq(s2[0], S2A), "answer_acc": eq(fa[0], PA)},
            "donor": {"chunk1_slot1": {"donor_consistent": eq(s2[1], S2D), "recipient_consistent": eq(s2[1], S2A),
                                       "unparsed": float(np.mean([u is None for u in s2[1]]))},
                      "chunk2_slot3": {"donor_consistent": eq(fa[1], PD), "recipient_consistent": eq(fa[1], PA),
                                       "unparsed": float(np.mean([u is None for u in fa[1]]))}},
            "chance_overlap": {"chunk1": eq(S2D, S2A), "chunk2": eq(PD, PA)}}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--bs", type=int, default=200); ap.add_argument("--seed", type=int, default=10001); ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args(); dev = a.device; check_format()
    args = json.load(open(os.path.join(a.run, "args.json"))); assert args["arm"] == "mask" and args["task"] == "4x4" and args["k"] == 2, args
    amp = torch.bfloat16 if dev.startswith("cuda") and torch.cuda.get_device_capability(dev)[0] >= 8 else None
    m = MultGen(s5.load_gpt2(dev), "mask").to(dev); m.load_state_dict(torch.load(os.path.join(a.run, "model.pt"), map_location=dev)); m.eval()
    r = run(m, dev, amp, a.n, a.bs, a.seed); r.update({"run": a.run, "eval_dtype": str(amp)})
    json.dump(r, open(os.path.join(a.run, "donor100.json"), "w", newline="\n"), indent=1); print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
