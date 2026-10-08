"""THEORY note 152-3 (descriptive): the text-CoT and implicit-CoT baselines' extrapolation on §152's ext_4x8_low8 items.
Items: exactly multrow.ext_4x8_low8's (multcol.gen_operands, rng default_rng(14204): A = 1,000 4-digit a, then B = 1,000 8-digit b, LSB first).
For j = 4..8 the problem is a x (b_1 .. b_j) (j = 4 is the in-distribution check; j = 5..8 extrapolate past the 4x4 training), and an answer counts
when its low 8 digits equal (a x b_1..j) mod 10^8, the quantity §152's ext_4x8_low8 scores per j. Also reported: the full-product exact rate.
  TC-M     boxes/runs_mult/m98_tc_4x4_s1/model.pt (text CoT, §98): question = digits * digits ||, free greedy decoding (the 4x4 CoT length is not
           imposed), the answer = the digit tokens after the first " ####".
  ICoT-SI  the released 4x4 checkpoint (ext/icotsi/4x4/state_dict.bin; boxes/icotsi98's input format and two-EOS stop), the answer after "####".
CPU only (torch threads <= 3; run at Idle priority). Writes theory/results/ext_baselines_s152.json.
    python theory/eval_ext_baselines.py [--n 1000] [--threads 3]
"""
import os
os.environ["BOXES_NB"] = "5"                                                # boxes.core must load with BOXES_NB=5 (as boxes.icotsi98)
import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from boxes import s5  # noqa: E402
from boxes import core as bc  # noqa: E402
from boxes.multcol import gen_operands  # noqa: E402
from boxes.mult98 import DIGIT_IDS, HASH, QSEP, TOK  # noqa: E402
from boxes.icotsi98 import EOS, generate, load_icot  # noqa: E402

DIG = {int(t): d for d, t in enumerate(DIGIT_IDS)}


def items(n, seed=14204):
    rng = np.random.default_rng(seed)
    return gen_operands(rng, n, 4), gen_operands(rng, n, 8)                 # the same call order as multrow.ext_4x8_low8


def to_int(row):
    return int("".join(str(x) for x in row[::-1]))


def low8(v):
    return [int(c) for c in str(v % 10 ** 8).zfill(8)[::-1]]                 # LSB first


def digits_after_hash(toks, hash_ids):
    """LSB-first digits after the first occurrence of hash_ids in toks (stops at the first non-digit token); None without a hash."""
    t = list(toks); h = list(hash_ids)
    for i in range(len(t) - len(h) + 1):
        if t[i:i + len(h)] == h:
            out = []
            for x in t[i + len(h):]:
                if int(x) not in DIG:
                    break
                out.append(DIG[int(x)])
            return out
    return None


@torch.no_grad()
def tcm_generate(g, q, max_new):
    """Greedy from the question rows; stop when every row has a hash followed by a non-digit token, or at max_new."""
    x = torch.tensor(q); h, past = g(g.embed(x)); t = g.logits(h[:, -1]).argmax(-1); out = [t]
    for _ in range(max_new - 1):
        h, past = g(g.embed(t)[:, None], past); t = g.logits(h[:, 0]).argmax(-1); out.append(t)
        if len(out) % 16 == 0 and all(_closed(r) for r in torch.stack(out, 1).numpy()):
            break
    return torch.stack(out, 1).numpy()


def _closed(r):
    """True when the row has ' ####' followed by digits and then a non-digit token."""
    t = list(r); h = list(HASH)
    for i in range(len(t) - len(h) + 1):
        if t[i:i + len(h)] == h:
            rest = t[i + len(h):]
            return any(int(x) not in DIG for x in rest[1:]) if rest else False
    return False


def strip_pad(d):
    """The ICoT format zero-pads the product at the top (LSB-first: trailing zeros); drop them before the full-product comparison."""
    d = list(d)
    while len(d) > 1 and d[-1] == 0:
        d = d[:-1]
    return d


def score(ans_digits, truth, n_ans):
    """The answer is truncated to n_ans = 4 + j digits (the format's product length): TC-M keeps emitting digits after its answer."""
    ans_digits = None if ans_digits is None else ans_digits[:n_ans]
    lo = ans_digits is not None and len(ans_digits) >= 8 and ans_digits[:8] == low8(truth)
    full = ans_digits is not None and len(ans_digits) > 0 and strip_pad(ans_digits) == [int(c) for c in str(truth)[::-1]]
    return lo, full


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=1000); ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--bs", type=int, default=100); ap.add_argument("--js", default="4,5,6,7,8")
    a = ap.parse_args(); torch.set_num_threads(a.threads); dev = "cpu"; t0 = time.time()
    A, B = items(a.n); js = [int(x) for x in a.js.split(",")]
    res = {"items": "multrow.ext_4x8_low8 (gen_operands, seed 14204)", "n": a.n, "measure": "low 8 digits of a x (b_1..b_j) exact (as ext_4x8_low8); full = whole product",
           "tc_m": {}, "icot_si": {}}
    g = s5.load_gpt2(dev); g.load_state_dict(torch.load(os.path.join(ROOT, "boxes", "runs_mult", "m98_tc_4x4_s1", "model.pt"), map_location=dev)); g.eval()
    for j in js:
        lo, fu, sample = [], [], None
        for s in range(0, a.n, a.bs):
            idx = np.arange(s, min(a.n, s + a.bs))
            q = np.concatenate([DIGIT_IDS[A[idx]], np.full((len(idx), 1), TOK["*"]), DIGIT_IDS[B[idx, :j]], np.tile(QSEP, (len(idx), 1))], 1)
            dec = tcm_generate(g, q, max_new=40 + 24 * j)
            for i, row in enumerate(dec):
                l_, f_ = score(digits_after_hash(row, HASH), to_int(A[idx[i]]) * to_int(B[idx[i], :j]), 4 + j); lo.append(l_); fu.append(f_)
            if sample is None:
                sample = bc.ENC.decode([int(x) for x in dec[0]])
        res["tc_m"][str(j)] = {"low8": float(np.mean(lo)), "full": float(np.mean(fu)), "sample": sample}
        print("TC-M j", j, res["tc_m"][str(j)]["low8"], res["tc_m"][str(j)]["full"], round(time.time() - t0), flush=True)
    del g
    gi = load_icot(os.path.join(ROOT, "ext", "icotsi", "4x4", "state_dict.bin"), dev).eval()
    for j in js:
        lo, fu, sample = [], [], None
        src = [" ".join(map(str, A[i])) + " * " + " ".join(map(str, B[i, :j])) for i in range(a.n)]
        ids = [bc.tok(" " + x + " ") + [EOS] for x in src]
        for s in range(0, a.n, a.bs):
            dec = generate(gi, ids[s:s + a.bs], 24 + 4 * j)
            for i, row in enumerate(dec):
                txt = bc.ENC.decode([int(x) for x in row if x != EOS])
                d = [int(c) for c in txt.split("####", 1)[1].split() if c.isdigit()] if "####" in txt else None
                l_, f_ = score(d, to_int(A[s + i]) * to_int(B[s + i, :j]), 4 + j); lo.append(l_); fu.append(f_)
                if sample is None:
                    sample = txt
        res["icot_si"][str(j)] = {"low8": float(np.mean(lo)), "full": float(np.mean(fu)), "sample": sample}
        print("ICoT-SI j", j, res["icot_si"][str(j)]["low8"], res["icot_si"][str(j)]["full"], round(time.time() - t0), flush=True)
    res["sec"] = round(time.time() - t0)
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "ext_baselines_s152.json"), "w", newline=chr(10)), indent=1)


if __name__ == "__main__":
    main()
