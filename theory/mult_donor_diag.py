"""THEORY note 114-4 follow-up / §121's shortcut-blocking check (Audit (a)), $0, CPU: does the carried slot-2 latent of a
Markov multiplication model carry the running sum S_2 or the multiplier prefix m = b_1 + 10 b_2? Causal donor test.
Markov chain on 4x4 (accel99 markov: segment j = [prefix a, slot j-1's latent (or nothing for j = 1), op j] -> slot j;
the masked/plain fork decodes step j's text from slot j). The recipient R = (a, b) runs ops 1-2; its slot-2 carried latent
is replaced by the donor D's slot-2 latent; op 3 then runs on R's prefix and b_3, and the step-3 decode's S_3 digits are
compared with three hypotheses:
  recipient-consistent   S_3(R) = a_R * (m_R + 100 b3_R)
  sum-donor-consistent   S_2(D) + 100 * a_R * b3_R           (the latent carries the running sum)
  prefix-donor-consistent a_R * (m_D + 100 b3_R)             (the latent carries the multiplier prefix)
  input-donor-consistent  a_D * (m_D + 100 b3_R)             (the latent carries a and the prefix; the segment uses the
                                                             carried a, not the visible one; added after the n = 20 smoke)
Donor types: SAMEPREFIX (m_D = m_R, a_D != a_R, so S_2 differs) and SAMESUM (a_D * m_D = a_R * m_R with m_D != m_R, found by
search over 4-digit a_D). Plus INTACT (no patch) for reference. A prefix carrier gives SAMEPREFIX -> recipient-consistent
and SAMESUM -> prefix-donor-consistent; a sum carrier gives SAMEPREFIX -> sum-donor-consistent and SAMESUM ->
recipient-consistent. Scored on S_3's 7 digits exactly (LSB first), and as mean per-digit agreement on digits 2-4 (the digits the step-3
add and the recompute determine; chance 0.1). In-context models (§99 CTRL / TF-SLOT: the chunk keeps the recipient's ops
visible) patch the slot-2 INPUT through mult98.MultGen.chunk's override; §107 PAUSE-4 adds its 3 pauses per segment;
§100 k = 2 models are skipped (step 3 starts from the snapped step-2 text, so no latent carries S_2 across it).
Output theory/results/mult_donor_diag.json.
    python theory/mult_donor_diag.py --runs a99_markov_4x4_s1 --n 300
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from boxes import s5  # noqa: E402
from boxes.accel99 import MultGen99  # noqa: E402
from boxes.mult98 import DIGIT_IDS, TOK, MultGen  # noqa: E402

TOK2DIG = {int(t): d for d, t in enumerate(DIGIT_IDS)}
STEP3_LEN = 17                                                             # " + P_3(7) ( S_3(7) )"


def lsb(v, L):
    return [int(c) for c in str(int(v)).zfill(L)[::-1]][:L]


@torch.no_grad()
def seg(m, prefix, prev, op):
    """One Markov segment [prefix, prev latent, op (, pauses)] -> (carried latent, slot hidden, cache), as accel99."""
    g = m.g; dt = g.embed(prefix).dtype
    parts = [g.embed(prefix)] + ([prev[:, None].to(dt)] if prev is not None else []) + [g.embed(m.digtok[op])[:, None]]
    if m.n_pause:
        parts.append(m.pause.to(dt)[None, None].expand(len(prefix), m.n_pause, -1))
    h, past = g(torch.cat(parts, 1), None); s_in = h[:, -1]; hs, past = g(s_in[:, None], past)
    return s_in, hs[:, 0], past


@torch.no_grad()
def run(m, A, B, patch2=None):
    """A (N, 4) a digits LSB, B (N, 4) b digits LSB. Returns the slot-2 carried latent (Markov) or slot-2 input (in-context
    chunk) and the step-3 decoded S_3 digits. In-context models (markov False) use mult98.MultGen.chunk's override."""
    prefix = torch.tensor(np.concatenate([DIGIT_IDS[A], np.full((len(A), 1), TOK["*"])], 1))
    if m.markov:
        s1, _, _ = seg(m, prefix, None, torch.tensor(B[:, 0]))
        s2, _, _ = seg(m, prefix, s1, torch.tensor(B[:, 1]))
        lat2 = s2 if patch2 is None else patch2
        _, h3, past = seg(m, prefix, lat2, torch.tensor(B[:, 2]))
        d = m.fork_greedy(h3, past, STEP3_LEN).numpy()
    else:
        m.override = None if patch2 is None else (1, patch2)
        _, decs, _, ins = MultGen.chunk(m, prefix, None, torch.tensor(B[:, :3]), step_len=[5, 15, STEP3_LEN], decode="last")
        m.override = None; s2 = ins[:, 1]; d = decs[-1].numpy()
    lp = 1 + 7                                                              # '+', 7 P digits, then '('
    S3 = np.array([[TOK2DIG.get(int(t), -1) for t in row[lp + 1:lp + 8]] for row in d])
    return s2, S3


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", nargs="+", default=["a99_markov_4x4_s1"]); ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--threads", type=int, default=3); ap.add_argument("--seed", type=int, default=11401)
    a = ap.parse_args(); torch.set_num_threads(a.threads)
    try:
        import psutil
        psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS if os.name == "nt" else 19)
    except Exception:
        pass
    rng = np.random.default_rng(a.seed); n = a.n
    aR = rng.integers(1000, 10000, n); bR = rng.integers(1000, 10000, n); mR = bR % 100; b3R = (bR // 100) % 10
    # SAMEPREFIX donors: same m, different a
    aP = (aR + rng.integers(1, 8999, n) - 1000) % 9000 + 1000; aP = np.where(aP == aR, (aR % 9000) + 1000 + 1, aP)
    bP = mR + 100 * rng.integers(0, 100, n); bP = np.where(bP < 1000, bP + 1000 * rng.integers(1, 10, n), bP)
    # SAMESUM donors: a_D * m_D = a_R * m_R with m_D != m_R, a_D 4-digit
    keep, aS, mS = [], [], []
    for i in range(n):
        S2 = int(aR[i]) * int(mR[i]); found = None
        for md in rng.permutation(np.arange(1, 100)):
            if md != mR[i] and S2 % md == 0 and 1000 <= S2 // md <= 9999:
                found = (S2 // md, int(md)); break
        keep.append(found is not None); aS.append(found[0] if found else 1000); mS.append(found[1] if found else 1)
    keep = np.array(keep); aS = np.array(aS); mS = np.array(mS)
    bS = mS + 100 * rng.integers(10, 100, n)
    D = lambda v: np.array([lsb(x, 4) for x in v])  # noqa: E731
    S3 = lambda aa, mm, b3: np.array([lsb(int(x) * (int(y) + 100 * int(z)), 7) for x, y, z in zip(aa, mm, b3)])  # noqa: E731
    out_p = os.path.join(ROOT, "theory", "results", "mult_donor_diag.json")
    res = json.load(open(out_p)) if os.path.exists(out_p) else {}
    res["_meta"] = {"n": n, "samesum_found": int(keep.sum())}
    for run_name in a.runs:
        args = json.load(open(os.path.join(ROOT, "boxes", "runs_mult", run_name, "args.json")))
        arm = args.get("arm99")
        if arm is None:
            continue                                                        # §100 k = 2: step 3 starts from snapped text (no slot-2 carrier)
        m = MultGen99(s5.load_gpt2("cpu"), "genm", std=False); m.markov = arm in ("markov", "pause4", "anchorread", "read") or bool(args.get("markov"))   # §121 arms are Markov
        if arm == "pause4":
            m.enable_mlat(1, 3)
        m.load_state_dict(torch.load(os.path.join(ROOT, "boxes", "runs_mult", run_name, "model.pt"), map_location="cpu")); m.eval()
        AR, BR = D(aR), D(bR)
        lat_R, s3_int = run(m, AR, BR)
        lat_P, _ = run(m, D(aP), D(bP)); _, s3_P = run(m, AR, BR, patch2=lat_P)
        lat_S, _ = run(m, D(aS), D(bS)); _, s3_S = run(m, AR, BR, patch2=lat_S)
        rec = S3(aR, mR, b3R)
        sumP = np.array([lsb(int(x) * int(y) + 100 * int(z) * int(w), 7) for x, y, z, w in zip(aP, mR, aR, b3R)])
        preS = S3(aR, mS, b3R)
        eq = lambda u, v, sel=None: float((u == v).all(1)[sel].mean()) if sel is not None else float((u == v).all(1).mean())  # noqa: E731
        inpP = S3(aP, mR, b3R); inpS = S3(aS, mS, b3R)                     # the latent carries (a, prefix) and the segment uses the carried a
        R = {"intact_recipient": eq(s3_int, rec),
             "SAMEPREFIX": {"recipient": eq(s3_P, rec), "sum_donor": eq(s3_P, sumP), "input_donor": eq(s3_P, inpP), "unchanged_vs_intact": eq(s3_P, s3_int)},
             "SAMESUM": {"recipient": eq(s3_S, rec, keep), "prefix_donor": eq(s3_S, preS, keep), "input_donor": eq(s3_S, inpS, keep), "unchanged_vs_intact": eq(s3_S, s3_int, keep)}}
        dg = lambda u, v, sel=None: float((u[:, 2:5] == v[:, 2:5]).mean(1)[sel].mean()) if sel is not None else float((u[:, 2:5] == v[:, 2:5]).mean())  # noqa: E731
        R["digits2_4"] = {"intact_recipient": dg(s3_int, rec),
                          "SAMEPREFIX": {"recipient": dg(s3_P, rec), "sum_donor": dg(s3_P, sumP), "input_donor": dg(s3_P, inpP)},
                          "SAMESUM": {"recipient": dg(s3_S, rec, keep), "prefix_donor": dg(s3_S, preS, keep), "input_donor": dg(s3_S, inpS, keep)}}
        res[run_name] = R; print(run_name, json.dumps(R), flush=True)
        json.dump(res, open(out_p, "w", newline="\n"), indent=1)


if __name__ == "__main__":
    main()
