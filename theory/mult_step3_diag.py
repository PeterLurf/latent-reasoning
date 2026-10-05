"""THEORY multiplication step-3 diagnosis ($0, CPU, existing 4x4 models; descriptive; the user's directive via Audit 00:05).
Step texts (mult98, the implicit-CoT repo's format, digits LSB first): step 1 " P_1"; step j > 1 " + P_j ( S_j )"; step 4
" + P_4" (no running sum); the answer " #### S_4". P_j = a * b_j (shifted), S_j = sum of P_1..P_j. So steps 1 and 4 need no
carried state; step 2 needs S_1; step 3 needs S_2; the answer needs S_3. per_slot_decode_acc in eval.json is the test
ROLLOUT (the chain runs on the model's own slot inputs; forks are read, never fed back).
Tests per model (bench 4x4 test, first N items):
  base  the standard chunk (start None, ops 1-4): per-slot accuracy; step-3 anatomy: P_3 part vs S_3 part exact, and
        S_3 per-digit accuracy (LSB first; S_3's two lowest digits equal S_2's).
  T1    restart from the TRUE step-2 text with op 3 alone (c = 1), and with ops 3-4 (c = 2, + answer): local step-3
        competence given S_2 as text.
  T2    restart from the TRUE step-1 text with ops 2-4 (c = 3, + answer): op 3 after ONE latent hand-off instead of two.
  carry step-3 per-digit error split by whether a carry enters that digit in S_2 + P_3 (P_3 as written, already
        shifted by its 2 leading zeros; from the true digits):
        a compute (addition) failure concentrates on carry digits; a lost carrier does not.
  probe linear (ridge, closed form) probes on the carried slot-2 input (the latent that feeds op 3), fitted on 4,000
        synthetic 4x4 problems (random a, b; seed 31) and scored on 1,000 more: S_2 digits vs the multiplier digits
        b_1, b_2 (10-way one-vs-rest least squares on standardised latents).
  T1 / T2 are out of distribution for models whose text-start chunks stopped at 6k (K-CURR) or never existed; they are
  reported but read only for the §100 k = 2 models (whose chunk 2 starts from the snapped step-2 text).
Output theory/results/mult_step3_diag.json.
    python theory/mult_step3_diag.py --n 600 --threads 3
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
from boxes.mult98 import DIGIT_IDS, TOK, MultGen, answer_correct, load  # noqa: E402

TOK2DIG = {int(t): d for d, t in enumerate(DIGIT_IDS)}
MODELS = {  # name -> (constructor kind, markov, m_lat, n_pause)
    "a99_markov_4x4_s1": ("a99", True, 1, 0), "a99_ctrl_4x4_s1": ("a99", False, 1, 0), "a99_tfslot_4x4_s1": ("a99", False, 1, 0),
    "a99_tfslot_4x4_s3": ("a99", False, 1, 0), "a107_pause4_4x4_s1": ("a99", True, 1, 3), "m100_gen2_4x4_s1": ("m98", False, 1, 0),
    "m100_mask2_4x4_s1": ("m98mask", False, 1, 0)}


def build(name):
    kind, markov, m_lat, n_pause = MODELS[name]; g = s5.load_gpt2("cpu")
    if kind == "a99":
        m = MultGen99(g, "genm", std=False); m.markov = markov
        if n_pause:
            m.enable_mlat(m_lat, n_pause)
    else:
        m = MultGen(g, "mask" if kind == "m98mask" else "genm")
    m.load_state_dict(torch.load(os.path.join(ROOT, "boxes", "runs_mult", name, "model.pt"), map_location="cpu")); m.eval()
    return m


def digits_of(tokens):
    return np.array([[TOK2DIG.get(int(t), -1) for t in row] for row in tokens])


@torch.no_grad()
def run_chunk(m, D, idx, c0, c1, bs):
    """Chunk over ops c0..c1-1 (0-based) from the TRUE step-c0 text (None if c0 = 0). -> decodes per op, answer ok, slot ins."""
    n = D["n"]; T = torch.tensor; decs = [[] for _ in range(c0, c1)]; ok = []; ins = []
    for s in range(0, len(idx), bs):
        ii = idx[s:s + bs]; prefix = T(np.concatenate([DIGIT_IDS[D["A"][ii]], np.full((len(ii), 1), TOK["*"])], 1))
        start = T(D["steps"][c0 - 1][ii]) if c0 > 0 else None; last = c1 == n
        _, dd, ans, inn = m.chunk(prefix, start, T(D["B"][ii, c0:c1]), step_len=[D["steps"][j].shape[1] for j in range(c0, c1)],
                                  ans_len=D["ans"].shape[1] if last else None)
        for jj, d in enumerate(dd):
            decs[jj].append(d.numpy())
        if last:
            ok.append(answer_correct(ans.numpy(), D["prod"][ii]))
        ins.append(inn.float().numpy())
    decs = [np.concatenate(x) for x in decs]
    acc = {c0 + jj + 1: float((d == D["steps"][c0 + jj][idx]).all(1).mean()) for jj, d in enumerate(decs)}
    return decs, (float(np.concatenate(ok).mean()) if ok else None), acc, np.concatenate(ins)


def anatomy(dec3, true3):
    """Step-3 tokens: '+', P_3 digits, '(', S_3 digits, ')'. Exact P part, exact S part, S per-digit accuracy."""
    lp = int(np.where(true3[0] == TOK["("])[0][0]); P = slice(1, lp); S = slice(lp + 1, true3.shape[1] - 1)
    p_ok = (dec3[:, P] == true3[:, P]).all(1); s_ok = (dec3[:, S] == true3[:, S]).all(1)
    return {"P3_exact": float(p_ok.mean()), "S3_exact": float(s_ok.mean()), "S3_exact_given_P3_ok": float(s_ok[p_ok].mean()) if p_ok.any() else None,
            "S3_digit_acc_lsb_first": (dec3[:, S] == true3[:, S]).mean(0).round(4).tolist()}


@torch.no_grad()
def synth_latents(m, n, bs, seed=31):
    rng = np.random.default_rng(seed); A = rng.integers(0, 10, (n, 4)); B = rng.integers(0, 10, (n, 4))
    A[:, 3] = rng.integers(1, 10, n); B[:, 3] = rng.integers(1, 10, n)                  # 4-digit operands (LSB first)
    a = A @ (10 ** np.arange(4)); s2 = a * (B[:, 0] + 10 * B[:, 1]); out = []
    for s in range(0, n, bs):
        prefix = torch.tensor(np.concatenate([DIGIT_IDS[A[s:s + bs]], np.full((len(A[s:s + bs]), 1), TOK["*"])], 1))
        _, _, _, ins = m.chunk(prefix, None, torch.tensor(B[s:s + bs]))
        out.append(ins[:, 1].float().reshape(len(A[s:s + bs]), -1).numpy())
    S2 = np.array([[int(c) for c in str(v).zfill(7)[::-1]] for v in s2])
    return np.concatenate(out), S2, B


def carry_split(dec3, true3, S2d, P3d):
    """Per S_3 digit k: accuracy when a carry enters digit k of S_2 + P_3 * 100 vs when none does."""
    lp = int(np.where(true3[0] == TOK["("])[0][0]); S = slice(lp + 1, true3.shape[1] - 1); L = S.stop - S.start
    add = np.zeros((len(S2d), L), int); add[:, :S2d.shape[1]] += S2d[:, :L]
    add[:, :min(L, P3d.shape[1])] += P3d[:, :L]                     # the dataset's P_3 is already shifted (2 leading zeros)
    carry_in = np.zeros_like(add); c = np.zeros(len(add), int)
    for k in range(L):
        carry_in[:, k] = c; c = (add[:, k] + c) // 10
    ok = dec3[:, S] == true3[:, S]; out = {}
    for k in range(L):
        c1 = carry_in[:, k] == 1
        out[k] = {"acc_carry": float(ok[c1, k].mean()) if c1.any() else None, "n_carry": int(c1.sum()),
                  "acc_nocarry": float(ok[~c1, k].mean()) if (~c1).any() else None}
    return out


def ridge_probe(Xtr, ytr, Xte, yte, lam=10.0):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6; A = (Xtr - mu) / sd; B = (Xte - mu) / sd
    Y = np.eye(10)[ytr]; W = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ Y)
    return float(((B @ W).argmax(1) == yte).mean())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=600); ap.add_argument("--bs", type=int, default=100)
    ap.add_argument("--threads", type=int, default=3); ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--probe_n", type=int, default=4000)
    a = ap.parse_args(); torch.set_num_threads(a.threads)
    try:
        import psutil
        psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS if os.name == "nt" else 19)
    except Exception:
        pass
    D = load(os.path.join(ROOT, "bench", "data", "mult_4x4_test.txt")); idx = np.arange(min(a.n, len(D["A"])))
    S2 = digits_of(D["steps"][1][idx]); lp2 = int(np.where(D["steps"][1][0] == TOK["("])[0][0]); S2 = S2[:, lp2 + 1:-1]
    outp = os.path.join(ROOT, "theory", "results", "mult_step3_diag.json")
    res = json.load(open(outp)) if os.path.exists(outp) else {}
    res["_meta"] = {"n": len(idx), "step_tokens": [int(D["steps"][j].shape[1]) for j in range(4)]}
    for name in a.models:
        if not os.path.exists(os.path.join(ROOT, "boxes", "runs_mult", name, "model.pt")):
            continue
        m = build(name); R = {}
        decs, ok, acc, ins = run_chunk(m, D, idx, 0, 4, a.bs)
        R["base"] = {"per_slot": acc, "answer": ok, "step3": anatomy(decs[2], D["steps"][2][idx])}
        _, _, acc1, _ = run_chunk(m, D, idx, 2, 3, a.bs); d12, ok12, acc12, _ = run_chunk(m, D, idx, 2, 4, a.bs)
        R["T1_restart_from_true_step2"] = {"op3_alone": acc1, "ops3_4": acc12, "answer": ok12, "step3": anatomy(d12[0], D["steps"][2][idx])}
        d2, ok2, acc2, _ = run_chunk(m, D, idx, 1, 4, a.bs)
        R["T2_restart_from_true_step1"] = {"ops2_4": acc2, "answer": ok2, "step3": anatomy(d2[1], D["steps"][2][idx])}
        P3 = digits_of(D["steps"][2][idx]); lp3 = int(np.where(D["steps"][2][0] == TOK["("])[0][0]); P3 = P3[:, 1:lp3]
        R["base"]["step3_carry_split"] = carry_split(decs[2], D["steps"][2][idx], S2, P3)
        X, S2s, Bs = synth_latents(m, a.probe_n + 1000, a.bs); h = a.probe_n; pr = {}
        for k in range(S2s.shape[1]):
            if len(np.unique(S2s[:h, k])) > 1:
                pr["S2_digit%d" % k] = ridge_probe(X[:h], S2s[:h, k], X[h:], S2s[h:, k])
        for k in (0, 1, 2, 3):
            pr["b%d" % (k + 1)] = ridge_probe(X[:h], Bs[:h, k], X[h:], Bs[h:, k])
        R["probe_slot2_latent"] = pr
        res[name] = R
        print(name, json.dumps(R)[:900], flush=True)
        json.dump(res, open(outp, "w", newline="\n"), indent=1)
        del m


if __name__ == "__main__":
    main()
