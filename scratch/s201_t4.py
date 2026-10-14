"""§201 T4 SNAP-CHAIN (note 201-7; reader theory/score_s201_t4.py; plan Part III / T4): when does writing a text checkpoint beat more latent passes?
fp32 (no autocast), the exact delta loop, one nested model per call; items: test_set, the first 1,000 per n, n in {9, 10, 11, 12, 14, 16}.
WRITE W at a segment pass t: the model's own aux decoder greedily decodes e_{t+1} (its own prompt; up to len(current expression) + 1 tokens). If the
symbols parse as a valid expression of the nested grammar (operand (op operand)*, operand = a number 0..10 or a bracketed expression; also
nested170.eval_tokens), the item restarts: think input [EOT, e_{t+1}, " ."], S = 0, the answer decoder sees e_{t+1}; else it continues latent.
HALT: the §201 T2 PRIMARY rule (SHIRYAEV-HEAD refit on the T1 train dump exactly as scratch/s201_t2.py; the threshold A), evaluated online per segment
on the same per-pass features (s201_t1's CLEAN features); a segment's own pass index counts from 1.
POLICIES (total cap 32 passes; at most 3 checkpoints; the final segment halts by the T2 rule, censored -> its last pass):
  LATENT_BEST  : no writes, the answer at the train-chosen best fixed K' (§201 T2's best_fixed_k_train)
  LATENT_HALT  : no writes, the T2 primary halt
  TIMEOUT7     : latent until the T2 halt fires; not fired by segment pass 7 -> W at 7 (an invalid parse continues latent to the halt)
  FIX2/4/7     : W at segment pass k (each of the first 3 segments), then the final segment by the T2 halt
  LEN_IN_RANGE : W at the first segment pass (1..16) whose decoded string has <= 7 operators (once), then the final segment by the T2 halt
-> <run>/eval_s201_snap.json {policies: {NAME: {acc_by_n, mean_passes_by_n, mean_tokens_by_n, mean_checkpoints_by_n}}, qW_kw7_by_n (level-0 greedy
   whole-string aux exact at pass 7 vs the true e_8), parse_valid_by_n (level-0 decode at pass 7), segment_k0_by_n (K' 0 accuracy of every
   post-checkpoint segment, TIMEOUT7), best_fixed_k_train, A, bs, card, dtype}
    python scratch/s201_t4.py <run> [device cuda:2] [--n_items 1000] [--ns 9,10,11,12,14,16]"""
import json
import os
import sys
import time

ROOT = "D:/Latent"; sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scratch")); os.chdir(ROOT)
os.environ.setdefault("BOXES_NB", "5")
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_grad_enabled(False)
import boxes.deltamem as DM  # noqa: E402
import boxes.halt173 as H  # noqa: E402
import boxes.nested170 as N  # noqa: E402
import s201_t1 as T1  # noqa: E402  (the write_seq recorder; retention)
import s201_t2 as T2  # noqa: E402

TCAP = 32; MAXCK = 3; DECMAX = 16; BS = 250
POLICIES = ("LATENT_BEST", "LATENT_HALT", "TIMEOUT7", "FIX2", "FIX4", "FIX7", "LEN_IN_RANGE")
INV = {v[0]: k for k, v in N.TOK.items()}                                  # token id -> symbol
NUMS = {" %d" % v for v in range(11)}; OPSYM = {" " + o for o in N.OPS}


def valid(syms):
    """strict nested-grammar check: expr := operand (op operand)*, operand := number | '(' expr ')'; plus nested170.eval_tokens."""
    pos = [0]

    def operand():
        if pos[0] < len(syms) and syms[pos[0]] in NUMS:
            pos[0] += 1; return True
        if pos[0] < len(syms) and syms[pos[0]] == " (":
            pos[0] += 1
            if not expr() or pos[0] >= len(syms) or syms[pos[0]] != " )":
                return False
            pos[0] += 1; return True
        return False

    def expr():
        if not operand():
            return False
        while pos[0] < len(syms) and syms[pos[0]] in OPSYM:
            pos[0] += 1
            if not operand():
                return False
        return True
    if not syms or not expr() or pos[0] != len(syms):
        return False
    try:
        N.eval_tokens(syms); return True
    except Exception:
        return False


def decode_syms(tokens):
    """greedy aux tokens -> symbols up to EOT (None if an unknown token appears before EOT)."""
    out = []
    for t in tokens:
        t = int(t)
        if t == N.EOT:
            return out
        if t not in INV:
            return None
        out.append(INV[t])
    return out                                                              # no EOT within L: the decoded prefix


def head_and_A(run):
    """the T2 primary head + threshold A refit exactly as scratch/s201_t2.py (train dump of this run)."""
    Dtr = np.load(os.path.join(T2.DUMP, run + "_train.npz")); Xtr, _, ctr, _ = T2.feats(Dtr); ntr = Dtr["n"]; TM = Xtr.shape[1] - 1
    T = np.arange(TM + 1); rows = (slice(None), slice(1, None)); head = T2.logistic(Xtr[rows].reshape(-1, Xtr.shape[-1]), (T[None, :] >= ntr[:, None])[rows].reshape(-1))
    p_hat = 1.0 / float(ntr.mean()); pre = T[None, :] < (ntr[:, None] - 1); pre[:, 0] = False; C_FA = float(1 - ctr[pre].mean())
    post = (T[None, :-1] >= ntr[:, None]) & (T[None, :-1] >= 1) & ctr[:, :-1]; c_d = float((~ctr[:, 1:][post]).mean()) + 0.002
    return head, p_hat * C_FA / (p_hat * C_FA + c_d)


def segment(m, its, dev, T, dec_passes, lazy=None):
    """one segment for a batch of items (each its own think input), passes 0..T: the T1 CLEAN features (s201_t1's per-pass recording, inlined so the
    state is available for decoding) + greedy aux decodes at dec_passes -> (feature dict of numpy arrays (b, T+1, ...), {t: [syms or None]})."""
    mem = m.mem; HA, HD = mem.h_add, mem.h_del; B = len(its); x, msk = m.think_inputs(its, dev); lut = set(int(v) for v in N.NUM_IDS)
    rpos = torch.tensor([len(N.ids(it["syms"])) + 1 for it in its], device=dev); L = max(len(it["syms"]) for it in its) + 1
    out = {k: [] for k in ("logits", "aux_bit", "aux_prob", "beta", "ret", "ri_add", "ri_del")}; S = mem.zero_state(B, dev); rprev = None; dec = {}
    nanh = torch.full((B, HD), float("nan"), device=dev); nanb = torch.full((B,), float("nan"), device=dev)
    with H.ReadTap() as tap:
        for t in range(T + 1):
            if t:
                T1.REC["on"] = True; Sp = S
                try:
                    S = m.write(S, x, msk)
                finally:
                    T1.REC["on"] = False
                d = T1.REC["d"]; T1.REC["d"] = None; mf = d["mask"].float()[..., None]
                out["beta"].append((d["be"] * mf).sum(1) / mf.sum(1)); out["ret"].append(T1.retention(Sp[:, HA:HA + HD], d))
            else:
                out["beta"].append(nanh); out["ret"].append(nanh)
            tap.arm(); lg = m.dec_logits(S, its, dev); tap.on = False; out["logits"].append(lg[:, N.NUM_IDS].float())
            xq = tap.x[torch.arange(B, device=dev), rpos]; r = H.read_vec(mem, xq, S); ra, rd = r[:, :HA * DM.DH], r[:, HA * DM.DH:]
            if rprev is None:
                out["ri_add"].append(nanb); out["ri_del"].append(nanb)
            else:
                out["ri_add"].append((ra - rprev[0]).norm(dim=1) / rprev[0].norm(dim=1).clamp_min(1e-30)); out["ri_del"].append((rd - rprev[1]).norm(dim=1) / rprev[1].norm(dim=1).clamp_min(1e-30))
            rprev = (ra, rd)
            if t:
                xa = torch.tensor([m.aux_prompt(t)] * B, device=dev)
                p1 = torch.softmax(m.g.logits(T1.fwd_read(m.g, m.g.embed(xa), mem, S)[:, -1]).float(), -1); t1 = p1.argmax(-1)
                p2 = torch.softmax(m.g.logits(T1.fwd_read(m.g, m.g.embed(torch.cat([xa, t1[:, None]], 1)), mem, S)[:, -1]).float(), -1)
                isnum = torch.tensor([int(v) in lut for v in t1.tolist()], device=dev)
                out["aux_bit"].append(isnum & (p2.argmax(-1) == N.EOT)); out["aux_prob"].append(p1[:, N.NUM_IDS].sum(-1) * p2[:, N.EOT])
                if t in dec_passes:
                    g_ = m.aux_greedy(S, t, L); dec[t] = [decode_syms(row) for row in g_]
            else:
                out["aux_bit"].append(torch.zeros(B, dtype=torch.bool, device=dev)); out["aux_prob"].append(nanb)
            if t and lazy is not None and t <= DECMAX:                     # LEN_IN_RANGE: decode only the rows whose <= 7-operator string is not found yet
                idx = [i for i in range(B) if not lazy[i]]
                if idx:
                    g_ = m.aux_greedy(S[idx], t, L); col = dec.setdefault(t, [None] * B)
                    for r_, i in enumerate(idx):
                        s_ = decode_syms(g_[r_]); col[i] = s_
                        if s_ is not None and sum(x_ in OPSYM for x_ in s_) <= 7:
                            lazy[i] = t
    return {k: torch.stack(v, 1).cpu().numpy() for k, v in out.items()}, dec


def run_level(m, its, dev, T, dec_passes, head, A, lazy=False):
    """segments for a list of items in batches -> (answers (n, T+1), halt (n,) = the first firing pass or T, fired (n,) bool, decodes {t: list},
    found (n,) = LEN_IN_RANGE's first pass with <= 7 operators (0 = none) when lazy)."""
    ans, pis, dec, found = [], [], {}, []; NI = len(its)
    for i in range(0, len(its), BS):
        ch = its[i:i + BS]; lz = [0] * len(ch) if lazy else None
        F, d = segment(m, ch, dev, T, dec_passes, lz)
        D = {**F, "t": np.array([it["t"] for it in ch])}
        X, a_, _, _ = T2.feats(D); rows = X[:, 1:]; p = np.zeros(a_.shape); p[:, 1:] = head(rows.reshape(-1, X.shape[-1])).reshape(len(a_), -1)
        ans.append(a_); pis.append(p); found.extend(lz or [0] * len(ch))
        for t, v in d.items():                                              # decode columns aligned with the item order (None = not decoded)
            dec.setdefault(t, [None] * NI)[i:i + len(v)] = v
    ans, pis = np.concatenate(ans), np.concatenate(pis); fire = pis >= A; fire[:, 0] = False; fired = fire.any(1)
    return ans, np.where(fired, fire.argmax(1), T), fired, dec, np.array(found)


def policy(name, its, m, dev, head, A, bk, lvl0):
    """-> per item (answer, total passes, decoded tokens, checkpoints), and every post-checkpoint segment's item (for T4-d's K' 0).
    Level-synchronous: a NON-final level (checkpoints < 3) runs only to its write pass (FIX k: k; TIMEOUT7: 7; LEN_IN_RANGE: <= 16, lazily decoded);
    an item that must continue latent (no write, or an invalid parse) is re-run from its segment start to the T2 halt (the computation is
    deterministic, so this equals continuing). The final level runs to the remaining budget and halts by the T2 rule (censored -> its last pass)."""
    n_it = len(its); res = [None] * n_it; k0 = []
    if name in ("LATENT_BEST", "LATENT_HALT"):
        ans, halt, fired, _, _ = lvl0
        for i in range(n_it):
            tp = bk if name == "LATENT_BEST" else int(halt[i]); res[i] = (ans[i, tp], tp, 0, 0)
        return res, k0
    kw = {"FIX2": 2, "FIX4": 4, "FIX7": 7, "TIMEOUT7": 7}.get(name)
    cur = [{"syms": it["syms"], "t": it["t"], "i": i, "used": 0, "tok": 0, "ck": 0} for i, it in enumerate(its)]; finish = []
    while cur:
        final = cur[0]["ck"] >= MAXCK or (name == "LEN_IN_RANGE" and cur[0]["ck"] >= 1)
        rem_max = max(TCAP - c["used"] for c in cur)
        if final:
            finish.extend(cur); break
        if name == "LEN_IN_RANGE":
            Tl = min(DECMAX, rem_max); ans, halt, fired, dec, found = run_level(m, cur, dev, Tl, set(), head, A, lazy=True)
        else:
            Tl = min(kw, rem_max); reuse = lvl0 is not None and cur[0]["ck"] == 0 and kw == 7
            ans, halt, fired, dec, found = (lvl0 if reuse else run_level(m, cur, dev, Tl, {kw}, head, A))
        nxt = []
        for j, c in enumerate(cur):
            rem = TCAP - c["used"]
            if name == "TIMEOUT7" and fired[j] and halt[j] <= 7 and halt[j] <= rem:
                res[c["i"]] = (ans[j, int(halt[j])], c["used"] + int(halt[j]), c["tok"], c["ck"]); continue     # halted before the timeout
            w = (int(found[j]) or None) if name == "LEN_IN_RANGE" else (kw if kw <= rem else None)
            s_ = dec.get(w, [None] * len(cur))[j] if w is not None else None
            if w is not None and s_ is not None and valid(s_):
                nxt.append({"syms": s_, "t": c["t"], "i": c["i"], "used": c["used"] + w, "tok": c["tok"] + len(s_) + 1, "ck": c["ck"] + 1})
            else:
                finish.append(c)                                            # continue latent in this segment, no further writes
        cur = nxt; k0.extend(nxt)
    if finish:                                                              # run the continuing / final segments to their budget, halt by T2
        T = max(TCAP - c["used"] for c in finish); ans, halt, fired, _, _ = run_level(m, finish, dev, T, set(), head, A)
        for j, c in enumerate(finish):
            rem = TCAP - c["used"]; h = min(int(halt[j]) if fired[j] else T, rem); res[c["i"]] = (ans[j, h], c["used"] + h, c["tok"], c["ck"])
    return res, k0


def main():
    run = sys.argv[1]; dev = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else "cuda:2"
    opt = lambda k, d: type(d)(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d   # noqa: E731
    n_items = opt("--n_items", 1000); NS = [int(v) for v in opt("--ns", "9,10,11,12,14,16").split(",")]
    rd = os.path.join(ROOT, "boxes", "runs_fsq", run); hj = json.load(open(os.path.join(rd, "eval_s201_halt.json"))); bk = int(hj["best_fixed_k_train"])
    kind, m, a = H.load_model(rd, dev); DM.FAST_ADD = bool(a.get("fast_add")); DM.CHUNK_DELTA = 0; m.eval(); head, A = head_and_A(run)
    TE = N.test_set(int(a.get("per_n", 1000))); t0 = time.time(); torch.cuda.reset_peak_memory_stats(dev)
    big = [it for it in TE if it["n"] == 16][:BS]; segment(m, big, dev, TCAP, set(range(1, DECMAX + 1)))     # memory check at the largest shape
    peak = torch.cuda.max_memory_allocated(dev) / 2 ** 30; assert peak < 7.0, ("memory check", peak); print(run, "memcheck %.2f GiB" % peak, flush=True)
    out = {p: {"acc_by_n": {}, "mean_passes_by_n": {}, "mean_tokens_by_n": {}, "mean_checkpoints_by_n": {}} for p in POLICIES}
    qw, pv, sk0 = {}, {}, {}
    for n in NS:
        its = [it for it in TE if it["n"] == n][:n_items]; tgt = np.array([it["t"] for it in its])
        lvl0 = run_level(m, [{"syms": it["syms"], "t": it["t"]} for it in its], dev, TCAP, {7}, head, A)
        d7 = lvl0[3].get(7, [None] * len(its)); true8 = [N.aux_of(it)[6][0] if len(N.aux_of(it)) >= 7 else None for it in its]
        qw[str(n)] = float(np.mean([s_ is not None and s_ == e_ for s_, e_ in zip(d7, true8)])); pv[str(n)] = float(np.mean([s_ is not None and valid(s_) for s_ in d7]))
        for p in POLICIES:
            res, k0 = policy(p, its, m, dev, head, A, bk, lvl0 if p != "LEN_IN_RANGE" else None)
            a_ = np.array([r_[0] for r_ in res]); out[p]["acc_by_n"][str(n)] = float((a_ == tgt).mean())
            out[p]["mean_passes_by_n"][str(n)] = float(np.mean([r_[1] for r_ in res])); out[p]["mean_tokens_by_n"][str(n)] = float(np.mean([r_[2] for r_ in res]))
            out[p]["mean_checkpoints_by_n"][str(n)] = float(np.mean([r_[3] for r_ in res]))
            if p == "TIMEOUT7" and k0:                                     # T4-d: K' 0 accuracy of every post-checkpoint segment
                l0 = []
                for i in range(0, len(k0), BS):
                    ch = k0[i:i + BS]; x, msk = m.think_inputs(ch, dev); S = m.mem.zero_state(len(ch), dev)
                    l0.append((m.dec_logits(S, ch, dev)[:, N.NUM_IDS].argmax(-1).cpu().numpy() == np.array([c["t"] for c in ch])))
                sk0[str(n)] = {"acc": float(np.concatenate(l0).mean()), "n_segments": len(k0)}
            elif p == "TIMEOUT7":
                sk0[str(n)] = {"acc": None, "n_segments": 0}
        print(run, "n", n, {p: round(out[p]["acc_by_n"][str(n)], 3) for p in POLICIES}, "qW7 %.3f valid7 %.3f" % (qw[str(n)], pv[str(n)]), "%.0fs" % (time.time() - t0), flush=True)
        json.dump({"policies": out, "qW_kw7_by_n": qw, "parse_valid_by_n": pv, "segment_k0_by_n": sk0, "best_fixed_k_train": bk, "A": A, "bs": BS,
                   "card": torch.cuda.get_device_name(dev), "dtype": "float32 (no autocast)", "memcheck_peak_gib": round(peak, 3), "n_items_per_n": n_items, "partial": n != NS[-1],
                   "def": "note 201-7 T4 SNAP-CHAIN; see the script docstring (scratch/s201_t4.py)"}, open(os.path.join(rd, "eval_s201_snap.json"), "w", newline="\n"), indent=1)


if __name__ == "__main__":
    main()
