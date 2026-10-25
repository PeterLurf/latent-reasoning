"""ANSWER-WRITE (AW) $0 build-side checks (AW_draft.md sec. 2 / 4 / 5). CPU, 2 threads, idle priority. Base GPT-2 only (prosqa/ckpt/gpt2_pretrained.bin):
no trained model is loaded and no gate reading is made here (the binding gate G-AW0 is aw_gate.py, run only after the draft is committed).
  TOK   on 4,000 items (d 1..8, stream 23473): every AW pass input x_t = [EOT] + enc(" " + text + " " + QUESTION(u, S_c)) + SFX has the text rows of
        tok_ow's think as a prefix, exactly 7 cue + 4 suffix tokens after them, no digit and no number token after the text, and ':' as its last token
        (the write position). Pass 1's input equals st_ask_name_local.tok_askname's class-(i) dec wherever that item's class is (i).
  W2    the single-token write (gather the ':' row of h_6 and h_w, write_seq on one token) equals the masked full-length write (DM.gpt_forward_mem_r with a
        one-hot write mask): state and qkv gradient within 1e-10 relative in fp64 (fp32 GEMM-reorder floor reported), FAST_ADD off / on, with and without the lower cache, at gamma 0 / 0.5 / 1, from a non-zero state with
        a non-zero read (W_o, gate randomised).
  LEAK  prototype of the build fixture FX-LEAK: with S_K fixed, the decoder logits are bitwise invariant to the think inputs (real cues / wrong cues /
        another item's text); a planted leak (the think pass's ':' argmax token appended to the decoder input) is detected; and the state channel is live
        (different think inputs -> different S_K -> different logits), so the fixture is not vacuous.
  PAIRS the d = 2 swap pairs (st_cue_check C5's law: items stream 23244, redraw rng 23245; first 1,000 kept) and their sig overlap with a replay of the
        12k-step training stream (d law default_rng(1 + 22331), OW.Stream(SEED_TRAIN, EXCL)); also TEST2 overlap.
  LEN   think lengths (plain / ST / AW) for the cost model.
    python -B docs/research_answerpos_2026-10-25/aw_checks.py            -> aw_checks.json
"""
import importlib.util
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2"); os.environ.setdefault("MKL_NUM_THREADS", "2"); os.environ.setdefault("BOXES_NB", "5")
try:
    import psutil
    psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS if os.name == "nt" else 19)
except Exception:  # noqa: BLE001
    pass
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(2)
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
import boxes.deltamem as DM  # noqa: E402
import boxes.inject_ow as IO  # noqa: E402
import boxes.ow_words as OW  # noqa: E402
from boxes.nested170 import EOT, NUM_IDS, pad  # noqa: E402

OUT = os.path.join(HERE, "aw_checks.json"); ENC = IO.ENC; SFX = IO.SFX; COLON = SFX[-1]; NUMS = set(int(v) for v in NUM_IDS)
_sp = importlib.util.spec_from_file_location("st_ask_name_local", os.path.join(ROOT, "docs", "research_steptold_2026-10-24", "st_ask_name_local.py"))
AN = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(AN)


def cue(it, c):
    return OW.QUESTION.format(u=it["unit"], A=it["names"]["spine"][c])


def aw_inputs(it):
    """AW pass inputs for c = 1..d (pass t reads input min(t, d)); the write position is the last token (':')."""
    return [[EOT] + ENC.encode(" " + it["text"] + " " + cue(it, c)) + SFX for c in range(1, it["d"] + 1)]


def chk_tok(n=4000):
    s = OW.Stream(23473, frames=OW.FRAME_TRAIN, **IO.OWKW); bad = []; n_i = 0; n_i_eq = 0; n_inp = 0
    for k in range(n):
        it = s.draw(1 + k % 8); th = IO.tok_ow(it)["think"]; xs = aw_inputs(it)
        for c, x in enumerate(xs, 1):
            n_inp += 1; tail = x[len(th):]; s_ = cue(it, c)
            ok = x[:len(th)] == th and len(tail) == 11 and tail[-4:] == SFX and x[-1] == COLON and not (set(tail) & NUMS) and not any(ch.isdigit() for ch in s_ + " The answer is:")
            if not ok:
                bad.append({"sig": it["sig"], "c": c})
        if it["d"] <= 2:
            o = AN.tok_askname(it)
            if o["ask_class"] == "i":
                n_i += 1; n_i_eq += int(o["dec"] == xs[0])
    return {"items": n, "inputs": n_inp, "bad": len(bad), "bad_first": bad[:5], "askname_class_i_items": n_i, "pass1_equals_askname_class_i_dec": n_i_eq,
            "pass": not bad and n_i > 0 and n_i_eq == n_i}


def write_masked(g, mem, x, S, pos, h_low):
    """the AW write by definition: gpt_forward_mem_r with a one-hot write mask at pos (only ':' writes)."""
    msk = torch.zeros(x.shape, dtype=torch.bool); msk[torch.arange(x.shape[0]), pos] = True
    return DM.gpt_forward_mem_r(g, g.embed(x), mem, S, mask=msk, h_low=h_low)[1]


def write_at(g, mem, x, S, pos, h_low):
    """the AW write as the build implements it: blocks (lower cached or not), read at block mem.layer, upper blocks, then write_seq on the gathered
    ':' rows only (one token, no mask). The same deltamem functions as gpt_forward_mem_r, in its order."""
    B, T = x.shape; D = g.D; hd = D // g.H; h = (g.embed(x) + DM.wpe_slice(g, T)[None]) if h_low is None else h_low; h6 = None
    for i in range(0 if h_low is None else mem.layer, g.L):
        if i == mem.layer:
            h6 = h; h = DM.read_fixed(mem, h, S)
        h = DM._block(g, i, h, hd)
    ar = torch.arange(B)
    return DM.write_seq(mem, h6[ar, pos][:, None], h[ar, pos][:, None], S, None)


def chk_w2(g):
    """fp32: report the GEMM-reorder floor (the (B, T, D) vs (B, 1, D) qkv matmuls take different BLAS paths). fp64 (deltamem's internal .float()
    casts redirected to float64 inside this check only): the two writes AND the gradient of a random readout of the new state w.r.t. mem.qkv.weight
    must agree to <= 1e-10 (relative): the same computation, reordering only (the 10-09 'exactness in fp64' rule)."""
    s = OW.Stream(23474, frames=OW.FRAME_TRAIN, **IO.OWKW); its = [s.draw(2) for _ in range(6)]
    seqs = [aw_inputs(it)[0] for it in its]; ids_, _ = pad(seqs); x = torch.tensor(ids_); pos = torch.tensor([len(q) - 1 for q in seqs])
    res = []; fa0 = DM.FAST_ADD; orig_float = torch.Tensor.float
    for dt in (torch.float32, torch.float64):
        if dt == torch.float64:
            torch.Tensor.float = lambda self, *a_, **k_: self.to(torch.float64)  # noqa: E731
            gg = IO.load_trunk("cpu", {"scratch": ""}).double()
        else:
            gg = g
        try:
            torch.manual_seed(23474); m = IO.InjectOW(gg, 6, "top", 6, 6, None).to(dt); mem = m.mem
            with torch.no_grad():
                mem.wo.weight.normal_(0, 0.02); mem.gate.weight.normal_(0, 0.02); mem.fade.fill_(1.0)
                S0 = torch.randn(len(its), 12, 64, 64, dtype=dt) * 0.05; Wr = torch.randn(len(its), 12, 64, 64, dtype=dt)
            for fast in (False, True):
                DM.FAST_ADD = fast
                for gam in (0.0, 0.5, 1.0):
                    with torch.no_grad():
                        mem.gamma.fill_(gam)
                    for cached in (False, True):
                        out = []
                        for fn in (write_masked, write_at):
                            mem.zero_grad(set_to_none=True); hl = DM.gpt_lower(gg, gg.embed(x), mem.layer).detach() if cached else None
                            Sn = fn(gg, mem, x, S0, pos, hl); (Sn * Wr).sum().backward(); out.append((Sn.detach(), mem.qkv.weight.grad.detach().clone()))
                        (A, gA), (Bv, gB) = out
                        rel = float((A - Bv).abs().max() / A.abs().max()); relg = float((gA - gB).abs().max() / gA.abs().max())
                        res.append({"dtype": str(dt).split(".")[-1], "fast_add": fast, "gamma": gam, "cached": cached, "bitwise": bool(torch.equal(A, Bv)),
                                    "rel_state": rel, "rel_grad_qkv": relg, "state_changed": bool(not torch.equal(A, S0))})
        finally:
            torch.Tensor.float = orig_float
        gg.zero_grad(set_to_none=True)
    DM.FAST_ADD = fa0
    f64 = [r for r in res if r["dtype"] == "float64"]; f32 = [r for r in res if r["dtype"] == "float32"]
    return {"cases": res, "fp32_rel_state_max": max(r["rel_state"] for r in f32), "fp32_rel_grad_max": max(r["rel_grad_qkv"] for r in f32),
            "fp64_rel_state_max": max(r["rel_state"] for r in f64), "fp64_rel_grad_max": max(r["rel_grad_qkv"] for r in f64),
            "pass": all(r["rel_state"] <= 1e-10 and r["rel_grad_qkv"] <= 1e-10 and r["state_changed"] for r in f64)}


def chk_leak(g):
    """prototype FX-LEAK on base GPT-2 + a randomised memory (read live)."""
    torch.manual_seed(23475); m = IO.InjectOW(g, 6, "top", 6, 6, None); mem = m.mem; DM.FAST_ADD = True
    with torch.no_grad():
        mem.wo.weight.normal_(0, 0.02); mem.fade.fill_(1.0); mem.gamma.fill_(1.0)
    s = OW.Stream(23475, frames=OW.FRAME_TRAIN, **IO.OWKW); its = [s.draw(2) for _ in range(6)]; items = [IO.tok_ow(it) for it in its]
    wrong = []
    for it in its:                                                          # wrong cue: S_0 at pass 1, S_1 at pass 2 (names only)
        wrong.append([[EOT] + ENC.encode(" " + it["text"] + " " + cue(it, c - 1)) + SFX for c in range(1, 3)])
    other = [aw_inputs(its[(i + 1) % len(its)]) for i in range(len(its))]   # another item's text and cues
    variants = {"real": [aw_inputs(it) for it in its], "wrong_cue": wrong, "other_text": other}

    def run_k(xs_items, K=8):
        S = mem.zero_state(len(xs_items), "cpu"); Td = 2
        xs = []
        for t in range(1, Td + 1):
            seqs = [q[min(t, len(q)) - 1] for q in xs_items]; ids_, _ = pad(seqs); xs.append((torch.tensor(ids_), torch.tensor([len(q) - 1 for q in seqs]), seqs))
        lastx = None
        for t in range(1, K + 1):
            x, pos, seqs = xs[min(t, Td) - 1]; S = write_at(g, mem, x, S, pos, None); lastx = (x, pos)
        return S, lastx

    def leaky_dec(S, lastx):
        """planted leak: append the think pass's ':' argmax number token to the decoder input."""
        x, pos = lastx; h = DM.gpt_forward_mem_r(g, g.embed(x), mem, S)[0]; tok = g.logits(h[torch.arange(len(pos)), pos])[:, NUM_IDS].argmax(-1)
        items2 = [dict(it, dec=it["dec"] + [int(NUM_IDS[int(k)])]) for it, k in zip(items, tok)]
        return m.dec_logits(S, items2, "cpu")
    with torch.no_grad():
        S_fix = torch.randn(len(its), 12, 64, 64) * 0.05; out = {}
        for nm, v in variants.items():
            S_K, lastx = run_k(v); out[nm] = {"dec_fixedS": m.dec_logits(S_fix, items, "cpu"), "dec_ownS": m.dec_logits(S_K, items, "cpu"), "leaky_fixedS": leaky_dec(S_fix, lastx)}
    inv = all(torch.equal(out["real"]["dec_fixedS"], out[k]["dec_fixedS"]) for k in variants)
    planted = any(not torch.equal(out["real"]["leaky_fixedS"], out[k]["leaky_fixedS"]) for k in ("wrong_cue", "other_text"))
    live = any(not torch.equal(out["real"]["dec_ownS"], out[k]["dec_ownS"]) for k in ("wrong_cue", "other_text"))
    dec_same = all(it["dec"] == [EOT] + ENC.encode(" " + r["text"]) + SFX for it, r in zip(items, its))
    return {"decoder_invariant_given_S": inv, "planted_leak_detected": planted, "state_channel_live": live, "decoder_tokens_text_only": dec_same,
            "pass": inv and planted and live and dec_same}


def chk_pairs(steps=12000, bs=64, cap=1000):
    r2 = np.random.default_rng(23245); s2 = OW.Stream(23244, frames=OW.FRAME_TRAIN, **IO.OWKW); kept = []; tried = 0
    while len(kept) < cap and tried < 3000:
        rec = s2.draw(2); don = OW.redraw(r2, rec)
        if don is None or don["sig"] == rec["sig"]:
            continue
        tried += 1; own, dfin, ca, cb = rec["answer"], don["answer"], IO.cf(rec, don, 1, "a"), IO.cf(rec, don, 1, "b")
        if len({own, dfin, ca}) == 3 and cb not in (own, dfin) and len(ENC.encode(rec["text"])) == len(ENC.encode(don["text"])):
            kept.append((rec, don, ca, cb))
    t0 = time.time(); S = IO.build_sets(); ex = S["excl"]; st = OW.Stream(OW.SEED_TRAIN, ex, frames=OW.FRAME_TRAIN, **IO.OWKW); rd = np.random.default_rng(1 + IO.SEED_D)
    train = set()
    for _ in range(steps):
        d = int(rd.integers(1, 3))
        for _ in range(bs):
            train.add(st.draw(d)["sig"])
    psig = {x["sig"] for p in kept for x in p[:2]}; t2 = {it["sig"] for it in S["test2"][2]}
    return {"tried": tried, "kept": len(kept), "cfa_eq_cfb_in_kept": sum(int(p[2] == p[3]) for p in kept), "train_items_replayed": steps * bs,
            "distinct_train_sigs": len(train), "pair_sigs": len(psig), "overlap_train": len(psig & train), "overlap_test2_d2": len(psig & t2),
            "test2_sha": S["shas"]["test2_sha"], "replay_sec": round(time.time() - t0), "pass": len(kept) == cap and not (psig & train) and not (psig & t2)}


def chk_len(n=1000):
    s = OW.Stream(23473 + 100, frames=OW.FRAME_TRAIN, **IO.OWKW); L = {}
    for d in (1, 2, 3, 4, 8):
        its = [s.draw(d) for _ in range(n)]; th = [len(IO.tok_ow(it)["think"]) for it in its]
        L[str(d)] = {"think_plain_mean": float(np.mean(th)), "think_st_mean": float(np.mean(th)) + 7, "think_aw_mean": float(np.mean(th)) + 11,
                     "think_aw_max": int(max(th)) + 11}
    return L


def main():
    t0 = time.time(); res = {"script": os.path.basename(__file__)}
    res["TOK"] = chk_tok(); print("TOK", res["TOK"]["pass"], flush=True)
    g = IO.load_trunk("cpu", {"scratch": ""}); g.eval()
    res["W2"] = chk_w2(g); print("W2", res["W2"]["pass"], flush=True)
    res["LEAK"] = chk_leak(g); print("LEAK", res["LEAK"], flush=True)
    res["LEN"] = chk_len(); print("LEN done", flush=True)
    res["PAIRS"] = chk_pairs(); print("PAIRS", {k: v for k, v in res["PAIRS"].items() if k != "test2_sha"}, flush=True)
    res["all_pass"] = all(res[k]["pass"] for k in ("TOK", "W2", "LEAK", "PAIRS")); res["sec"] = round(time.time() - t0)
    with open(OUT, "w", newline="\n") as f:
        json.dump(res, f, indent=1); f.write("\n")
    print("ALL_PASS", res["all_pass"], res["sec"], "s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
