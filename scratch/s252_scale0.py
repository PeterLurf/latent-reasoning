"""Note 252-3 SCALE stage 0 ($0, eval only; a mechanism read): does a base model answer the pass-1 question by the entity's NAME when the op sentences are permuted?
-> scratch/s252_scale0/probe.json (+ gate.json). Local 4060 (cuda:0) fp32 (or --device cpu).
Items (REG law, mul_s_p 0): CHAIN = TEST2 d 2; PERM = TEST-PERM d 2 (boxes.perm_render, 622fb717's renderer; d 2 -> S_2's sentence before S_1's); D1 = TEST2 d 1.
Scored stratum: no-wrap at step 1 (the unreduced step-1 value OW._NW_APPLY(op_1, S_0, b_1) in 0..10, so base-model arithmetic is correct); n >= 300 per set (else
more items from the same TEST2 stream, asserted). Question: ' How many <u> does <S_1> have? The answer is:' after the full text (= inject_st.cue_str(it, 1) + SFX).
Models: Qwen/Qwen2.5-0.5B (§231's revision 060db649..., HF, input [<|endoftext|>] + tok(' ' + prompt)) and GPT-2 small base (the repo trunk, load_trunk; input [EOT] +
enc(' ' + prompt)), both FEW-SHOT: 3 d 1 examples from a disjoint REG stream (OW.Stream(25200, TEST2 excl), the first 3 draws), each 'text How many <u> does <S_1> have?
The answer is: v', joined with the query by '\\n\\n'; s233_askname_s1 (the trunk fine-tuned on this format) ZERO-SHOT on its trained pass-1 input ([EOT] + enc(' ' + text +
' ' + QUESTION(S_1)) + SFX, asserted == tok_aw's think_cues[0]).
Prediction: argmax over the 11 candidates ' 0'..' 10' of the summed log-prob of the candidate's continuation tokens after the prompt (the tokenizer's own ids for the
candidate string, appended to the prompt ids; prefix-stability of that concatenation is checked in the gate). Per item: sig, v1, pred, the 11 scores.
    python -B scratch/s252_scale0.py gate probe [--device cuda:0]"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as Fn

ROOT = "D:/Latent"; sys.path.insert(0, ROOT); os.chdir(ROOT)
from boxes import inject_ow as IO  # noqa: E402
from boxes import inject_aw as AW  # noqa: E402
import boxes.inject_st as ST  # noqa: E402
import boxes.ow_words as OW  # noqa: E402
import boxes.perm_render as PR  # noqa: E402
import boxes.deltamem as DM  # noqa: E402
from boxes.nested170 import EOT  # noqa: E402

OUT = ROOT + "/scratch/s252_scale0"; QWEN = "Qwen/Qwen2.5-0.5B"; QREV = "060db6499f32faf8b98477b0a26969ef7d8b9987"; QBOS = 151643; SEED_FS = 25200; NMIN = 300
CANDS = [" %d" % v for v in range(11)]


def raw1(it):
    op = it["ops"][0]; b = it["values"][it["src_j"][0]] if op.endswith("_s") else it["operands"][0]; r = OW._NW_APPLY(op, it["values"][0], b)
    assert r % 11 == it["values"][1], ("raw1", op, it["values"][0], b, r, it["values"][1]); return r


def question(it):
    return " " + ST.cue_str(it, 1) + " The answer is:"


def sets_():
    s = IO.build_sets(mul_s_p=0); d1, d2 = s["test2"][1], s["test2"][2]
    nw = lambda its: [it for it in its if 0 <= raw1(it) <= 10]  # noqa: E731
    P_, info = PR.test_perm_sets({2: d2}, (2,)); perm = P_[2]
    for a_, b_ in zip(d2, perm):
        o_ = b_["order"]; assert o_.index("S2") < o_.index("S1"), "PERM d 2: S_2's sentence first"
    out = {"CHAIN": nw(d2), "PERM": [p_ for p_, c_ in zip(perm, d2) if 0 <= raw1(c_) <= 10], "D1": nw(d1)}
    fs_s = OW.Stream(SEED_FS, s["excl"], frames=OW.FRAME_TRAIN, mul_s_p=0, **IO.OWKW); fs = []
    while len(fs) < 3:                                                      # the first 3 NO-WRAP d 1 draws (a wrapped example would show mod-11 arithmetic; only no-wrap is scored)
        f = fs_s.draw(1)
        if 0 <= raw1(f) <= 10:
            fs.append(f)
    test_sigs = {it["sig"] for d in s["test2"] for it in s["test2"][d]}; assert not any(f["sig"] in test_sigs for f in fs), "few-shot: disjoint from TEST2"
    info["extended"] = {}
    if len(out["D1"]) < NMIN:                                               # Note 252-3: n < 300 -> more d 1 items from the same TEST2 stream (replayed, then continued)
        st = OW.Stream(IO.SEED_TEST2_REG, frames=OW.FRAME_TRAIN, mul_s_p=0, **IO.OWKW); rep = {d: [st.draw(d) for _ in range(IO.TEST_N)] for d in IO.TEST_DS}
        assert all([x["sig"] for x in rep[d]] == [x["sig"] for x in s["test2"][d]] for d in IO.TEST_DS), "TEST2 stream replay"
        seen = test_sigs | {f["sig"] for f in fs}; extra = []; tries = 0
        while len(out["D1"]) < NMIN and tries < 50000:
            it = st.draw(1); tries += 1
            if it["sig"] in seen:
                continue
            seen.add(it["sig"]); extra.append(it["sig"])
            if 0 <= raw1(it) <= 10:
                out["D1"].append(it)
        info["extended"]["D1"] = {"draws": tries, "new_items": len(extra), "sigs": extra}
    assert all(len(v) >= NMIN for v in out.values()), ("n < %d after extension" % NMIN, {k: len(v) for k, v in out.items()})
    return out, fs, info


def fewshot_prefix(fs):
    return "\n\n".join(f["text"] + question(f) + " %d" % f["values"][1] for f in fs) + "\n\n"


def score_trunk(g, prompts_ids, cand_ids, dev, bs=50):
    """the repo trunk (GPT-2 small): summed continuation log-prob per candidate (single- or multi-token)."""
    out = []
    for i in range(0, len(prompts_ids), bs):
        ch = prompts_ids[i:i + bs]
        for p_ in ch:
            sc = []
            seqs = [p_ + c for c in cand_ids]; T = max(len(q) for q in seqs); x = torch.full((len(seqs), T), EOT, dtype=torch.long)
            for r, q in enumerate(seqs):
                x[r, :len(q)] = torch.tensor(q)
            x = x.to(dev); D = g.w("wte.weight").shape[1]; hd = D // g.H; h = g.embed(x) + DM.wpe_slice(g, T)[None]
            for li in range(g.L):
                h = DM._block(g, li, h, hd)
            lp = Fn.log_softmax(g.logits(Fn.layer_norm(h, (D,), g.w("ln_f.weight"), g.w("ln_f.bias"))).float(), -1)
            for r, c in enumerate(cand_ids):
                sc.append(float(sum(lp[r, len(p_) - 1 + k, c[k]] for k in range(len(c)))))
            out.append(sc)
    return out


def score_hf(qm, prompts_ids, cand_ids, dev):
    out = []
    for p_ in prompts_ids:
        seqs = [p_ + c for c in cand_ids]; T = max(len(q) for q in seqs); x = torch.full((len(seqs), T), QBOS, dtype=torch.long); am = torch.zeros_like(x)
        for r, q in enumerate(seqs):
            x[r, :len(q)] = torch.tensor(q); am[r, :len(q)] = 1
        lp = Fn.log_softmax(qm(input_ids=x.to(dev), attention_mask=am.to(dev)).logits.float(), -1)
        out.append([float(sum(lp[r, len(p_) - 1 + k, c[k]] for k in range(len(c)))) for r, c in enumerate(cand_ids)])
    return out


def load_qwen(dev):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(QWEN, revision=QREV); qm = AutoModelForCausalLM.from_pretrained(QWEN, revision=QREV, dtype=torch.float32).to(dev).eval(); return tok, qm


def load_gpt2(dev, path=None):
    g = AW.load_trunk(dev, {"scratch": ""})
    if path:
        g.load_state_dict(torch.load(path, map_location=dev, weights_only=True))
    g.eval(); return g


def prompts(model, its, fs, tok=None):
    if model == "askname":
        return [[EOT] + IO.ENC.encode(" " + it["text"] + " " + ST.cue_str(it, 1)) + AW.SFX for it in its]
    pre = fewshot_prefix(fs)
    if model == "qwen":
        return [[QBOS] + tok(" " + pre + it["text"] + question(it), add_special_tokens=False)["input_ids"] for it in its]
    return [[EOT] + IO.ENC.encode(" " + pre + it["text"] + question(it)) for it in its]


def cands(model, tok=None):
    if model == "qwen":
        return [tok(c, add_special_tokens=False)["input_ids"] for c in CANDS]
    return [IO.ENC.encode(c) for c in CANDS]


def gate(dev):
    S, fs, info = sets_(); o = {"n": {k: len(v) for k, v in S.items()}, "perm_info": info, "fewshot_sigs": [f["sig"] for f in fs]}
    tok, qm = load_qwen("cpu"); del qm
    o["cand_tokens"] = {"qwen": cands("qwen", tok), "gpt2": cands("gpt2")}
    pq = prompts("qwen", S["CHAIN"][:50], fs, tok); pg = prompts("gpt2", S["CHAIN"][:50], fs)
    pre = fewshot_prefix(fs)
    o["prefix_stable_qwen"] = all([QBOS] + tok(" " + pre + it["text"] + question(it) + c, add_special_tokens=False)["input_ids"] == p_ + ci
                                  for it, p_ in zip(S["CHAIN"][:50], pq) for c, ci in zip(CANDS, o["cand_tokens"]["qwen"]))
    o["prefix_stable_gpt2"] = all([EOT] + IO.ENC.encode(" " + pre + it["text"] + question(it) + c) == p_ + ci
                                  for it, p_ in zip(S["CHAIN"][:50], pg) for c, ci in zip(CANDS, o["cand_tokens"]["gpt2"]))
    a = json.load(open("boxes/runs_fsq/s233_askname_s1/args.json")) if os.path.exists("boxes/runs_fsq/s233_askname_s1/args.json") else {}
    pa = prompts("askname", S["CHAIN"][:200], fs)
    o["askname_input_eq_think_cue1"] = all(p_ == AW.tok_aw(it, "value", "step", "answer")["think_cues"][0] for it, p_ in zip(S["CHAIN"][:200], pa))
    o["raw1_checked_all"] = True; o["fewshot_text"] = fewshot_prefix(fs)
    o["pass"] = (all(v >= NMIN for v in o["n"].values()) and o["askname_input_eq_think_cue1"]
                 and all(len(c) == 1 for c in o["cand_tokens"]["gpt2"]))
    o["note"] = "prefix_stable_* are reported (the scorer appends the candidate's own ids; a False means the joint tokenization would merge differently)"
    return o


def probe(dev):
    S, fs, info = sets_(); res = {"n": {k: len(v) for k, v in S.items()}, "perm_info": info, "fewshot": fewshot_prefix(fs), "models": {}}
    for model in ("qwen", "gpt2_base", "askname"):
        with torch.no_grad():
            if model == "qwen":
                tok, m = load_qwen(dev); cid = cands("qwen", tok); sc = lambda P: score_hf(m, P, cid, dev)  # noqa: E731
                mk = lambda its: prompts("qwen", its, fs, tok)  # noqa: E731
            else:
                m = load_gpt2(dev, "boxes/runs_fsq/s233_askname_s1/model.pt" if model == "askname" else None); cid = cands("gpt2"); sc = lambda P: score_trunk(m, P, cid, dev)  # noqa: E731
                mk = (lambda its: prompts("askname", its, fs)) if model == "askname" else (lambda its: prompts("gpt2", its, fs))  # noqa: E731
            r = {}
            for k, its in S.items():
                scs = np.array(sc(mk(its))); pred = scs.argmax(1); y = np.array([it["values"][1] for it in its])
                r[k] = {"n": len(its), "acc": float((pred == y).mean()), "items": {"sig": [it["sig"] for it in its], "v1": y.tolist(), "pred": pred.tolist(), "scores": scs.round(4).tolist()}}
            res["models"][model] = r
        del m; torch.cuda.empty_cache()
    res["def"] = ("Note 252-3 SCALE stage 0: REG TEST2 d 2 (CHAIN) / TEST-PERM d 2 (PERM, 622fb717 renderer) / TEST2 d 1 (D1), no-wrap step-1 stratum; pass-1 question; "
                  "argmax over ' 0'..' 10' by summed continuation log-prob; Qwen2.5-0.5B (rev 060db649) and GPT-2 base 3-shot (disjoint REG stream 25200), "
                  "s233_askname_s1 zero-shot on its trained pass-1 input; fp32")
    res["qwen_revision"] = QREV; res["device"] = dev
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("parts", nargs="+"); ap.add_argument("--device", default="cuda:0"); o_ = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    for p_ in o_.parts:
        r = {"gate": gate, "probe": probe}[p_](o_.device); json.dump(r, open(os.path.join(OUT, p_ + ".json"), "w", newline="\n"), indent=1, default=str)
        print(p_, ("pass " + str(r["pass"])) if p_ == "gate" else "written", flush=True)
