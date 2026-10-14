"""THEORY note 204-1 (385c0db9; pre-stated, descriptive): an external test of STEPFACTOR's D1 ("result tokens in teacher-forced step strings are
memory-free") on SIM-CoT's own step decoder. Model: internlm/SIM_COT-GPT2-Coconut checkpoint_28 (prosqa/ckpt_ext/hf/simcot_coconut/checkpoint_28;
docs/GSM8K_REPRO.md). STEP 0 (done): the checkpoint holds the step decoder - expainable_llm.* (a separate 12-layer GPT-2 with its own lm_head) - and
the base input embedding (embedding.weight).
Protocol, from InternLM/SIM-CoT Coconut/coconut.py (explain_mode v1_aug, w_prompt False; args/gsm_simcot.yaml: c_thought 2, max_latent_stage 5):
  base: question + "\\n" + <|start-latent|>, 10 continuous thoughts (each the base model's last hidden state after ln_f at the previous position;
        bench.models.CoconutGSM's loop, which reproduced 44.8), z_0..z_9.
  step groups: each gold step "<<e>>" tokenised as encode(step + "\\n") (dataset.tokenize_sample), cut from the "<<" token through ">>" (coconut.py's
        group loop); stage k (1..5) reads the thought pair (z_{2k-2}, z_{2k-1}); stage 5 merges groups 5.. (only used here when it holds ONE step).
  decoder input: [z_a, z_b] + base-embedding(group tokens) + base-embedding(EOS); position ids 1..L; the expainable_llm transformer + its lm_head;
        targets = the group tokens and the EOS (the labels of coconut.py), predicted from the previous position.
Conditions per stage: TRUE (the item's own pair), ZERO (both thought vectors 0), MEAN (the mean pair over the valid items having that stage).
Items: GSM8K-Aug valid (500, gsm8k_data.load_aug('valid')); PRIMARY = stages k = 1..min(m, 4) (one gold step each); stage 5 DESCRIPTIVE and separate
(Theory): every item with m >= 5, the merged groups 5..m with "\\n" between ">>" and "<<" (coconut.py), each token keyed 's5:<category>'. Token categories: theory/stepfactor/
step_stats.char_cats / tok_cats (on "<<e>>": the leading-space form's categories without its first character), operand grounding from
gsm8k_data.parse_recomputable (opd_q = question literal, opd_r = earlier result, opd_k = constant, opd_u = item not parsed); the EOS target = "eot".
Readout: teacher-forced top-1 by category and condition; D1's pre-stated reading: value top-1 (ZERO / MEAN) within 0.05 of TRUE while operand top-1
drops >= 0.20; falsifier = a value drop >= 0.15.
-> boxes/runs_fsq/s204_simcot_d1/eval_simcot_d1.json
    python scratch/simcot_d1.py [device cuda:1]"""
import json
import os
import sys
import time

ROOT = "D:/Latent"; sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "prosqa")); sys.path.insert(0, os.path.join(ROOT, "theory", "stepfactor"))
import numpy as np  # noqa: E402
import tiktoken  # noqa: E402
import torch  # noqa: E402

import boxes.gsm8k_data as G  # noqa: E402
from gpt2c import END, EOS, GPT2C, START, load_state  # noqa: E402
from step_stats import char_cats, tok_cats  # noqa: E402

CK = os.path.join(ROOT, "prosqa", "ckpt_ext", "hf", "simcot_coconut", "checkpoint_28"); K = 10; C_TH = 2; STAGES = 5
LT, GT = 16791, 4211                                                       # "<<", ">>" (GPT-2)


def main():
    dev = sys.argv[1] if len(sys.argv) > 1 else "cuda:1"; t0 = time.time(); enc = tiktoken.get_encoding("gpt2")
    sd = load_state(CK, dev); emb = sd["embedding.weight"]
    base = GPT2C({k: v for k, v in sd.items() if not k.startswith(("expainable_llm.", "embedding."))})
    from transformers import GPT2Config, GPT2LMHeadModel
    dec = GPT2LMHeadModel(GPT2Config(tie_word_embeddings=False)).to(dev).eval()
    dsd = {k[len("expainable_llm."):]: v for k, v in sd.items() if k.startswith("expainable_llm.")}
    miss = dec.load_state_dict(dsd, strict=False); assert not miss.unexpected_keys and all(".attn.bias" in k or "masked_bias" in k for k in miss.missing_keys), miss
    R, _ = G.load_aug("valid", filter_bad=False); assert len(R) == 500
    items = []
    for r in R:
        steps = ["<<%s>>" % e for e in r["eqs"]]; groups = []
        for s in steps:                                                     # coconut.py's group loop on the flattened encode(step + "\n") tokens
            tk = enc.encode(s + "\n"); i = 0
            while i < len(tk):
                if tk[i] == LT:
                    g_ = [LT]; i += 1
                    while i < len(tk):
                        g_.append(tk[i])
                        if tk[i] == GT:
                            break
                        i += 1
                    groups.append(g_)
                i += 1
        tpl, _ = G.parse_recomputable(r); cats = []
        for t, e in enumerate(r["eqs"]):
            s_, cc = char_cats(e, tpl["eqt"][t] if tpl is not None else None); ids, tc = tok_cats(s_[1:], cc[1:]); cats.append((ids, tc))
        items.append({"r": r, "groups": groups, "cats": cats, "tpl_ok": tpl is not None})
    with torch.no_grad():                                                   # base thoughts z_0..z_9 (bench CoconutGSM's loop)
        for it in items:
            x = base.embed(torch.tensor(enc.encode(it["r"]["q"] + "\n") + [START], device=dev))[None]; zs = []
            for _ in range(K):
                h, _, _ = base.forward(x); zs.append(h[0, -1]); x = torch.cat([x, h[:, -1:, :]], 1)
            it["z"] = torch.stack(zs)
    stage_items = {}; n_want = 0
    for ii, it in enumerate(items):
        m = len(it["r"]["eqs"])
        for k in range(1, min(m, 4) + 1):                                   # PRIMARY: stages 1..4, one gold step each
            n_want += 1; ids, tc = it["cats"][k - 1]
            if len(it["groups"]) >= k and it["groups"][k - 1] == ids:      # the tokenisations agree (else skipped, counted)
                stage_items.setdefault(k, []).append(ii)
        if m >= 5:                                                          # DESCRIPTIVE (Theory): stage 5 = groups 5.. merged with "\n" between ">>" and "<<"
            n_want += 1
            if len(it["groups"]) == m and all(it["groups"][j] == it["cats"][j][0] for j in range(4, m)):
                stage_items.setdefault(5, []).append(ii)

    def target(it, k):
        """(target ids incl. EOS, categories; stage 5 tags each token with its step 's5:' prefix and the separators 'sep')."""
        if k <= 4:
            ids, tc = it["cats"][k - 1]; return ids + [EOS], tc + ["eot"]
        ids, cat = [], []
        for j in range(4, len(it["r"]["eqs"])):
            if ids:
                ids.append(198); cat.append("s5:sep")
            ids += it["cats"][j][0]; cat += ["s5:" + c_ for c_ in it["cats"][j][1]]
        return ids + [EOS], cat + ["s5:eot"]
    mean_z = {k: torch.stack([items[i]["z"][2 * k - 2:2 * k] for i in v]).mean(0) for k, v in stage_items.items()}
    R_ = {}; skipped = n_want - sum(len(v) for v in stage_items.values())
    with torch.no_grad():
        for k, idx in sorted(stage_items.items()):
            for cond in ("TRUE", "ZERO", "MEAN"):
                for i in idx:
                    it = items[i]; tg, cat = target(it, k)
                    zp = it["z"][2 * k - 2:2 * k] if cond == "TRUE" else (torch.zeros(2, 768, device=dev) if cond == "ZERO" else mean_z[k])
                    inp = torch.cat([zp, emb[torch.tensor(tg, device=dev)]], 0)[None]; L = inp.shape[1]
                    lg = dec(inputs_embeds=inp, attention_mask=torch.ones(1, L, device=dev), position_ids=torch.arange(1, L + 1, device=dev)[None]).logits[0]
                    pred = lg[C_TH - 1:C_TH - 1 + len(tg)].argmax(-1).tolist()
                    for p_, t_, c_ in zip(pred, tg, cat):                   # stage 5's tokens never enter the primary (unprefixed) keys
                        for key in (((cond, c_), (cond, "k%d:%s" % (k, c_)), (cond, "all")) if k <= 4 else ((cond, c_), (cond, "s5:all"))):
                            R_.setdefault(key, []).append(p_ == t_)
    out = {"top1": {}, "n": {}}
    for (cond, c_), v in sorted(R_.items()):
        out["top1"].setdefault(cond, {})[c_] = float(np.mean(v)); out["n"].setdefault(cond, {})[c_] = len(v)
    T = out["top1"]
    def d(c_, cond):
        return (T["TRUE"].get(c_, np.nan) - T[cond].get(c_, np.nan)) if c_ in T["TRUE"] else None
    out["d1"] = {cond: {"value_drop": d("val", cond), "operand_q_drop": d("opd_q", cond), "operand_r_drop": d("opd_r", cond), "operator_drop": d("op", cond),
                        "format_drop": d("fmt", cond)} for cond in ("ZERO", "MEAN")}
    for cond, v in out["d1"].items():
        opd = [x for x in (v["operand_q_drop"], v["operand_r_drop"]) if x is not None]
        v["reading"] = ("FALSIFIED (value drop >= 0.15)" if (v["value_drop"] is not None and v["value_drop"] >= 0.15) else
                        "D1-AS-PREDICTED" if (v["value_drop"] is not None and abs(v["value_drop"]) <= 0.05 and opd and min(opd) >= 0.20) else "NEITHER")
    out.update({"n_stage_items": {str(k): len(v) for k, v in stage_items.items()}, "n_skipped_tokenisation": int(skipped), "card": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu",
                "dtype": "float32", "sec": round(time.time() - t0), "checkpoint": "internlm/SIM_COT-GPT2-Coconut checkpoint_28",
                "def": "note 204-1: SIM-CoT step decoder teacher-forced on gold '<<e>>' + EOS reading (TRUE / ZERO / MEAN) thought pairs; top-1 by step_stats category"})
    od = os.path.join(ROOT, "boxes", "runs_fsq", "s204_simcot_d1"); os.makedirs(od, exist_ok=True)
    json.dump(out, open(os.path.join(od, "eval_simcot_d1.json"), "w", newline="\n"), indent=1)
    print(json.dumps({c: {k_: round(v_, 3) for k_, v_ in T[c].items() if ":" not in k_} for c in T}), flush=True); print(json.dumps(out["d1"]), flush=True)


if __name__ == "__main__":
    main()
