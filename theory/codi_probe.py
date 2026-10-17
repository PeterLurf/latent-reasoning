"""Theory stream, broadened (exploratory): do public GPT-2 latent reasoners (CODI, SIM-CoT-on-CODI; GSM8K-Aug, 6
latents) show the toy model's mechanisms — thoughts that read earlier thoughts, and a horizon at the trained budget?

Self-contained inference (no CODI code): GPT-2 small with the checkpoint's LoRA merged (r 128, alpha 32, targets c_attn,
c_proj, c_fc), the latent projection (Linear-GELU-Linear-LN), CODI's GPT-2 test recipe (question + <bot>, latent_i =
prj(last hidden), <eot>, greedy decoding, answer = last number after "The answer is:").
Conditions:
  --n_latent N          latent iterations at inference (trained: 6)
  --block_from j        from latent j on (1-based), latent positions cannot attend to earlier latent positions (all
                        layers; they still see the question and <bot>); answer tokens are never masked
  --attn                record per-layer attention mass of each latent position on {question, <bot>, earlier latents, self}
    PYTHONPATH=<pylib> python theory/codi_probe.py --ckpt simcot_codi --n 400 --n_latent 6
"""
import argparse
import json
import math
import os
import re
import sys
import time

import torch
import torch.nn as nn

SP = os.environ.get("CODI_SP", "C:/Users/peter/AppData/Local/Temp/claude/D--/2f0034d5-4d85-51eb-880b-bab41adb0dff/scratchpad")
torch.set_num_threads(int(os.environ.get("THEORY_THREADS", "6")))


def load(ckpt):
    from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
    path = f"{SP}/hf/{ckpt}"
    if os.path.exists(f"{path}/model-00001-of-00001.safetensors"):
        from safetensors.torch import load_file
        sd = load_file(f"{path}/model-00001-of-00001.safetensors")
    else:
        sd = torch.load(f"{path}/pytorch_model.bin", map_location="cpu", weights_only=False)
    pre = "codi.base_model.model."
    scale = 32 / 128
    base = {}
    for k, v in sd.items():
        if not k.startswith(pre):
            continue
        k2 = k[len(pre):]
        if ".lora_" in k2:
            continue
        k2 = k2.replace(".base_layer", "")
        base[k2] = v.float()
    for k, v in sd.items():
        if k.startswith(pre) and ".lora_A." in k:
            mod = k[len(pre):].split(".lora_A.")[0]
            A = v.float(); B = sd[k.replace("lora_A", "lora_B")].float()
            base[mod + ".weight"] = base[mod + ".weight"] + scale * (B @ A).T      # Conv1D weight is (in, out)
    cfg = GPT2Config(vocab_size=base["transformer.wte.weight"].shape[0])
    cfg._attn_implementation = "eager"          # attention weights are needed for --attn
    m = GPT2LMHeadModel(cfg)
    missing, unexpected = m.load_state_dict(base, strict=False)
    missing = [x for x in missing if not x.endswith("attn.bias") and not x.endswith("masked_bias")]
    assert not missing, missing
    prj = nn.Sequential(nn.Dropout(0.0), nn.Linear(768, 768), nn.GELU(), nn.Linear(768, 768))
    prj.add_module("ln", nn.LayerNorm(768))
    prj.load_state_dict({k[4:]: v.float() for k, v in sd.items() if k.startswith("prj.")})
    tok = GPT2Tokenizer(f"{SP}/hf/simcot_codi/vocab.json", f"{SP}/hf/simcot_codi/merges.txt")
    tok.add_special_tokens({"pad_token": "[PAD]"})
    return m.eval(), prj.eval(), tok


def gsm8k(n):
    d = json.load(open(f"{SP}/hf/data/gsm8k_test.json", encoding="utf-8"))
    out = []
    for q, cot, a in list(zip(d["question"], d["cot"], d["answer"]))[:n]:
        q = q.strip().replace("  ", " ")
        a = float(str(a).replace(",", "").split("####")[-1].strip())
        nsteps = len(re.findall(r"<<", cot)) if cot else None
        out.append((q, a, nsteps))
    return out


def extract(s):
    s = s.replace(",", "")
    if "The answer is:" in s:
        s = s.split("The answer is:")[-1]
    nums = re.findall(r"-?\d+\.?\d*", s)
    return float(nums[-1]) if nums else float("inf")


@torch.no_grad()
def run(m, prj, tok, data, n_latent=6, block_from=None, attn=False, bs=16, max_new=40, dev="cpu", latent_capture=None, latent_override=None):
    """latent_capture (eval-only, §210 note 210-2): a list -> one (b, n_latent, D) tensor per chunk of the continuous-thought INPUTS (the prj output fed at
    latent positions 1..n_latent), in data order. latent_override: {"lat": (N, n_latent, D) tensor in data order, "pos": positions (1-based) replaced, "rows":
    optional (N,) bool of items replaced} -> at latent position j in pos, the input of each replaced item is lat[item, j - 1] (later positions are computed
    from the model's own outputs). Both None = the original computation."""
    V = m.config.vocab_size
    bot, eot = V - 2, V - 1
    wte = m.transformer.wte
    res, att_rows = [], []
    for i in range(0, len(data), bs):
        chunk = data[i:i + bs]
        tok.padding_side = "left"
        enc = {k: v.to(dev) for k, v in tok([q for q, _, _ in chunk], return_tensors="pt", padding=True).items()}
        ids = torch.cat([enc["input_ids"], torch.full((len(chunk), 1), bot, device=dev)], 1)
        mask = torch.cat([enc["attention_mask"], torch.ones(len(chunk), 1, dtype=torch.long, device=dev)], 1)
        pos = (mask.cumsum(1) - 1).clamp(min=0)
        out = m(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=True, output_hidden_states=True)
        pkv = out.past_key_values
        P = ids.shape[1]
        h = prj(out.hidden_states[-1][:, -1:, :])
        caps = [] if latent_capture is not None else None
        lat_mask = mask.clone()
        nextpos = pos[:, -1:] + 1
        for j in range(1, n_latent + 1):
            if latent_override is not None and j in latent_override["pos"]:          # 210-2: externally supplied latent inputs
                lo = latent_override["lat"][i:i + len(chunk), j - 1].to(h.device, h.dtype)[:, None, :]
                rw = latent_override.get("rows")
                h = lo if rw is None else torch.where(rw[i:i + len(chunk)].to(h.device)[:, None, None], lo, h)
            if caps is not None:
                caps.append(h[:, 0].detach().cpu())
            cur = torch.ones(len(chunk), 1, dtype=torch.long, device=dev)
            am = torch.cat([lat_mask, cur], 1)
            if block_from is not None and j >= block_from:
                am = am.clone(); am[:, P:P + j - 1] = 0                 # hide earlier latent positions
            o = m(inputs_embeds=h, attention_mask=am, position_ids=nextpos, past_key_values=pkv, use_cache=True,
                  output_hidden_states=True, output_attentions=attn)
            pkv = o.past_key_values
            if attn:
                row = []
                for a in o.attentions:                                   # (B, H, 1, T)
                    a = a[:, :, 0].mean(1)                               # (B, T)
                    row.append([float(a[:, :P - 1].sum(1).mean()), float(a[:, P - 1].mean()),
                                float(a[:, P:P + j - 1].sum(1).mean()) if j > 1 else 0.0, float(a[:, -1].mean())])
                att_rows.append((j, row))
            lat_mask = torch.cat([lat_mask, cur], 1)
            nextpos = nextpos + 1
            h = prj(o.hidden_states[-1][:, -1:, :])
        if caps is not None:
            latent_capture.append(torch.stack(caps, 1))
        # answer: <eot> then greedy decoding; answer tokens see everything
        x = wte(torch.full((len(chunk), 1), eot, device=dev))
        am = lat_mask
        gen = [[] for _ in chunk]
        done = torch.zeros(len(chunk), dtype=torch.bool, device=dev)
        for t in range(max_new):
            am = torch.cat([am, torch.ones(len(chunk), 1, dtype=torch.long, device=dev)], 1)
            o = m(inputs_embeds=x, attention_mask=am, position_ids=nextpos, past_key_values=pkv, use_cache=True)
            pkv = o.past_key_values
            nxt = o.logits[:, -1, :V - 1].argmax(-1)
            for b in range(len(chunk)):
                if not done[b]:
                    gen[b].append(int(nxt[b]))
            done |= nxt == tok.eos_token_id
            if done.all():
                break
            x = wte(nxt[:, None]); nextpos = nextpos + 1
        for (q, a, ns), g in zip(chunk, gen):
            txt = tok.decode([t for t in g if t != tok.eos_token_id])
            p = extract(txt)
            res.append({"ok": abs(p - a) < 1e-4, "pred": p, "gold": a, "nsteps": ns, "txt": txt[:80], "ntok": len(g)})
    out = {"acc": sum(r["ok"] for r in res) / len(res), "n": len(res), "rows": res}
    if attn:
        agg = {}
        for j, row in att_rows:
            agg.setdefault(j, []).append(row)
        out["attn"] = {j: [[float(sum(r[l][c] for r in rows) / len(rows)) for c in range(4)] for l in range(len(rows[0]))]
                       for j, rows in agg.items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="simcot_codi")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--n_latent", type=int, nargs="+", default=[6])
    ap.add_argument("--block_from", type=int, default=None)
    ap.add_argument("--attn", action="store_true")
    ap.add_argument("--out", default="theory/results/codi_probe.json")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    m, prj, tok = load(a.ckpt)
    m, prj = m.to(a.device), prj.to(a.device)
    data = gsm8k(a.n)
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for nl in a.n_latent:
        key = f"{a.ckpt}|n{a.n}|L{nl}|block{a.block_from}"
        t0 = time.time()
        o = run(m, prj, tok, data, nl, a.block_from, a.attn, dev=a.device)
        bydepth = {}
        for r in o["rows"]:
            if r["nsteps"] is not None:
                bydepth.setdefault(r["nsteps"], []).append(r["ok"])
        o["by_steps"] = {k: [sum(v) / len(v), len(v)] for k, v in sorted(bydepth.items())}
        res[key] = {k: v for k, v in o.items() if k != "rows"}
        res[key]["examples"] = o["rows"][:5]
        json.dump(res, open(a.out, "w"), indent=1)
        print("%s acc %.3f (%.0fs) by_steps %s" % (key, o["acc"], time.time() - t0,
              {k: round(v[0], 2) for k, v in o["by_steps"].items()}), flush=True)
        if a.attn:
            for j, rows in o["attn"].items():
                print("  latent %d: per-layer mass [question, bot, earlier latents, self] L0 %s | L5 %s | L11 %s" % (
                    j, [round(x, 2) for x in rows[0]], [round(x, 2) for x in rows[5]], [round(x, 2) for x in rows[11]]))


if __name__ == "__main__":
    main()
