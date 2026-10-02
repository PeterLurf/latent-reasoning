"""GSM8K step-2 prep (no registration needed; timing only): s/step of GPT-2 small CoT training on real GSM8K-Aug
sequences (official Coconut source, question + "\n" + steps + "### answer"), batch 32, fp32, AdamW, on one local card.
Uses boxes.core.GPT2KV (our trainable GPT-2). Prints s/step and the implied time per epoch of 385,620 examples.
    python scratch/gsm_time_proxy.py --device cuda:0 [--steps 60] [--bs 32] [--latent 0]
--latent K adds K extra sequential forward passes per step over the question prefix (a rough Coconut-stage proxy:
Coconut runs K+1 passes over a growing prefix with the latents filled in)."""
import argparse
import json
import os
import random
import sys
import time

import tiktoken
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from boxes.core import GPT2KV  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cuda:0"); ap.add_argument("--steps", type=int, default=60)
ap.add_argument("--bs", type=int, default=32); ap.add_argument("--latent", type=int, default=0)
a = ap.parse_args()
enc = tiktoken.get_encoding("gpt2")
lines = open(os.path.join(ROOT, "bench/data/gsm8k_aug_icotsi/train.txt"), encoding="utf-8").read().split("\n")
rng = random.Random(0); sample = rng.sample([l for l in lines if "||" in l], 4000)
seqs = []
for l in sample:
    q, rest = l.split("||", 1); steps, ans = rest.split("##")[0].strip(), rest.split("##")[-1].strip()
    seqs.append(enc.encode(q + "\n" + "\n".join(steps.split(" ")) + "\n### " + ans) + [50256])
sd = torch.load(os.path.join(ROOT, "prosqa/ckpt/gpt2_pretrained.bin"), map_location="cpu", weights_only=True)
m = GPT2KV(sd).to(a.device); opt = torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=0.01)
mean_len = sum(map(len, seqs)) / len(seqs)
def batch(i):
    b = seqs[i * a.bs:(i + 1) * a.bs]; L = max(map(len, b))
    x = torch.full((len(b), L), 50256); msk = torch.zeros(len(b), L)
    for j, s in enumerate(b):
        x[j, :len(s)] = torch.tensor(s); msk[j, :len(s)] = 1
    return x.to(a.device), msk.to(a.device)
torch.cuda.synchronize(a.device); t = None
for step in range(a.steps):
    if step == 10:
        torch.cuda.synchronize(a.device); t = time.time()
    x, msk = batch(step % (len(seqs) // a.bs))
    for _ in range(a.latent):                                      # Coconut-stage proxy: extra sequential passes
        with torch.no_grad():
            m(m.embed(x[:, : x.shape[1] // 2]))
    h, _ = m(m.embed(x[:, :-1])); logits = m.logits(h)
    loss = (F.cross_entropy(logits.reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1), reduction="none") * msk[:, 1:].reshape(-1)).sum() / msk[:, 1:].sum()
    opt.zero_grad(); loss.backward(); opt.step()
torch.cuda.synchronize(a.device); sps = (time.time() - t) / (a.steps - 10)
mem = torch.cuda.max_memory_allocated(a.device) / 2**30
print(json.dumps({"device": torch.cuda.get_device_name(a.device), "bs": a.bs, "latent_passes": a.latent,
                  "mean_tokens": round(mean_len, 1), "s_per_step": round(sps, 4),
                  "epoch_hours_385620": round(sps * 385620 / a.bs / 3600, 2), "max_mem_gb": round(mem, 2)}))
