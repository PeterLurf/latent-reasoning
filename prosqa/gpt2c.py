"""Self-contained GPT-2 forward pass and Coconut latent rollout for analysing a trained Coconut ProsQA checkpoint.

Plain torch (no transformers), so every attention probability and mask is under our control. Matches the official
implementation (facebookresearch/coconut): the continuous thought for latent position j is the final-layer hidden state
(after ln_f) at the position before it; KV-cache semantics are reproduced by recomputing the causal prefix, which
gives identical activations for every earlier position.
"""
from __future__ import annotations

import math
import re

import torch
import torch.nn.functional as F

START, END, LATENT = 50257, 50258, 50259      # tokenizer.add_tokens order in the official run.py
EOS = 50256


def load_state(path, device):
    sd = torch.load(path, map_location="cpu", weights_only=False)
    if "state_dict" in sd and not any(k.endswith("wte.weight") for k in sd):
        sd = sd["state_dict"]
    out = {}
    for k, v in sd.items():
        k2 = re.sub(r"^(module\.)?(_fsdp_wrapped_module\.)?(base_causallm\.)?(transformer\.)?", "", k)
        out[k2] = v.float().to(device)
    return out


class GPT2C:
    def __init__(self, sd, n_layer=None, n_head=12):
        if n_layer is None:                  # infer from the checkpoint (addendum N trains truncated GPT-2)
            n_layer = sum(1 for k in sd if k.startswith("h.") and k.endswith(".ln_1.weight"))
        self.sd, self.L, self.H = sd, n_layer, n_head
        self.wte, self.wpe = sd["wte.weight"], sd["wpe.weight"]

    def embed(self, ids):
        return self.wte[ids]

    @torch.no_grad()
    def forward(self, x, extra=None, attn_q=None):
        """x: (B, T, 768) input embeddings (positions 0..T-1). extra: additive (B, T, T) mask or None.
        attn_q: list of query positions whose attention probs to return, as (L, B, H, len(attn_q), T).
        Returns final hidden (after ln_f), logits, attention."""
        sd, B, T, D = self.sd, x.shape[0], x.shape[1], x.shape[2]
        h = x + self.wpe[:T][None]
        causal = torch.full((T, T), float("-inf"), device=x.device).triu(1)
        mask = causal[None] if extra is None else causal[None] + extra
        att_out = []
        hd = D // self.H
        for i in range(self.L):
            p = f"h.{i}."
            a = F.layer_norm(h, (D,), sd[p + "ln_1.weight"], sd[p + "ln_1.bias"])
            qkv = a @ sd[p + "attn.c_attn.weight"] + sd[p + "attn.c_attn.bias"]
            q, k, v = qkv.split(D, -1)
            q, k, v = (t.view(B, T, self.H, hd).transpose(1, 2) for t in (q, k, v))
            s = (q @ k.transpose(-1, -2)) / math.sqrt(hd) + mask[:, None]
            pr = s.softmax(-1)
            if attn_q is not None:
                att_out.append(pr[:, :, attn_q, :])
            o = (pr @ v).transpose(1, 2).reshape(B, T, D)
            h = h + o @ sd[p + "attn.c_proj.weight"] + sd[p + "attn.c_proj.bias"]
            a = F.layer_norm(h, (D,), sd[p + "ln_2.weight"], sd[p + "ln_2.bias"])
            m = F.gelu(a @ sd[p + "mlp.c_fc.weight"] + sd[p + "mlp.c_fc.bias"], approximate="tanh")
            h = h + m @ sd[p + "mlp.c_proj.weight"] + sd[p + "mlp.c_proj.bias"]
        h = F.layer_norm(h, (D,), sd["ln_f.weight"], sd["ln_f.bias"])
        logits = h @ self.wte.T
        return h, logits, (torch.stack(att_out) if attn_q is not None else None)

    mask_answer = False      # PROTOCOL6 addendum G model: <end-latent> and answer positions cannot see the question

    def _answer_mask(self, n_q, K, T, B):
        """Additive (B, T, T): rows <end-latent> (position n_q + 1 + K) onwards, columns before <start-latent> (n_q)."""
        m = torch.zeros(B, T, T, device=self.wte.device)
        m[:, n_q + 1 + K:, :n_q] = float("-inf")
        return m

    @torch.no_grad()
    def rollout(self, q_ids, K, answers, extra=None, trace=False):
        """Coconut inference on one question, batched over B conditions (extra: (B, T_max, T_max) additive masks
        indexed by absolute position, or None). q_ids: question tokens (question + "\\n"). answers: list of token
        lists scored after <end-latent>. Returns per-condition log-prob of each answer, the query positions of the
        K thought steps and, if trace, attention (L, B, H, K, T) from those positions (taken in the final pass)."""
        dev = self.wte.device
        B = 1 if extra is None else extra.shape[0]
        pre = torch.tensor(q_ids + [START], device=dev)
        x = self.embed(pre)[None].expand(B, -1, -1).clone()
        qpos = [len(q_ids) + j for j in range(K)]           # start token, then thoughts 1..K-1
        for j in range(K):
            T = x.shape[1]
            h, _, _ = self.forward(x, None if extra is None else extra[:, :T, :T])
            x = torch.cat([x, h[:, -1:, :]], 1)              # thought j+1
        lp, keep = [], None
        for ans in answers:
            tail = torch.tensor([END] + ans, device=dev)
            xx = torch.cat([x, self.embed(tail)[None].expand(B, -1, -1)], 1)
            T = xx.shape[1]
            ex = None if extra is None else extra[:, :T, :T]
            if self.mask_answer:
                am = self._answer_mask(len(q_ids), K, T, B)
                ex = am if ex is None else ex + am
            h, logits, att = self.forward(xx, ex, attn_q=qpos if trace else None)
            l = logits.log_softmax(-1)
            n0 = x.shape[1]                                   # position of END; answer token i sits at n0 + 1 + i
            tok = torch.tensor(ans, device=dev)
            lp.append(l[:, n0:n0 + len(ans), :].gather(-1, tok[None, :, None].expand(B, -1, 1)).squeeze(-1).sum(-1))
            if trace and keep is None:
                keep = att
        return torch.stack(lp, 1), qpos, keep
