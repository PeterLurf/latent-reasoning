"""PROTOCOL6 addendum G: bypass-blocked Coconut. Answer tokens may not attend to anything before <|start-latent|>
(question and statements), so the answer can reach the graph only through the continuous thoughts.

    python patch_mask.py /workspace/coconut       (after patch_coconut.py; idempotent)

Mechanics: GPT-2 keeps its default SDPA attention; torch's scaled_dot_product_attention is wrapped to add MASK.extra
(and GPT2Attention._attn likewise for the eager path used by the test) (B, 1, T, T additive, absolute
positions, sliced to the current query/key window so KV-cached passes are covered); Coconut.forward and
Coconut.generate build MASK.extra from input_ids. Rule: query positions at or after the answer marker "###" (token
21017) — in generation, every position after <|end-latent|> — get -inf on key positions before <|start-latent|>.
MASK.check records the largest attention mass that crosses the mask (must be 0).
"""
import os
import sys

d = sys.argv[1]
MARK = "# PROTOCOL6-G mask"

cp = os.path.join(d, "coconut.py")
src = open(cp).read()
if MARK not in src:
    assert "vectorised feedback" in src, "apply patch_coconut.py first"
    head = f'''{MARK}
import types as _types
from transformers.models.gpt2 import modeling_gpt2 as _mg

MASK = _types.SimpleNamespace(extra=None, check=False, crossed=0.0)
ANS_ID = 21017                                    # "###": first token of every answer


def _build_extra(input_ids, start_id, end_id, generating):
    """(B, 1, T, T) additive mask: answer queries may not see keys before <|start-latent|>."""
    B, T = input_ids.shape
    dev = input_ids.device
    ar = torch.arange(T, device=dev)
    start = (input_ids == start_id).float().argmax(1)
    endp = (input_ids == end_id).float().argmax(1)
    if generating:
        qfrom = endp
    else:
        has = (input_ids == ANS_ID).any(1)
        qfrom = torch.where(has, (input_ids == ANS_ID).float().argmax(1), torch.full_like(start, T))
    # <|end-latent|> and every answer position (the <|start-latent|> position stays open: its final state is thought 1)
    blockq = (ar[None, :] >= qfrom[:, None]) | (ar[None, :] == endp[:, None])
    blockk = ar[None, :] < start[:, None]                       # (B, T) keys
    m = (blockq[:, :, None] & blockk[:, None, :])
    return torch.zeros(B, 1, T, T, device=dev).masked_fill(m[:, None], float("-inf")), m


_orig_attn = _mg.GPT2Attention._attn


def _attn_masked(self, query, key, value, attention_mask=None, head_mask=None):
    if MASK.extra is not None:
        q, k = query.size(-2), key.size(-2)
        ex = MASK.extra[:, :, k - q:k, :k].to(query.dtype)
        attention_mask = ex if attention_mask is None else attention_mask + ex
        if MASK.check:
            out, w = _orig_attn(self, query, key, value, attention_mask, head_mask)
            blocked = torch.isinf(ex) & (ex < 0)
            MASK.crossed = max(MASK.crossed, float((w * blocked).sum(-1).max()))
            return out, w
    return _orig_attn(self, query, key, value, attention_mask, head_mask)


_mg.GPT2Attention._attn = _attn_masked

_orig_sdpa = torch.nn.functional.scaled_dot_product_attention


def _sdpa_masked(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, **kw):
    """SDPA path (default, memory-efficient): causal/padding mask + MASK.extra; same rule as the eager path."""
    if MASK.extra is None:
        return _orig_sdpa(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, **kw)
    q, k = query.size(-2), key.size(-2)
    ex = MASK.extra[:, :, k - q:k, :k].to(query.dtype)
    if attn_mask is None:
        base = torch.full((q, k), float("-inf"), device=query.device, dtype=query.dtype).triu(k - q + 1)
    elif attn_mask.dtype == torch.bool:
        base = torch.zeros(attn_mask.shape, device=query.device, dtype=query.dtype).masked_fill(~attn_mask, float("-inf"))
    else:
        base = attn_mask
    return _orig_sdpa(query, key, value, attn_mask=base + ex, dropout_p=dropout_p, is_causal=False, **kw)


torch.nn.functional.scaled_dot_product_attention = _sdpa_masked
'''
    src = src.replace("Outputs = namedtuple(", head + "\n\nOutputs = namedtuple(", 1)
    # training / scoring forward: build the mask from the full input_ids once per call
    src = src.replace('''        logits = []

        latent_indices = (''', '''        logits = []
        MASK.extra = _build_extra(input_ids, self.start_latent_id, self.end_latent_id,
                                  getattr(self, "_generating", False))[0]

        latent_indices = (''', 1)
    src = src.replace('''        logits = torch.cat(logits, dim=-2)''', '''        logits = torch.cat(logits, dim=-2)
        if not getattr(self, "_generating", False):
            MASK.extra = None''', 1)
    # generation: every generated token is an answer token
    src = src.replace('''        tokens = input_ids[0].detach().tolist()''', '''        tokens = input_ids[0].detach().tolist()
        self._generating = True''', 1)
    src = src.replace('''        # get other tokens
        for _ in range(max_new_tokens - 1):
            outputs = self.base_causallm(inputs_embeds=new_inputs_embeds)''', '''        # get other tokens
        base_ids = input_ids[0].tolist()
        for _ in range(max_new_tokens - 1):
            L = new_inputs_embeds.shape[1]
            ids = torch.tensor(base_ids + [0] * (L - len(base_ids)), device=input_ids.device)[None]
            MASK.extra = _build_extra(ids, self.start_latent_id, self.end_latent_id, True)[0]
            outputs = self.base_causallm(inputs_embeds=new_inputs_embeds)''', 1)
    src = src.replace('''        if synced_gpus:''', '''        self._generating = False
        MASK.extra = None
        if synced_gpus:''', 1)
    open(cp, "w").write(src)
    print("MASK patch applied to coconut.py")
else:
    print("MASK patch already in coconut.py")

print("run.py unchanged (SDPA attention with the wrapped kernel)")
