"""CPU, tiny random models, no downloads: (1) cache object / attribute names in the local transformers; (2) RoPE re-rotation identity on a tiny
Llama with llama3 rope_scaling: keys cached at positions p+D equal keys cached at p re-rotated by D (fp32); (3) transplanting re-rotated latent rows
into an unshifted run gives the same next-token logits as the unshifted run (self-transplant across a shift)."""
import torch, transformers
torch.set_num_threads(2); torch.manual_seed(0)
from transformers import GPT2Config, GPT2LMHeadModel, LlamaConfig, LlamaForCausalLM
print("transformers", transformers.__version__)
g = GPT2LMHeadModel(GPT2Config(n_layer=2, n_embd=64, n_head=2, vocab_size=100, n_positions=256)).eval()
o = g(input_ids=torch.randint(0, 100, (1, 7)), use_cache=True); c = o.past_key_values
print("GPT2 cache type", type(c).__name__, "has layers", hasattr(c, "layers"), [a for a in ("key_cache", "value_cache") if hasattr(c, a)])
if hasattr(c, "layers"): print("  layer attrs", [a for a in ("keys", "values") if hasattr(c.layers[0], a)], tuple(c.layers[0].keys.shape))
cfg = LlamaConfig(hidden_size=128, intermediate_size=256, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, vocab_size=100,
                  max_position_embeddings=4096, rope_theta=500000.0,
                  rope_scaling={"rope_type": "llama3", "factor": 32.0, "high_freq_factor": 4.0, "low_freq_factor": 1.0, "original_max_position_embeddings": 8192})
m = LlamaForCausalLM(cfg).eval().float()
ids = torch.randint(0, 100, (1, 12)); D = 37
def run(off):
    with torch.no_grad():
        return m(input_ids=ids, position_ids=torch.arange(12)[None] + off, use_cache=True)
a, b = run(0), run(D)
ka = a.past_key_values.layers[0].keys; kb = b.past_key_values.layers[0].keys
rot = m.model.rotary_emb
def rerotate(k, delta):
    cos, sin = rot(k.transpose(1, 2), torch.full((1, k.shape[2]), delta))     # cos/sin at position delta
    cos, sin = cos[:, None], sin[:, None]
    rh = torch.cat((-k[..., k.shape[-1] // 2:], k[..., : k.shape[-1] // 2]), -1)
    return k * cos + rh * sin
print("layer-0 keys: max|k(p+D) - R(D)k(p)| = %.2e ; max|k(p+D) - k(p)| = %.2e" % ((kb - rerotate(ka, D)).abs().max(), (kb - ka).abs().max()))
for l in range(2):
    kl_a, kl_b = a.past_key_values.layers[l].keys, b.past_key_values.layers[l].keys
    print(" layer %d max|k(p+D) - R(D)k(p)| = %.2e, values diff %.2e" % (l, (kl_b - rerotate(kl_a, D)).abs().max(), (b.past_key_values.layers[l].values - a.past_key_values.layers[l].values).abs().max()))
# transplant: take shifted run's last 4 rows, re-rotate by -D, write into the unshifted cache, compare next-step logits
import copy
ca = copy.deepcopy(a.past_key_values)
for l in range(2):
    kb_l, vb_l = b.past_key_values.layers[l].keys, b.past_key_values.layers[l].values
    ca.layers[l].keys[:, :, -4:] = rerotate(kb_l[:, :, -4:], -D); ca.layers[l].values[:, :, -4:] = vb_l[:, :, -4:]
nxt = torch.tensor([[5]])
with torch.no_grad():
    la = m(input_ids=nxt, position_ids=torch.tensor([[12]]), past_key_values=copy.deepcopy(a.past_key_values), use_cache=True).logits
    lt = m(input_ids=nxt, position_ids=torch.tensor([[12]]), past_key_values=ca, use_cache=True).logits
    lr = m(input_ids=nxt, position_ids=torch.tensor([[12]]), past_key_values=copy.deepcopy(b.past_key_values), use_cache=True).logits
print("next-step logits: transplant(re-rotated) vs intact max diff %.2e ; raw shifted cache (no re-rotation, query at 12) vs intact %.2e" % ((lt - la).abs().max(), (lr - la).abs().max()))
