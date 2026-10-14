"""shared helpers for the MATH_contraction checks (generator copied from boxes/nested170.py; GPT-2 lower blocks; delta write maps).
on REAL GPT-2 block-6 keys of nested inputs (read-only use of D:/Latent/prosqa/ckpt/gpt2_pretrained.bin; the generator is re-implemented here,
copied from boxes/nested170.py, so nothing under D:/Latent is imported or written). CPU, 3 threads, idle priority."""
import ctypes, json, sys
ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x40)
import numpy as np, torch, torch.nn.functional as F, tiktoken
torch.set_num_threads(3); torch.manual_seed(0)

# ---------------- generator (copied from boxes/nested170.py) ----------------
P_ = 11; OPS = ["+", "-", "*", "/"]; PREC = {"+": 1, "-": 1, "*": 2, "/": 2}; ENC = tiktoken.get_encoding("gpt2")
VOCAB = [" %d" % v for v in range(P_)] + [" " + o for o in OPS] + [" (", " )", " =", " ."]
TID = {w: ENC.encode(w)[0] for w in VOCAB}; INV = {v: pow(v, P_ - 2, P_) for v in range(1, P_)}
def apply(o, a, b): return (a + b) % P_ if o == "+" else (a - b) % P_ if o == "-" else (a * b) % P_ if o == "*" else (a * INV[b]) % P_
PAIRS = {(o, v): [(a, b) for a in range(P_) for b in range(P_) if not (o == "/" and b == 0) and apply(o, a, b) == v] for o in OPS for v in range(P_)}
class Node:
    __slots__ = ("op", "l", "r", "v")
    def __init__(self, v=None, op=None, l=None, r=None): self.v, self.op, self.l, self.r = v, op, l, r
def generate(rng, n):
    t = int(rng.integers(0, P_)); root = Node(v=t); leaves = [root]
    for _ in range(n):
        i = int(rng.integers(0, len(leaves))); leaf = leaves.pop(i); o = OPS[int(rng.integers(0, 4))]; prs = PAIRS[(o, leaf.v)]; a, b = prs[int(rng.integers(0, len(prs)))]
        leaf.op, leaf.l, leaf.r, leaf.v = o, Node(v=a), Node(v=b), None; leaves.insert(i, leaf.r); leaves.insert(i, leaf.l)
    return root, t
def render(nd, parent=None, right=False):
    if nd.op is None: return [" %d" % nd.v]
    s = render(nd.l, nd.op, False) + [" " + nd.op] + render(nd.r, nd.op, True)
    if parent is not None and (PREC[nd.op] < PREC[parent] or (right and PREC[nd.op] == PREC[parent])): s = [" ("] + s + [" )"]
    return s
EOT = 50256

# ---------------- GPT-2 lower blocks (deltamem._block / gpt_lower) ----------------
sd = torch.load("D:/Latent/prosqa/ckpt/gpt2_pretrained.bin", map_location="cpu", weights_only=True)
sd = {k: v.float() for k, v in sd.items() if not k.endswith(".attn.bias")}
H, D, DH = 12, 768, 64
def block(i, h):
    B, T, _ = h.shape; pre = f"h.{i}."
    a = F.layer_norm(h, (D,), sd[pre + "ln_1.weight"], sd[pre + "ln_1.bias"])
    q, k, v = (a @ sd[pre + "attn.c_attn.weight"] + sd[pre + "attn.c_attn.bias"]).split(D, -1)
    q, k, v = (t.view(B, T, H, DH).transpose(1, 2) for t in (q, k, v))
    o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    h = h + o.transpose(1, 2).reshape(B, T, D) @ sd[pre + "attn.c_proj.weight"] + sd[pre + "attn.c_proj.bias"]
    a = F.layer_norm(h, (D,), sd[pre + "ln_2.weight"], sd[pre + "ln_2.bias"])
    m = F.gelu(a @ sd[pre + "mlp.c_fc.weight"] + sd[pre + "mlp.c_fc.bias"], approximate="tanh")
    return h + m @ sd[pre + "mlp.c_proj.weight"] + sd[pre + "mlp.c_proj.bias"]
@torch.no_grad()
def delta_kv(tok):
    """the memory's write inputs at init and gamma = 0: u = LN_6(h6) (block 6's ln_1), k / v = block 6's c_attn k / v rows; delta heads = heads 6..11."""
    x = torch.tensor([tok]); h = sd["wte.weight"][x] + sd["wpe.weight"][: len(tok)][None]
    for i in range(6): h = block(i, h)
    a = F.layer_norm(h, (D,), sd["h.6.ln_1.weight"], sd["h.6.ln_1.bias"])
    _, k, v = (a @ sd["h.6.attn.c_attn.weight"] + sd["h.6.attn.c_attn.bias"]).split(D, -1)
    k = k.view(len(tok), H, DH)[:, 6:].double(); v = v.view(len(tok), H, DH)[:, 6:].double()
    return F.normalize(k, dim=-1), v                                         # (L, 6, 64) unit keys, values

ALPHA = 1 / (1 + np.exp(-5.0)); BETA = 1.0
def pass_maps(kh, vh, alpha=ALPHA, beta=BETA):
    """one pass's affine map Delta -> Delta P + C for one head (Delta: v x k, right-multiplied, token order as write_seq)."""
    d = kh.shape[1]; Pm = torch.eye(d, dtype=torch.float64); C = torch.zeros(vh.shape[1], d, dtype=torch.float64)
    for t in range(kh.shape[0]):
        Ht = alpha * (torch.eye(d, dtype=torch.float64) - beta * torch.outer(kh[t], kh[t]))
        Pm = Pm @ Ht; C = C @ Ht + beta * torch.outer(vh[t], kh[t])
    return Pm, C

@torch.no_grad()
def delta_kv_emb(emb):
    """as delta_kv for input embeddings emb (T, D) at positions 0..T-1 (causal) -> unit keys / values of ALL T positions (T, 6, 64)."""
    h = emb[None] + sd["wpe.weight"][: emb.shape[0]][None]
    for i in range(6): h = block(i, h)
    a = F.layer_norm(h, (D,), sd["h.6.ln_1.weight"], sd["h.6.ln_1.bias"])
    _, k, v = (a @ sd["h.6.attn.c_attn.weight"] + sd["h.6.attn.c_attn.bias"]).split(D, -1)
    T = emb.shape[0]; k = k.view(T, H, DH)[:, 6:].double(); v = v.view(T, H, DH)[:, 6:].double()
    return F.normalize(k, dim=-1), v

def restricted(Pm, kh):
    """P restricted to span(keys) (P maps span(K) into itself): (top singular value, spectral radius) of U^T P U, U an orthonormal basis of span(K)."""
    U, s, _ = torch.linalg.svd(kh.T, full_matrices=False); U = U[:, s > 1e-8 * s.max()]
    R = U.T @ Pm @ U
    return float(torch.linalg.svdvals(R).max()), float(torch.linalg.eigvals(R).abs().max()), U.shape[1]
