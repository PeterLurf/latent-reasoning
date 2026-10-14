"""MATH_contraction check N1: the init-time open-loop Jacobian of one delta pass, P = alpha^L H_1 ... H_L (H = I - beta k k^T, beta = 1),
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

rng = np.random.default_rng(199); out = {}
for n in [1, 2, 4, 6, 8, 9, 12]:
    rows = []; states = []
    for it in range(24):
        root, _ = generate(rng, n); tok = [EOT] + [TID[s] for s in render(root)] + [TID[" ."]]
        kd, vd = delta_kv(tok); L = len(tok)
        sv_all, comp, gram = [], [], []
        maps = []
        for hh in range(6):
            Pm, C = pass_maps(kd[:, hh], vd[:, hh]); maps.append((Pm, C))
            s = torch.linalg.svdvals(Pm); sv_all.append(s.numpy())
            r = int(torch.linalg.matrix_rank(kd[:, hh], tol=1e-6)); comp.append(r)
            G = kd[:, hh] @ kd[:, hh].T; off = G[~torch.eye(L, dtype=torch.bool)].abs()
            gram.append(float(off.mean()))
            # energy identity: |v|^2 - |Q v|^2 = sum beta(2-beta)(k.v_(tau))^2 (Q = H_1..H_L without alpha), checked on a random v
            vv = torch.randn(64, dtype=torch.float64); cur = vv.clone(); diss = 0.0
            for t in range(L):                                                # row-vector action v^T H_1 H_2 ... = apply H_1 first
                c = float(kd[t, hh] @ cur); diss += BETA * (2 - BETA) * c * c; cur = cur - BETA * c * kd[t, hh]
            assert abs(float(vv @ vv) - float(cur @ cur) - diss) < 1e-9
        sv = np.stack(sv_all)
        rows.append(dict(L=L, top=float(sv.max()), med=float(np.median(sv)), mean_log=float(np.log(sv + 1e-300).mean()),
                         n_at_alphaL=float(np.mean([(s > ALPHA ** L * (1 - 1e-9)).sum() for s in sv])), rankK=float(np.mean(comp)),
                         frac_le_half=float((sv <= 0.5 * ALPHA ** L).mean()), mean_abs_cos_keys=float(np.mean(gram))))
        states.append(maps)
    # open-loop path independence at init: Delta_t = Delta_{t-1} P + C; zero start vs a donor's 8-pass state (derangement), 16 passes
    def run(maps, D0, K):
        Ds = [d.clone() for d in D0]
        for _ in range(K):
            Ds = [Ds[h] @ maps[h][0] + maps[h][1] for h in range(6)]
        return Ds
    zero = [torch.zeros(64, 64, dtype=torch.float64) for _ in range(6)]
    cos16, cos8, inc = [], [], []
    for i in range(len(states)):
        j = (i + 1) % len(states); donor = run(states[j], zero, 8)
        a = torch.cat([d.flatten() for d in run(states[i], zero, 16)]); b = torch.cat([d.flatten() for d in run(states[i], donor, 16)])
        cos16.append(float(F.cosine_similarity(a, b, dim=0)))
        a8 = torch.cat([d.flatten() for d in run(states[i], zero, 8)]); b8 = torch.cat([d.flatten() for d in run(states[i], donor, 8)])
        cos8.append(float(F.cosine_similarity(a8, b8, dim=0)))
        # clean-trajectory increment ratios log(|D_{t+1}-D_t| / |D_t-D_{t-1}|), t = 2,3,4
        Ds = [zero];
        for t in range(6): Ds.append(run(states[i], Ds[-1], 1))
        nr = [float(torch.cat([(Ds[t + 1][h] - Ds[t][h]).flatten() for h in range(6)]).norm()) for t in range(6)]
        inc.append([np.log(nr[t] / nr[t - 1]) for t in (2, 3, 4)])
    agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    agg.update(open_loop_delta_cos16=float(np.mean(cos16)), open_loop_delta_cos8=float(np.mean(cos8)),
               step_ratio_t234=[float(x) for x in np.mean(np.array(inc), 0)], alphaL=float(ALPHA ** agg["L"]))
    out[n] = agg; print(n, json.dumps({k: (round(v, 4) if isinstance(v, float) else [round(x, 3) for x in v]) for k, v in agg.items()})); sys.stdout.flush()

# random-key baseline (i.i.d. uniform unit keys in R^64), same lengths
rb = {}
for L in [5, 9, 14, 19, 29, 39]:
    tops, meds, nal = [], [], []
    for _ in range(50):
        kk = F.normalize(torch.randn(L, 64, dtype=torch.float64), dim=-1); Pm, _ = pass_maps(kk, torch.zeros(L, 64, dtype=torch.float64))
        s = torch.linalg.svdvals(Pm).numpy(); tops.append(s.max()); meds.append(np.median(s)); nal.append((s > ALPHA ** L * (1 - 1e-9)).sum())
    rb[L] = dict(top=float(np.mean(tops)), med=float(np.mean(meds)), n_at_alphaL=float(np.mean(nal)), alphaL=float(ALPHA ** L))
    print("random", L, rb[L])
json.dump({"gpt2_nested": out, "random_keys": rb}, open("init_jacobian.json", "w", newline="\n"), indent=1)
