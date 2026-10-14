"""Exact open-loop (gamma = 0) pass map of the 6 delta heads at INIT: k, v from block 6's ln_1 + c_attn on h6 (blocks 0..5 of pretrained GPT-2),
beta = 1, alpha = sigmoid(5). One pass: S <- S M + B, M = prod_t alpha (I - k_t k_t^T) (token order, right-multiplied), B = the pass's write from 0.
Reports per item and head: spectral radius of M, ||M||_2, and the clean-trajectory step ratio log(||B M^t|| / ||B M^(t-1)||) for t = 1..6
(increment S_{t+1} - S_t = B M^t). Read-only on the repo; CPU, 3 threads."""
import sys, math
import numpy as np, torch, torch.nn.functional as F
torch.set_num_threads(3)
sys.path.insert(0, "D:/Latent")
from boxes import nested170 as N
sd = torch.load("D:/Latent/prosqa/ckpt/gpt2_pretrained.bin", map_location="cpu", weights_only=True)
sd = {k.replace("transformer.", ""): v.float() for k, v in sd.items()}
D, H, hd = 768, 12, 64
def block(i, h):
    p = f"h.{i}."; B_, T, _ = h.shape
    a = F.layer_norm(h, (D,), sd[p+"ln_1.weight"], sd[p+"ln_1.bias"])
    q, k, v = (a @ sd[p+"attn.c_attn.weight"] + sd[p+"attn.c_attn.bias"]).split(D, -1)
    q, k, v = (t.view(B_, T, H, hd).transpose(1, 2) for t in (q, k, v))
    o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    h = h + o.transpose(1, 2).reshape(B_, T, D) @ sd[p+"attn.c_proj.weight"] + sd[p+"attn.c_proj.bias"]
    a = F.layer_norm(h, (D,), sd[p+"ln_2.weight"], sd[p+"ln_2.bias"])
    m = F.gelu(a @ sd[p+"mlp.c_fc.weight"] + sd[p+"mlp.c_fc.bias"], approximate="tanh")
    return h + m @ sd[p+"mlp.c_proj.weight"] + sd[p+"mlp.c_proj.bias"]
alpha = 1 / (1 + math.exp(-5.0))
rng = np.random.default_rng(12345)
rows = []
with torch.no_grad():
    for it in range(240):
        n = int(rng.integers(1, 9)); root, t = N.generate(rng, n); syms = N.render(root)
        ids = [N.EOT] + N.ids(syms) + [N.TID[" ."]]; T = len(ids)
        x = sd["wte.weight"][torch.tensor(ids)][None] + sd["wpe.weight"][:T][None]
        h = x
        for i in range(6):
            h = block(i, h)
        a6 = F.layer_norm(h, (D,), sd["h.6.ln_1.weight"], sd["h.6.ln_1.bias"])
        q, k, v = (a6 @ sd["h.6.attn.c_attn.weight"] + sd["h.6.attn.c_attn.bias"]).split(D, -1)
        k = k.view(T, H, hd)[:, 6:]; v = v.view(T, H, hd)[:, 6:]; q = q.view(T, H, hd)[:, 6:]
        kd = F.normalize(k, dim=-1)
        for hh in range(6):
            M = torch.eye(hd, dtype=torch.float64); Bm = torch.zeros(hd, hd, dtype=torch.float64)
            for tt in range(T):
                kk = kd[tt, hh].double()[:, None]; P = alpha * (torch.eye(hd, dtype=torch.float64) - kk @ kk.T)
                M = M @ P; Bm = Bm @ P + v[tt, hh].double()[:, None] @ kk.T
            ev = torch.linalg.eigvals(M).abs().max().item(); sn = torch.linalg.matrix_norm(M, 2).item()
            inc = [Bm]; 
            for j in range(7):
                inc.append(inc[-1] @ M)
            nr = [torch.linalg.matrix_norm(z).item() for z in inc]
            ratios = [math.log(nr[j+1] / nr[j]) for j in range(6)]
            # read-level (decoder-visible) increments at the last token's query: (S_{t+1}-S_t) q
            qd = F.normalize(q[-1, hh], dim=-1).double()
            rr = [torch.linalg.vector_norm(z @ qd).item() for z in inc]
            rratios = [math.log(rr[j+1] / rr[j]) for j in range(6)]
            rows.append((n, T, hh, ev, sn, ratios, rratios))
import collections
R = np.array([r[5] for r in rows]); RR = np.array([r[6] for r in rows]); EV = np.array([r[3] for r in rows]); SN = np.array([r[4] for r in rows]); TT = np.array([r[1] for r in rows])
print("items", len(rows)//6, "T mean %.1f (min %d max %d); alpha^T mean %.3f; T ln alpha mean %.3f" % (TT.mean(), TT.min(), TT.max(), (alpha**TT).mean(), (TT*math.log(alpha)).mean()))
print("spectral radius rho(M): median %.3f  IQR %.3f-%.3f ; ||M||_2 median %.3f" % (np.median(EV), *np.percentile(EV, [25, 75]), np.median(SN)))
print("state step ratio log(||B M^t||/||B M^(t-1)||), t=1..6, median over item x head:", np.round(np.median(R, 0), 3))
print("   mean:", np.round(R.mean(0), 3))
print("read-level step ratio (last-token query), median:", np.round(np.median(RR, 0), 3))
for hh in range(6):
    sel = [i for i, r in enumerate(rows) if r[2] == hh]
    print(" head %d: rho median %.3f, state ratio t=2..4 median %s" % (6+hh, np.median(EV[sel]), np.round(np.median(R[sel], 0)[1:4], 3)))
byn = collections.defaultdict(list)
for i, r in enumerate(rows):
    byn[r[0]].append(i)
for n_ in sorted(byn):
    s = byn[n_]; print(" n=%d T~%.1f rho med %.3f  state ratio t=2 med %.3f" % (n_, TT[s].mean(), np.median(EV[s]), np.median(R[s, 1])))
