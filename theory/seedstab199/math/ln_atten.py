"""N4: the LN_w Jacobian attenuation on the closed loop at init: ||J_LN(h)|| ~ max|w_ln| / sigma(h), h = the residual after block 11 (pre ln_f)
on nested think inputs; LN_w = block 11's ln_1 (deltamem.py l.81-83). Also LN_6 (block 6 ln_1) on h6 for comparison."""
import json, numpy as np, torch, torch.nn.functional as F
from gpt2lib import *
rng = np.random.default_rng(7); rows = {}
@torch.no_grad()
def resid(tok):
    x = torch.tensor([tok]); h = sd["wte.weight"][x] + sd["wpe.weight"][: len(tok)][None]; h6 = None
    for i in range(12):
        if i == 6: h6 = h.clone()
        h = block(i, h)
    return h6[0], h[0]
for n in [2, 4, 8, 12]:
    s6, s11, j6, j11 = [], [], [], []
    for _ in range(16):
        root, _ = generate(rng, n); tok = [EOT] + [TID[s] for s in render(root)] + [TID[" ."]]
        h6, hw = resid(tok)
        for h, w, S, J in ((h6, sd["h.6.ln_1.weight"], s6, j6), (hw, sd["h.11.ln_1.weight"], s11, j11)):
            sig = h[1:].std(-1, unbiased=False)                              # skip position 0 (the EOT sink, huge norm)
            S.append(float(sig.median())); 
            # exact operator norm of the LN Jacobian at one middle token: diag(w)/sigma (I - 1 1^T/D - y y^T/D)
            t = len(tok) // 2; y = (h[t] - h[t].mean()) / h[t].std(unbiased=False)
            Jm = torch.diag(w) / h[t].std(unbiased=False) @ (torch.eye(D) - torch.ones(D, D) / D - torch.outer(y, y) / D)
            J.append(float(torch.linalg.svdvals(Jm)[0]))
    rows[n] = dict(sigma_h6=float(np.median(s6)), sigma_hw=float(np.median(s11)), Jln6=float(np.median(j6)), Jln_w=float(np.median(j11)))
    print(n, rows[n])
json.dump(rows, open("ln_atten.json", "w", newline="\n"), indent=1)
