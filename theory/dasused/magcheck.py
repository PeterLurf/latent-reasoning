"""DAS-USED draft §4 CHECK on the code (boxes/deltamem.py add_read and the delta-rule step, read-only): (M1) an additive head's read is invariant to
rescaling that head's state under add_norm 'rms' (up to the 1e-6 regulariser), invariant to the JOINT rescaling (A, z) under 'den', and linear under None;
(M2) after a rescale c of the additive state, the next write W gives the read of c A + W: the first downstream read is unchanged (rms), the second is not
(the mix of old content and the new write changes); (M4) the delta rule at a shared key: S'k = alpha (1 - beta) S k + beta v, so beta = 1 replaces the old
content (no magnitude layering) and beta < 1 layers it with ratio alpha (1 - beta). -> theory/dasused/magcheck.json"""
import json
import os
import sys
import types

os.environ.setdefault("OMP_NUM_THREADS", "3")
sys.path.insert(0, "D:/Latent")
import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

import boxes.deltamem as DM  # noqa: E402

torch.manual_seed(0); torch.set_default_dtype(torch.float64)
B, H, DH = 4, 6, 64
pq = Fn.elu(torch.randn(B, H, DH)) + 1; A = torch.randn(B, H, DH, DH).abs() * 0.1 + torch.randn(B, H, DH, DH) * 0.05; z = A.abs().sum(-1)   # z > 0 like sum phi(k)
out = {"def": __doc__.split("->")[0].strip()}
for mode in ("rms", "den", None):
    mem = types.SimpleNamespace(add_norm=mode, add_gain=torch.rand(H) + 0.5)
    r0 = DM.add_read(mem, pq, A, z if mode == "den" else None); res = {}
    for c in (0.5, 2.0, 10.0):
        rc = DM.add_read(mem, pq, c * A, (c * z) if mode == "den" else None); res[str(c)] = {"max_rel_change": float(((rc - r0).norm(dim=-1) / r0.norm(dim=-1)).max()),
                                                                                         "ratio_norm": float((rc.norm(dim=-1) / r0.norm(dim=-1)).mean())}
    out["M1_" + str(mode)] = res
mem = types.SimpleNamespace(add_norm="rms", add_gain=torch.ones(H)); W = torch.randn(B, H, DH, DH) * 0.1
for c in (0.5, 2.0):
    r1, r1c = DM.add_read(mem, pq, A), DM.add_read(mem, pq, c * A); r2, r2c = DM.add_read(mem, pq, A + W), DM.add_read(mem, pq, c * A + W)
    out["M2_rms_c%s" % c] = {"first_read_rel_change": float(((r1c - r1).norm(dim=-1) / r1.norm(dim=-1)).max()),
                             "second_read_rel_change_mean": float(((r2c - r2).norm(dim=-1) / r2.norm(dim=-1)).mean())}
S = torch.randn(B, DH, DH); k = Fn.normalize(torch.randn(B, DH), dim=-1); v = torch.randn(B, DH); al = 0.99
for be in (1.0, 0.5):
    Sk = torch.einsum("bvk,bk->bv", S, k); Sn = al * (S - be * Sk[..., :, None] * k[..., None, :]) + be * v[..., :, None] * k[..., None, :]
    pred = al * (1 - be) * Sk + be * v; out["M4_beta%s" % be] = {"max_abs_err_vs_formula": float((torch.einsum("bvk,bk->bv", Sn, k) - pred).abs().max()),
                                                                "old_content_weight": al * (1 - be)}
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "magcheck.json"), "w", newline="\n"), indent=1)
print(json.dumps(out, indent=1))
