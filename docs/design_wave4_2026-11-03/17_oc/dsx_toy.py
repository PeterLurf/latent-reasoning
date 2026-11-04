"""Toy gradient-flow check of DSX (no_grad prefix + detach before each supervised pass), vs DSF (full graph).
Shared-weight additive recurrence S_t = S_{t-1} + tanh(W [S_{t-1}; x_t]); decoder Dec(S, text) = U S + V text (a text path, like our decoder).
Checks: (1) DSX grad on an item of depth d == sum over t in T_d of single-pass grads at the frozen on-policy S_{t-1};
(2) d(L)/d(S_s) for s < d is 0 under DSX, > 0 under DSF; (3) under DSX the gradient a deep item sends to W is the gradient of
'one hop from the given state'; nothing on that item asks pass s < d to change; (4) the decoder's text path (V) gets gradient from
every l_t, so a pass-d write and a decoder-side hop are both routes that lower l_d."""
import torch
torch.manual_seed(0)
D, X, K = 6, 4, 8
W = torch.randn(D, D + X) * 0.3; U = torch.randn(3, D) * 0.3; V = torch.randn(3, X) * 0.3
for p in (W, U, V):
    p.requires_grad_(True)
xs = [torch.randn(X) for _ in range(K)]; txt = torch.randn(X); a = torch.tensor(1)


def F(S, x):
    return S + torch.tanh(W @ torch.cat([S, x]))


def dec(S):
    return U @ S + V @ txt


def run(d, detach):
    S = torch.zeros(D); keep = []
    ctx = torch.no_grad() if detach else torch.enable_grad()
    with ctx:
        for t in range(1, d):
            S = F(S, xs[min(t, d) - 1])
            if not detach:
                S.retain_grad(); keep.append(S)
    L = 0.0; prev = []
    for t in range(d, K + 1):
        Sin = S.detach() if detach else S
        prev.append(Sin.clone().detach())
        S = F(Sin, xs[min(t, d) - 1]); L = L + torch.nn.functional.cross_entropy(dec(S)[None], a[None])
    L = L / (K - d + 1)
    return L, keep, prev


for d in (5, 8):
    for p in (W, U, V):
        p.grad = None
    L, _, prev = run(d, True); L.backward(); gW = W.grad.clone(); gV = V.grad.clone()
    # (1) reference: independent single-pass problems at the same frozen inputs
    for p in (W, U, V):
        p.grad = None
    Lr = 0.0
    for j, t in enumerate(range(d, K + 1)):
        Lr = Lr + torch.nn.functional.cross_entropy(dec(F(prev[j], xs[min(t, d) - 1]))[None], a[None])
    (Lr / (K - d + 1)).backward()
    print(f"d={d} DSX grad == sum of one-pass grads at frozen S_(t-1): {torch.allclose(gW, W.grad, atol=1e-7)}; |gV| (decoder text path) = {gV.norm():.3e}")
    # (2) gradient on early states under DSF
    for p in (W, U, V):
        p.grad = None
    L, keep, _ = run(d, False); L.backward()
    print(f"d={d} DSF |dL/dS_s| s<d:", [f"{k.grad.norm():.2e}" for k in keep], "| DSX: early passes ran under no_grad (no graph) -> 0 by construction")
