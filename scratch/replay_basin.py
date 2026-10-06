"""GPU use: python replay_basin.py --seed 1 --steps 2000 --every 50 --device cuda:0 --amp bf16 --out replay_gpu_s1.jsonl
   (--amp bf16 = autocast around the training chain exactly as fsq120.main; default none = fp32. Probes (every K steps) always run in fp32, eval mode, on fixed own-RNG batches.)
claude-13 (Audit research task, 2026-10-05 21:20): CPU fp32 replay of the first N steps of fsq120 FINAL S5, sigma=0 continuous
(--no_round), seeds 1 vs 2, with diagnostics every K steps. Read-only w.r.t. the repo: imports boxes.fsq120, writes only to the
scratchpad. Same init RNG (torch.manual_seed) and data RNG (np rng(seed)) as fsq120.main; micro-batching is inactive at L < 32.
    python replay_basin.py --seed 1 --steps 2000 --every 50 --threads 2 --out replay_s1.jsonl"""
import argparse, json, math, os, sys, time
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "D:/Latent"); os.environ.setdefault("BOXES_NB", "5")
try:                                                                        # local hard memory cap; absent on hosts that lack it -> no cap
    from boxes import memcap  # noqa: E402
except ImportError:
    memcap = None
from boxes import s5
from boxes.fsq120 import FSQ120, Family, make_batch, lmax_at, SHIFT, HALF, OFF, STTOK

ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, required=True); ap.add_argument("--steps", type=int, default=2000)
ap.add_argument("--every", type=int, default=50); ap.add_argument("--threads", type=int, default=2); ap.add_argument("--out", required=True)
ap.add_argument("--device", default="cpu"); ap.add_argument("--amp", choices=["none", "bf16"], default="none", help="bf16: autocast the training step as fsq120.main does on Ampere+ (parity with the 3090 runs); probes stay fp32"); ap.add_argument("--bs", type=int, default=64); ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--total", type=int, default=24000)
a = ap.parse_args(); torch.set_num_threads(a.threads); dev = a.device
memcap and memcap.apply(dev)                                                           # local hard cap (LATENT_MEM_CAP_GB; no-op on the cloud)
torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)                     # as fsq120.main: init RNG then data RNG
fam = Family("s5"); m = FSQ120(s5.load_gpt2(dev), fam).to(dev); m.no_round = True
opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.01)
# fixed probes (own RNG; never touch the data rng)
prng = np.random.default_rng(777)
PS = np.repeat(np.arange(120), 4); PO = prng.integers(0, 120, len(PS)); PN = s5.COMP[PS, PO]       # 480 one-step segments, 4 per start state
QS, QX, QL = make_batch(fam, np.random.default_rng(778), 256, 1)                                  # 256 L=1 chains for decoder-use probes
QS2, QX2, QL2 = make_batch(fam, np.random.default_rng(779), 256, 2)
grp = lambda names: [p for n, p in m.named_parameters() if any(n.startswith(k) for k in names)]
G_DOWN, G_UP, G_LN = grp(["down."]), grp(["up."]), grp(["ln."]); G_GPT = [p for n, p in m.named_parameters() if n.startswith("g.")]
gn = lambda ps: float(math.sqrt(sum(float((p.grad.float() ** 2).sum()) for p in ps if p.grad is not None)))

def eff_rank(Z):
    Zc = Z - Z.mean(0); ev = torch.linalg.eigvalsh(Zc.T @ Zc / max(1, Z.shape[0] - 1)).clamp_min(0); p = ev / ev.sum().clamp_min(1e-12)
    return float(1.0 / (p ** 2).sum().clamp_min(1e-12))

def between_ratio(Z, lab):
    tot = Z.var(0, unbiased=False).mean(); mu = Z.mean(0); cls = torch.unique(lab); b = 0.0
    for c in cls:
        sel = lab == c; b += float(sel.sum()) * ((Z[sel].mean(0) - mu) ** 2).mean()
    return float(b / len(Z) / tot.clamp_min(1e-12))

def nn_loo(Z, lab):
    D = torch.cdist(Z, Z); D.fill_diagonal_(float("inf")); return float((lab[D.argmin(1)] == lab).float().mean())

@torch.no_grad()
def probe():
    m.eval(); out = {}
    h = m.segment(m.start(len(PS), dev, PS), torch.tensor(PO, device=dev)); z = m.down(m.ln(h.float())); zb = torch.tanh(z + SHIFT) * HALF - OFF
    out.update(zb_std_in=float(zb.std(0).mean()), zb_std_in_max=float(zb.std(0).max()), zb_mean_norm=float(zb.mean(0).norm()), z_std_in=float(z.std(0).mean()),
               tanh_prime=float((1 - torch.tanh(z + SHIFT) ** 2).mean()), eff_rank=eff_rank(zb), h_std_in=float(h.float().std(0).mean()),
               R2_next=between_ratio(zb, torch.tensor(PN, device=dev)), R2_start=between_ratio(zb, torch.tensor(PS, device=dev)), R2_op=between_ratio(zb, torch.tensor(PO, device=dev)),
               nn_next=nn_loo(zb, torch.tensor(PN, device=dev)), nn_start=nn_loo(zb, torch.tensor(PS, device=dev)))
    e = m.embed_code(zb); out["e_row_norm"] = float(e.norm(dim=-1).mean()); out["e_std_in"] = float(e.std(0).mean())
    # decoder use on L=1 probe chains: true carrier vs mean-ablated vs shuffled carrier
    for tag, (S0, X, LB) in (("L1", (QS, QX, QL)), ("L2", (QS2, QX2, QL2))):
        loss, e_fin, _, _ = m.chain(m.start(len(S0), dev, S0), torch.tensor(X, device=dev), LB); ans = torch.tensor(fam.answer_tokens(LB[:, -1]), device=dev)
        l_true = float(m.answer_ce(e_fin, ans)); l_mean = float(m.answer_ce(e_fin.mean(0, keepdim=True).expand_as(e_fin), ans))
        perm = torch.randperm(len(S0), generator=torch.Generator().manual_seed(5)).to(dev); l_shuf = float(m.answer_ce(e_fin[perm], ans))
        out.update({f"{tag}_loss": l_true, f"{tag}_dmean": l_mean - l_true, f"{tag}_dshuf": l_shuf - l_true})
    m.train(); return out

out = open(a.out, "w"); hist = []; t0 = time.time()
rec0 = {"step": 0, "config": {"seed": a.seed, "device": dev, "amp": a.amp, "bs": a.bs, "lr": a.lr, "total": a.total, "steps": a.steps, "every": a.every, "torch": torch.__version__, "cuda": torch.version.cuda,
                            "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None, "image_tag": os.environ.get("IMAGE_TAG")}, **probe()}; out.write(json.dumps(rec0) + "\n"); out.flush(); print(json.dumps(rec0), flush=True)
m.train()
for step in range(1, a.steps + 1):
    for gr in opt.param_groups:
        gr["lr"] = a.lr * min(1, step / 100) * 0.5 * (1 + math.cos(math.pi * step / a.total))
    m.p_hard = 0.0 if step <= 2000 else min(1.0, (step - 2000) / 2000); m.train(); m.noise = 0.0
    L = int(rng.integers(1, lmax_at(step) + 1)); s0, x, lab = make_batch(fam, rng, a.bs, L)
    opt.zero_grad()
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(a.amp == "bf16" and dev.startswith("cuda"))):   # as fsq120.main: autocast around the chain only
        loss, _, _, _ = m.chain(m.start(a.bs, dev, s0), torch.tensor(x, device=dev), lab)
    loss.backward()
    g = {"g_down": gn(G_DOWN), "g_up": gn(G_UP), "g_ln": gn(G_LN), "g_gpt": gn(G_GPT)} if step % a.every == 0 else None
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); hist.append(float(loss.detach()))
    if step % a.every == 0:
        rec = {"step": step, "loss": float(np.mean(hist)), "L": L, "sec": round(time.time() - t0), **g, **probe()}; hist = []
        out.write(json.dumps(rec) + "\n"); out.flush(); print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
print("DONE", flush=True)
