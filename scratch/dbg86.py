# §86 debug (non-evidential): does the k=1 latent cell learn at all? Variants: as registered; head on standardised slot
# features; head lr 1e-3. 600 steps each, loss every 100.
import os, sys, json, time
os.environ["BOXES_NB"] = "5"
sys.path.insert(0, ".")
import numpy as np, torch, torch.nn.functional as F
from boxes import s5
from boxes.latent86 import LatentRestart, batch
dev = sys.argv[1]; var = sys.argv[2]; k = int(sys.argv[3]) if len(sys.argv) > 3 else 1
X, Y = s5.load_csv("bench/data/s5/S5=128.csv", limit=20000)
torch.manual_seed(0); rng = np.random.default_rng(0)
m = LatentRestart(s5.load_gpt2(dev)).to(dev)
E = m.g.w("wte.weight").detach(); muE, sdE = E.mean(0), E.std(0)
if var == "stdin":                                      # slot input standardised per batch, mapped to the wte cloud
    m.slot_tf = lambda h: muE + sdE * (h - h.mean(0).detach()) / (h.std(0).detach() + 1e-5)
if var == "lnscale":                                    # slot input rescaled to the token-embedding norm
    m.slot_tf = lambda h: h * (E.norm(dim=-1).mean() / (h.norm(dim=-1, keepdim=True) + 1e-6))
amp = torch.bfloat16 if torch.cuda.get_device_capability(dev)[0] >= 8 else None
groups = [{"params": [p for n, p in m.named_parameters() if not n.startswith("head")], "lr": 1e-4},
          {"params": list(m.head.parameters()), "lr": 1e-3 if var == "headlr" else 1e-4}]
opt = torch.optim.AdamW(groups, weight_decay=0.01); hist = []; t0 = time.time()
for step in range(1, 601):
    st, o, tg = batch(rng, X, Y, k, 64)
    with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
        lg, ins = m.chunk(torch.tensor(st, device=dev), torch.tensor(o, device=dev))
        loss = F.cross_entropy(lg.reshape(-1, 120).float(), torch.tensor(tg, device=dev).reshape(-1))
    opt.zero_grad(); loss.backward(); gn = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); hist.append(loss.item())
    if step % 100 == 0:
        print(var, step, round(float(np.mean(hist)), 3), "gradnorm", round(float(gn), 2), "slot-in std", round(float(ins.float().std(0).mean()), 4), round(time.time() - t0), flush=True); hist = []
