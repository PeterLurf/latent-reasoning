# §86 debug 2 (non-evidential): head directly on the last op token's output (no slot), k=1, 600 steps.
import os, sys, time
os.environ["BOXES_NB"] = "5"
sys.path.insert(0, ".")
import numpy as np, torch, torch.nn.functional as F
from boxes import s5
from boxes.latent86 import LatentRestart, batch
from boxes import core as bc
dev = sys.argv[1]
X, Y = s5.load_csv("bench/data/s5/S5=128.csv", limit=20000)
torch.manual_seed(0); rng = np.random.default_rng(0)
m = LatentRestart(s5.load_gpt2(dev)).to(dev)
amp = torch.bfloat16 if torch.cuda.get_device_capability(dev)[0] >= 8 else None
opt = torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=0.01); hist = []
for step in range(1, 601):
    st, o, tg = batch(rng, X, Y, 1, 64)
    st, o, tg = torch.tensor(st, device=dev), torch.tensor(o, device=dev), torch.tensor(tg, device=dev)
    with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
        arr = torch.tensor([bc.tok(" " + s5.s_arr(s5.ARR[int(e)])) for e in st.tolist()], device=dev); x = torch.cat([m.g.embed(arr), m.g.embed(m._ops(o[:, 0], dev))], 1)
        h, _ = m.g(x); lg = m.head(h[:, -1])
        loss = F.cross_entropy(lg.float(), tg[:, 0])
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); hist.append(loss.item())
    if step % 100 == 0:
        print("arrtext", step, round(float(np.mean(hist)), 3), flush=True); hist = []
