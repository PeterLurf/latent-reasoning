# memory check (non-evidential): fade_std, N = 8, micro-batch 16, forward + backward on cuda:0
import sys, random, torch; sys.path.insert(0, ".")
from boxes.fade89 import Fade
from boxes.train import load_model
from boxes.core import make_problem
dev = sys.argv[1]; mbs = int(sys.argv[2]); arm = sys.argv[3]
f = Fade(load_model("prosqa/ckpt/gpt2_pretrained.bin", dev), arm).to(dev)
amp = torch.bfloat16 if torch.cuda.get_device_capability(dev)[0] >= 8 else None
rng = random.Random(0); b = [make_problem(rng, 8, "all") for _ in range(mbs)]
torch.cuda.reset_peak_memory_stats(dev)
with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
    out = f(b, dev, 0.5, stage=4, train=True); loss = out["answer"] + out.get("thought", 0) + out.get("aux", 0)
loss.backward(); print(arm, "mbs", mbs, "peak GB", round(torch.cuda.max_memory_allocated(dev) / 2**30, 2))
