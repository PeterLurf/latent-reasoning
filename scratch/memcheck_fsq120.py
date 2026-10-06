"""Memory check for fsq120 FINAL S5 training at L = 32 (the largest shape), one optimizer step per --micro, under the card's own amp
(bf16 on Ampere+, fp32 below). Prints peak allocated / reserved GiB.   python scratch/memcheck_fsq120.py <device> <micro> [bs] [L]
(fsq120 applies --micro only at L >= 32: L 17-31 batches run at micro 1, so check L 31 micro 1 too - 2026-10-06 s17 OOM)"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from boxes import s5  # noqa: E402
from boxes.fsq120 import FSQ120, Family, make_batch  # noqa: E402

dev = sys.argv[1]; nmb = int(sys.argv[2]); bs = int(sys.argv[3]) if len(sys.argv) > 3 else 64; L = int(sys.argv[4]) if len(sys.argv) > 4 else 32
amp = torch.bfloat16 if torch.cuda.get_device_capability(dev)[0] >= 8 else None
fam = Family("s5"); m = FSQ120(s5.load_gpt2(dev), fam).to(dev); m.p_hard = 1.0
opt = torch.optim.AdamW(m.parameters(), lr=1e-4); rng = np.random.default_rng(0)
for _ in range(2):
    s0, x, lab = make_batch(fam, rng, bs, L); opt.zero_grad()
    for mi in range(nmb):
        sl = slice(mi * bs // nmb, (mi + 1) * bs // nmb)
        with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
            loss, _, _, _ = m.chain(m.start(len(x[sl]), dev, s0[sl]), torch.tensor(x[sl], device=dev), lab[sl], None)
        (loss / nmb).backward()
    opt.step()
G = 2 ** 30
print("MEMCHECK", dev, torch.cuda.get_device_name(dev), "amp", amp, "micro", nmb, "bs", bs, "L", L, "peak_alloc %.2f GiB reserved %.2f GiB total %.2f GiB"
      % (torch.cuda.max_memory_allocated(dev) / G, torch.cuda.max_memory_reserved(dev) / G, torch.cuda.get_device_properties(dev).total_memory / G))
