"""Short stability check for one GPU: repeated bf16/fp32 matmuls vs a fixed reference result."""
import sys
import time

import torch

dev = sys.argv[1]
secs = float(sys.argv[2]) if len(sys.argv) > 2 else 90
torch.manual_seed(0)
a = torch.randn(4096, 4096, device=dev)
b = torch.randn(4096, 4096, device=dev)
ref32 = (a @ b).double().sum().item()
ref16 = (a.bfloat16() @ b.bfloat16()).double().sum().item()
t0, n, bad = time.time(), 0, 0
while time.time() - t0 < secs:
    for _ in range(20):
        c = a @ b
        d = a.bfloat16() @ b.bfloat16()
    s32, s16 = c.double().sum().item(), d.double().sum().item()
    n += 1
    if s32 != ref32 or s16 != ref16:
        bad += 1
        print("MISMATCH", n, s32 - ref32, s16 - ref16, flush=True)
print(f"{dev}: {n} checks, {bad} mismatches, {time.time() - t0:.0f}s", flush=True)
