"""Fill most of a GPU's free VRAM with patterns, read back, compare (bad-VRAM screen)."""
import sys
import time

import torch

dev = sys.argv[1]
rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 4
free, total = torch.cuda.mem_get_info(dev)
chunk = 64 * 2**20 // 4                      # 64 MiB of int32
bufs = []
while torch.cuda.mem_get_info(dev)[0] > 400 * 2**20:
    bufs.append(torch.empty(chunk, dtype=torch.int32, device=dev))
print(f"{dev}: allocated {len(bufs) * 64} MiB of {total / 2**20:.0f} MiB", flush=True)
base = torch.arange(chunk, dtype=torch.int32, device=dev)
t0, bad = time.time(), 0
for r in range(rounds):
    for pat in (0x55555555, -0x55555556, 0):          # 0101.., 1010.., address-xor
        for i, x in enumerate(bufs):
            x.copy_(base ^ (pat if pat else i * 2654435761 % 2**31))
        torch.cuda.synchronize(dev)
        for i, x in enumerate(bufs):
            exp = base ^ (pat if pat else i * 2654435761 % 2**31)
            nbad = int((x != exp).sum())
            if nbad:
                bad += nbad
                print(f"round {r} pattern {pat:#x} chunk {i} ({i * 64} MiB): {nbad} bad words", flush=True)
print(f"{dev}: {rounds} rounds, {bad} bad words, {time.time() - t0:.0f}s", flush=True)
