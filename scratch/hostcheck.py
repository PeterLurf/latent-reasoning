"""Deterministic GPU host check (Theory note 135-6, 2026-10-06, after s135_base_s5_s2 on gen-b78 logged an impossible label share).
    python scratch/hostcheck.py [--gib 8] [--reps 1000] [--n 1024]
1. memory: fill --gib GiB in 256 MiB int32 blocks with a known pattern (index * 2654435761 + block), re-read and compare, 3 passes;
2. label share: tsd_loss135's dup / share on a FIXED integer label tensor (N = B_micro x 32 = 32 x 32 at --n 1024), --reps times, vs CPU;
3. matmul: a fixed fp32 (2048 x 768) @ (768 x 2048) product --reps / 10 times, every checksum vs the first and vs CPU (tolerance).
Prints one JSON line per test and a final HOSTCHECK PASS / FAIL line."""
import argparse
import json
import time

import torch

ap = argparse.ArgumentParser(); ap.add_argument("--gib", type=float, default=8); ap.add_argument("--reps", type=int, default=1000)
ap.add_argument("--n", type=int, default=1024); ap.add_argument("--device", default="cuda:0")
a = ap.parse_args(); dev = a.device; ok = True; t0 = time.time()
print(json.dumps({"gpu": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu", "torch": torch.__version__}), flush=True)

# 1. memory pattern (2026-10-07: b92's check OOMed on its own int64 temporaries with another process holding 260 MiB -> a false FAIL;
#    the pattern is now built / compared in 8M-element chunks (64 MiB temporaries) and an allocation OOM only ends the fill)
blk = 64 * 2 ** 20; CH = 8 * 2 ** 20                                        # int32 elements per 256 MiB block; chunk for the int64 pattern math


def pattern(b, p, i):
    return torch.arange(i, i + CH, device=dev, dtype=torch.int64).mul_(2654435761).add_(b + 7 * p).remainder_(2 ** 31).to(torch.int32)


nb = int(a.gib * 4); bad_mem = []; got = nb
for p in range(3):
    bufs = []
    for b in range(nb):
        try:
            x = torch.empty(blk, device=dev, dtype=torch.int32)
        except torch.OutOfMemoryError:
            got = min(got, b); break
        for i in range(0, blk, CH):
            x[i:i + CH] = pattern(b, p, i)
        bufs.append(x)
    torch.cuda.synchronize() if dev.startswith("cuda") else None
    for b, x in enumerate(bufs):
        nbad = sum(int((x[i:i + CH] != pattern(b, p, i)).sum()) for i in range(0, blk, CH))
        if nbad:
            bad_mem.append({"pass": p, "block": b, "mismatches": nbad})
    del bufs; torch.cuda.empty_cache()
print(json.dumps({"test": "memory", "gib": got / 4, "gib_requested": nb / 4, "passes": 3, "bad_blocks": bad_mem}), flush=True); ok &= not bad_mem

# 2. label share at the L 32 micro-batch shape
g = torch.Generator().manual_seed(135); lab = torch.randint(0, 120, (a.n,), generator=g)
eye_c = torch.eye(a.n, dtype=torch.bool); dup_c = (lab[:, None] == lab[None, :]) & ~eye_c; ref_sum = int(dup_c.sum())
ref_share = ref_sum / (a.n * (a.n - 1)); L_ = lab.to(dev); eye = torch.eye(a.n, dtype=torch.bool, device=dev); bad_sh = []
for r in range(a.reps):
    s_ = int(((L_[:, None] == L_[None, :]) & ~eye).sum())
    if s_ != ref_sum:
        bad_sh.append({"rep": r, "sum": s_})
print(json.dumps({"test": "label_share", "n": a.n, "reps": a.reps, "cpu_sum": ref_sum, "cpu_share": ref_share, "bad_reps": len(bad_sh), "first_bad": bad_sh[:5]}), flush=True)
ok &= not bad_sh

# 3. matmul checksum
g = torch.Generator().manual_seed(7); A = torch.randn(2048, 768, generator=g); B = torch.randn(768, 2048, generator=g)
cpu = float((A.double() @ B.double()).sum()); Ad, Bd = A.to(dev), B.to(dev); torch.backends.cuda.matmul.allow_tf32 = False
first = None; bad_mm = []; nrep = max(1, a.reps // 10)
for r in range(nrep):
    c = float((Ad @ Bd).double().sum())
    if first is None:
        first = c
    if c != first or abs(c - cpu) > 1e-3 * max(1.0, abs(cpu)) + 1.0:
        bad_mm.append({"rep": r, "sum": c})
print(json.dumps({"test": "matmul", "reps": nrep, "cpu_sum": cpu, "gpu_first": first, "bad_reps": len(bad_mm), "first_bad": bad_mm[:5]}), flush=True)
ok &= not bad_mm
print("HOSTCHECK", "PASS" if ok else "FAIL", "(%.0f s)" % (time.time() - t0), flush=True)
