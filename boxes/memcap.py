"""Local hard GPU-memory cap (user 2026-10-06: "only put processes on local cards that wont overfill"; Audit: enforce at runtime,
not by forecast). If LATENT_MEM_CAP_GB is set, this process's PyTorch allocations on `dev` are capped at that many GiB
(torch.cuda.set_per_process_memory_fraction): a job that needs more OOMs inside its own budget and the card cannot go over it.
The CUDA context (~0.3-0.5 GiB) is outside the cap, so the local workers set 6.0 on 8 GB cards (>= 1.5 GiB of the card free).
A no-op when the variable is unset (cloud runs)."""
import os


def apply(dev):
    cap = os.environ.get("LATENT_MEM_CAP_GB")
    if not cap or not str(dev).startswith("cuda"):
        return None
    import torch
    d = torch.device(dev); i = d.index if d.index is not None else torch.cuda.current_device()
    frac = min(1.0, float(cap) * 2 ** 30 / torch.cuda.get_device_properties(i).total_memory)
    torch.cuda.set_per_process_memory_fraction(frac, i)
    print("MEMCAP %s GiB on %s (fraction %.3f)" % (cap, dev, frac), flush=True)
    return frac
