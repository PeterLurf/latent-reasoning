"""v2 backlog worker self-test: with LATENT_MEM_CAP_GB set, a 7 GiB allocation on <dev> must raise OOM inside the cap.
Writes <marker> ("ok cap=<GiB>") only then.   python scratch/memcap_selftest.py <dev> <marker>"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from boxes import memcap  # noqa: E402

dev, marker = sys.argv[1], sys.argv[2]
assert memcap.apply(dev) is not None, "LATENT_MEM_CAP_GB not set"
try:
    x = torch.empty(int(6.5 * 2 ** 30), dtype=torch.uint8, device=dev); print("NOT CAPPED: 6.5 GiB allocated", flush=True); sys.exit(1)
except torch.OutOfMemoryError as e:
    print("capped:", str(e)[:160], flush=True); open(marker, "w").write("ok cap=%s\n" % os.environ["LATENT_MEM_CAP_GB"])
