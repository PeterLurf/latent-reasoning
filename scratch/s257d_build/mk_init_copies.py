"""§257 (DC2_v1 §5 RAM): the dch inits = unbuffered-read copies of dis_s<k>/model.pt. No host-side sha exists for the landed model.pt (the drain has no checksum), so the
read is cross-checked against the independently written dis_s<k>/ckpt_final.pt: both read UNBUFFERED, torch.load'ed from the bytes, every model tensor equal (torch.equal).
Only then is boxes/runs_fsq/dis_s<k>_init/model.pt written from those bytes and re-read unbuffered (sha256 equal) -> scratch/gate_s257d/init_copies.json.
    python -B scratch/s257d_build/mk_init_copies.py"""
import hashlib
import io
import json
import os
import sys

ROOT = "D:/Latent"; os.chdir(ROOT); sys.path.insert(0, ROOT + "/scratch"); sys.path.insert(0, ROOT + "/scratch/ram_scan")
import torch  # noqa: E402
from nobuf_sha import k32, GENERIC_READ, FILE_SHARE_READ, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING, FILE_FLAG_SEQUENTIAL_SCAN, sha_nobuf  # noqa: E402
import ctypes  # noqa: E402
import ctypes.wintypes as W  # noqa: E402


def nobuf_bytes(p):
    size = os.path.getsize(p); h = k32.CreateFileW(os.path.abspath(p), GENERIC_READ, FILE_SHARE_READ, None, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN, None)
    CH = 1 << 24; buf = k32.VirtualAlloc(None, CH, 0x3000, 0x04); got = W.DWORD(0); left = size; out = b""
    try:
        while left > 0:
            assert k32.ReadFile(h, buf, CH, ctypes.byref(got), None); n = min(got.value, left)
            if n == 0:
                break
            out += ctypes.string_at(buf, n); left -= n
    finally:
        k32.CloseHandle(h)
    return out


out = {}
MAJ = {1: "35813f1a41ea8085", 2: "a68cb45e4b4d153d"}                       # the majority of 6 streaming unbuffered reads each (11-01 17:5x; single reads flip on this box)
for k in (1, 2):
    for att in range(8):                                                     # retry until a read both matches the majority sha and equals ckpt_final's model
        mb = nobuf_bytes("boxes/runs_fsq/dis_s%d/model.pt" % k); cb = nobuf_bytes("boxes/runs_fsq/dis_s%d/ckpt_final.pt" % k)
        m = torch.load(io.BytesIO(mb), map_location="cpu", weights_only=True); c = torch.load(io.BytesIO(cb), map_location="cpu", weights_only=False)["model"]
        eq = sorted(m) == sorted(c) and all(torch.equal(m[x], c[x]) for x in m); sha = hashlib.sha256(mb).hexdigest()
        r = {"model_vs_ckpt_final_equal": eq, "n_tensors": len(m), "sha256": sha, "attempts": att + 1}; print(k, att, eq, sha[:16], flush=True)
        if eq and sha.startswith(MAJ[k]):
            break
        del m, c, mb, cb
    eq = eq and sha.startswith(MAJ[k])
    if eq:
        d = "boxes/runs_fsq/dis_s%d_init" % k; os.makedirs(d, exist_ok=True); p = d + "/model.pt"
        with open(p, "wb") as f:
            f.write(mb)
        r["copy_unbuffered_sha256"] = sha_nobuf(os.path.abspath(p)); r["copy_ok"] = r["copy_unbuffered_sha256"] == sha
    out["dis_s%d" % k] = r; del m, c, mb, cb
os.makedirs("scratch/gate_s257d", exist_ok=True); json.dump(out, open("scratch/gate_s257d/init_copies.json", "w", newline="\n"), indent=1)
print({k: (v["model_vs_ckpt_final_equal"], v.get("copy_ok"), v["sha256"][:16]) for k, v in out.items()})
