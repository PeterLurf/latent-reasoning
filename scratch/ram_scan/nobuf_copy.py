"""11-02 RAM: whole-file unbuffered reads of a 1.4 GB model now corrupt EVERY time (20/20 shas differed; streaming 16 MB-chunk hashes are stable). Streaming
unbuffered copy: read 16 MB chunks (FILE_FLAG_NO_BUFFERING), write each to <dst>.part and hash as you go; keep the copy only if (a) its streaming sha == the wanted sha
and (b) 3 streaming unbuffered re-reads of the copy == it; else retry (<= tries). -> (ok, attempts, sha).
    python -B scratch/ram_scan/nobuf_copy.py <src> <dst> <sha256 prefix>"""
import ctypes
import ctypes.wintypes as W
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nobuf_sha import k32, GENERIC_READ, FILE_SHARE_READ, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING, FILE_FLAG_SEQUENTIAL_SCAN, sha_nobuf  # noqa: E402


def copy_once(src, dst):
    size = os.path.getsize(src); h = k32.CreateFileW(os.path.abspath(src), GENERIC_READ, FILE_SHARE_READ, None, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN, None)
    CH = 1 << 24; buf = k32.VirtualAlloc(None, CH, 0x3000, 0x04); got = W.DWORD(0); left = size; hs = hashlib.sha256()
    try:
        with open(dst + ".part", "wb") as f:
            while left > 0:
                assert k32.ReadFile(h, buf, CH, ctypes.byref(got), None); n = min(got.value, left)
                if n == 0:
                    break
                c = ctypes.string_at(buf, n); hs.update(c); f.write(c); left -= n
    finally:
        k32.CloseHandle(h)
    return hs.hexdigest()


def copy_verified(src, dst, want, tries=10):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    for att in range(1, tries + 1):
        s = copy_once(src, dst)
        if s.startswith(want) and all(sha_nobuf(os.path.abspath(dst + ".part")) == s for _ in range(3)):
            os.replace(dst + ".part", dst); return True, att, s
    return False, tries, s


if __name__ == "__main__":
    ok, att, s = copy_verified(sys.argv[1], sys.argv[2], sys.argv[3]); print("ok" if ok else "FAILED", att, s); sys.exit(0 if ok else 1)
