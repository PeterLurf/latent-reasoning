"""sha256 of a file read with FILE_FLAG_NO_BUFFERING (bypasses the Windows page cache: the bytes come from the disk, not from cached RAM pages).
    python nobuf_sha.py <path> [<path> ...]"""
import ctypes
import ctypes.wintypes as W
import hashlib
import os
import sys

GENERIC_READ, OPEN_EXISTING, FILE_SHARE_READ = 0x80000000, 3, 1
FILE_FLAG_NO_BUFFERING, FILE_FLAG_SEQUENTIAL_SCAN = 0x20000000, 0x08000000
k32 = ctypes.WinDLL("kernel32", use_last_error=True)
k32.CreateFileW.restype = W.HANDLE
k32.CreateFileW.argtypes = [W.LPCWSTR, W.DWORD, W.DWORD, ctypes.c_void_p, W.DWORD, W.DWORD, W.HANDLE]
k32.ReadFile.argtypes = [W.HANDLE, ctypes.c_void_p, W.DWORD, ctypes.POINTER(W.DWORD), ctypes.c_void_p]
k32.VirtualAlloc.restype = ctypes.c_void_p
k32.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, W.DWORD, W.DWORD]


def sha_nobuf(path):
    size = os.path.getsize(path); h = k32.CreateFileW(path, GENERIC_READ, FILE_SHARE_READ, None, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN, None)
    if h == W.HANDLE(-1).value or h is None:
        raise OSError(ctypes.get_last_error())
    CH = 1 << 24; buf = k32.VirtualAlloc(None, CH, 0x3000, 0x04); got = W.DWORD(0); sha = hashlib.sha256(); left = size
    try:
        while left > 0:
            if not k32.ReadFile(h, buf, CH, ctypes.byref(got), None):
                raise OSError(ctypes.get_last_error())
            n = min(got.value, left)
            if n == 0:
                break
            sha.update(ctypes.string_at(buf, n)); left -= n
    finally:
        k32.CloseHandle(h)
    return sha.hexdigest()


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(p, sha_nobuf(p)[:16])
