"""THEORY §142 teacher-target cache: the TC-M teacher's (boxes/runs_mult/m98_tc_4x4_s1) own greedy S_1 .. S_4 for the 4x4 train split,
computed ONCE (deterministic greedy) instead of on every host. The split is built exactly as boxes.multrow builds it (load_bench of the
train file, or load_bench_head for a --train_n smoke, then dedup against the first test_n test items), and keyed by its sha256 together with
the teacher model.pt's sha256; multrow --teacher_cache refuses a cache whose shas do not match its own split / teacher.
    python scratch/teacher_cache142.py <out.npz> [device] [bs] [train_n (smoke)]
-> <out.npz> (dec int8 (N, 4, 8) teacher S_j digits LSB first, -1 = not a digit; ok bool (N, 4) = all 8 digits valid) + <out>.json
(split_sha, teacher_sha, n, teacher_invalid_share, teacher_vs_gold_mismatch_share over valid hand-offs, per-j shares). Resumable: a
.partial.npz is written every 100 batches."""
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("BOXES_NB", "5")
try:
    from boxes import memcap  # noqa: E402
except ImportError:
    memcap = None
import boxes.multrow as MR  # noqa: E402
from boxes.multcol import dedup, load_bench, load_bench_head  # noqa: E402

TEACHER = os.path.join(ROOT, "boxes", "runs_mult", "m98_tc_4x4_s1")


def split_sha(A, B):
    h = hashlib.sha256(); h.update(np.ascontiguousarray(A, np.int8).tobytes()); h.update(np.ascontiguousarray(B, np.int8).tobytes()); return h.hexdigest()


def file_sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def train_split(train_n=0, test_n=1000):
    """multrow's own split: identical calls in the identical order."""
    teA, teB = load_bench(os.path.join(ROOT, "bench", "data", "mult_4x4_test.txt")); teA, teB = teA[:test_n], teB[:test_n]
    p = os.path.join(ROOT, "bench", "data", "mult_train", "4x4_train.txt")
    trA, trB = load_bench_head(p, train_n) if train_n else load_bench(p)
    trA, trB, _ = dedup(trA, trB, teA, teB)
    return trA, trB


def main():
    out = sys.argv[1]; dev = sys.argv[2] if len(sys.argv) > 2 else "cuda:0"; bs = int(sys.argv[3]) if len(sys.argv) > 3 else 512
    train_n = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    memcap and dev.startswith("cuda") and memcap.apply(dev)
    amp = torch.bfloat16 if dev.startswith("cuda") and torch.cuda.get_device_capability(dev)[0] >= 8 else None
    trA, trB = train_split(train_n); N = len(trA); ssha = split_sha(trA, trB); tsha = file_sha(os.path.join(TEACHER, "model.pt"))
    T = MR.Teacher(TEACHER, 0, None, dev)                                   # CE mode: generation only (no probe, no hidden pass)
    dec = np.full((N, 4, MR.NDIG), -1, np.int8); ok = np.zeros((N, 4), bool); i0 = 0; part = out + ".partial.npz"; t0 = time.time()
    if os.path.exists(part):
        z = np.load(part)
        if str(z["split_sha"]) == ssha and str(z["teacher_sha"]) == tsha:
            i0 = int(z["done"]); dec[:i0], ok[:i0] = z["dec"][:i0], z["ok"][:i0]; print("resumed at", i0, flush=True)
    for k, i in enumerate(range(i0, N, bs)):
        _, d_, o_ = T.targets(trA[i:i + bs], trB[i:i + bs], 4, amp)
        dec[i:i + bs] = d_.cpu().numpy().astype(np.int8); ok[i:i + bs] = o_.cpu().numpy()
        if (k + 1) % 100 == 0:
            np.savez(part, dec=dec, ok=ok, done=min(i + bs, N), split_sha=ssha, teacher_sha=tsha)
            print("%d / %d (%.0f s)" % (min(i + bs, N), N, time.time() - t0), flush=True)
    gold = MR.digits8(MR.sj_values(trA, trB)); mism = ok & (dec != gold).any(-1)
    info = {"n": int(N), "train_n": train_n, "split_sha": ssha, "teacher": os.path.relpath(TEACHER, ROOT).replace(os.sep, "/"), "teacher_sha": tsha,
            "teacher_invalid_share": float(1 - ok.mean()), "teacher_vs_gold_mismatch_share": float(mism.sum() / max(ok.sum(), 1)),
            "invalid_by_j": [float(1 - ok[:, j].mean()) for j in range(4)], "mismatch_by_j": [float(mism[:, j].sum() / max(ok[:, j].sum(), 1)) for j in range(4)],
            "sec": round(time.time() - t0), "device": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu", "dtype": str(amp)}
    np.savez_compressed(out, dec=dec, ok=ok, split_sha=ssha, teacher_sha=tsha)
    json.dump(info, open(out + ".json", "w", newline=chr(10)), indent=1)
    if os.path.exists(part):
        os.remove(part)
    print("CACHE", json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
