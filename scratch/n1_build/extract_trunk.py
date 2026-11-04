"""§271 N1 (N1_v1 §10 item 4): of_s1's GPT-2 trunk for the DIRECT arms = the 'g.' tensors of boxes/runs_fsq/of_s1_init/model.pt (the streaming copy of of_s1/model.pt),
prefix stripped (the key set equals a direct-arm trunk's, e.g. s233_askname_s1/model.pt: 148 tensors), saved to boxes/runs_fsq/of_s1_trunk/model.pt.
RAM-fault safe (memory latent-box-transient-faults): the source is loaded twice and the two loads must be tensor-equal before saving; the saved file's sha256 is
read back unbuffered 3 times (majority); on a host the same script re-extracts from the shipped, sha-asserted of_s1_init and asserts --expect <sha16> (fail-closed).
    python -B scratch/n1_build/extract_trunk.py [--expect <sha16>]   -> prints TRUNK_SHA <sha256>"""
import argparse
import hashlib
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "boxes", "runs_fsq", "of_s1_init", "model.pt"); DST = os.path.join(ROOT, "boxes", "runs_fsq", "of_s1_trunk", "model.pt")
SRC_SHA = "9453d5ff26015853"                                                # of_s1/model.pt (unbuffered majority sha, 2026-11-04)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--expect", default=""); ap.add_argument("--out", default=None); ap.add_argument("--device", default=None)
    x = ap.parse_args()                                                     # --out / --device: appended by the queue runner (unused)
    assert sha(SRC).startswith(SRC_SHA), "of_s1_init sha"
    a = torch.load(SRC, map_location="cpu", weights_only=True); b = torch.load(SRC, map_location="cpu", weights_only=True)
    assert sorted(a) == sorted(b) and all(torch.equal(a[k], b[k]) for k in a), "two loads differ (RAM fault?)"
    t = {k[2:]: v.clone() for k, v in a.items() if k.startswith("g.")}; assert len(t) == 148, len(t)
    os.makedirs(os.path.dirname(DST), exist_ok=True); torch.save(t, DST)
    s = [sha(DST) for _ in range(3)]; s0 = max(set(s), key=s.count); assert s.count(s0) >= 2, s
    if x.expect:
        assert s0.startswith(x.expect), ("trunk sha", s0, x.expect)
    print("TRUNK_SHA", s0, flush=True)


if __name__ == "__main__":
    main()
