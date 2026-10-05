"""§116 note 116-4 addendum (Theory's request): copy-only unit test of Gen110.copy_loss after the no_grad fix.
Gen110 from pretrained GPT-2, masked fork, ONLY copy_loss (bs 64, lr 1e-4, N steps). Copy accuracy = greedy masked-fork
decode of the start state's own text from the slot fed with the start text's last hidden (the copy_loss path).
    python scratch/copy_unit_test.py --steps 800 --device cuda:2
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("BOXES_NB", "5")
from boxes import s5  # noqa: E402
from boxes.gen97 import STTOK, decode_to_el  # noqa: E402
from boxes.self110 import Gen110  # noqa: E402


@torch.no_grad()
def copy_acc(m, dev, n=300, seed=11604):
    cs = np.random.default_rng(seed).integers(0, 120, n); g = m.g; st = torch.tensor(STTOK[cs], device=dev)
    h, past = g(g.embed(st)); s_in = h[:, -1]; hs, past = g(s_in[:, None], past)
    d = m.greedy(hs[:, 0], past).cpu().numpy(); e, _ = decode_to_el(d)
    return float((e == cs).mean()), float((e >= 0).mean())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--steps", type=int, default=800); ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--device", default="cpu"); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(ROOT, "theory", "results", "s116_copy_unit_test.json"))
    a = ap.parse_args(); dev = a.device; torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    m = Gen110(s5.load_gpt2(dev)).to(dev); assert m.mask
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.01); m.train(); curve = []; t0 = time.time()
    acc0 = copy_acc(m.eval(), dev); m.train(); curve.append({"step": 0, "copy_acc": acc0[0], "parse": acc0[1]})
    for step in range(1, a.steps + 1):
        cs = rng.integers(0, 120, a.bs)
        loss = m.copy_loss(torch.tensor(STTOK[cs], device=dev), torch.tensor(cs, device=dev))
        assert loss.requires_grad, "copy_loss has no graph"
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0:
            m.eval(); acc = copy_acc(m, dev); m.train()
            curve.append({"step": step, "loss": float(loss), "copy_acc": acc[0], "parse": acc[1], "sec": round(time.time() - t0)}); print(curve[-1], flush=True)
    json.dump({"steps": a.steps, "bs": a.bs, "lr": a.lr, "device": dev, "curve": curve}, open(a.out, "w", newline="\n"), indent=1)


if __name__ == "__main__":
    main()
