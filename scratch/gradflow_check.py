"""Audit 2026-10-05 (after the §116 copy_loss no_grad bug): gradient-flow assertion per loss term, before any run of a
code path. For each term a run uses, backward on that term ALONE and assert a nonzero grad norm on the trainable
parameters; norms are logged per term. Terms computed inside one chunk are isolated by differencing the loss with the
term on and off on identical inputs (the main part cancels exactly). Noise terms (dither) are checked as a change in the
loss under fixed inputs plus a nonzero gradient through the noisy carry.
    python scratch/gradflow_check.py --paths accel99 mk108 fsq120 self110_copy --device cpu
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("BOXES_NB", "5")
from boxes import s5  # noqa: E402


def gnorm(m):
    return float(torch.sqrt(sum((p.grad.float() ** 2).sum() for p in m.parameters() if p.grad is not None)))


def term(m, f):
    m.zero_grad(set_to_none=True); loss = f(); assert loss.requires_grad, "term has no graph"; loss.backward(); n = gnorm(m)
    return n


def check_accel99(dev, out):
    from boxes.accel99 import MultGen99, latent_batch, textmix_batch
    from boxes.mult98 import load
    D = load(os.path.join(ROOT, "bench", "data", "mult_train", "4x4_train.txt"), limit=64); g = s5.load_gpt2(dev)
    m = MultGen99(g, "genm", std=False).to(dev); m.markov = True; m.train()
    rng = np.random.default_rng(0); pre, st_, ops, tg, ans = latent_batch(D, rng, 4, dev, 4)
    def L(reader, anchor):
        torch.manual_seed(0); m.reader, m.anchor = reader, anchor; return m.chunk(pre, st_, ops, step_tgt=tg, ans_tgt=ans)[0]
    out["accel99.main"] = term(m, lambda: L(False, False))
    out["accel99.reader"] = term(m, lambda: L(True, False) - L(False, False))
    out["accel99.anchor"] = term(m, lambda: L(False, True) - L(False, False))
    tb = textmix_batch(D, rng, 4, dev, 2); m.reader = m.anchor = False
    out["accel99.textmix"] = term(m, lambda: m.chunk(*tb[:3], step_tgt=tb[3], ans_tgt=tb[4])[0])


def check_mk108(dev, out):
    from boxes.gen97 import true_chain
    from boxes.mk108 import MK108
    m = MK108(s5.load_gpt2(dev)).to(dev); m.train(); rng = np.random.default_rng(0)
    S = rng.integers(0, 120, 4); O = rng.integers(0, 120, (4, 4)); tc = true_chain(S, O)
    st = m.start_from(els=S, dev=dev); o = torch.tensor(O, device=dev); t = torch.tensor(tc, device=dev)
    def L(reader, anchor):
        torch.manual_seed(0); m.reader, m.anchor = reader, anchor; return m.chunk(st, o, t, decode=None)[0]
    out["mk108.main"] = term(m, lambda: L(False, False))
    out["mk108.reader"] = term(m, lambda: L(True, False) - L(False, False))
    out["mk108.anchor"] = term(m, lambda: L(False, True) - L(False, False))


def check_fsq120(dev, out):
    from boxes.fsq120 import FSQ120, Family, make_batch
    for fam_name in ("s5", "ff"):
        fam = Family(fam_name); m = FSQ120(s5.load_gpt2(dev), fam).to(dev); m.train(); rng = np.random.default_rng(0)
        s0, x, lab = make_batch(fam, rng, 4, 4); X = torch.tensor(x, device=dev)
        st = lambda: m.start(4, dev, s0 if fam_name == "s5" else None)  # noqa: E731
        for mode in ("fsq", "no_round"):
            m.no_round = mode == "no_round"; m.p_hard = 1.0
            def L(noise):
                m.noise = noise; torch.manual_seed(0); return m.chain(st(), X, lab)[0]
            out[f"fsq120.{fam_name}.{mode}.main"] = term(m, lambda: L(0.0))
            out[f"fsq120.{fam_name}.{mode}.dither_grad"] = term(m, lambda: L(0.25))
            with torch.no_grad():
                out[f"fsq120.{fam_name}.{mode}.dither_loss_delta"] = abs(float(L(0.25)) - float(L(0.0)))
            gd = m.down.weight.grad; out[f"fsq120.{fam_name}.{mode}.down_grad"] = float(gd.norm()) if gd is not None else 0.0
        m.noise = 0.0


def check_self110_copy(dev, out):
    from boxes.gen97 import STTOK
    from boxes.self110 import Gen110
    m = Gen110(s5.load_gpt2(dev)).to(dev); m.train(); cs = np.random.default_rng(0).integers(0, 120, 4)
    out["self110.copy"] = term(m, lambda: m.copy_loss(torch.tensor(STTOK[cs], device=dev), torch.tensor(cs, device=dev)))
    out["self110.expose"] = term(m, lambda: m.expose_loss(torch.tensor(STTOK[cs], device=dev)))   # §124 TEXT-EXPOSE


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--paths", nargs="*", default=["accel99", "mk108", "fsq120", "self110_copy"])
    ap.add_argument("--device", default="cpu"); ap.add_argument("--out", default=os.path.join(ROOT, "theory", "results", "gradflow_check.json"))
    a = ap.parse_args(); out = {}
    for p in a.paths:
        globals()["check_" + p](a.device, out); print({k: v for k, v in out.items() if k.startswith(p.split("_")[0])}, flush=True)
    bad = {k: v for k, v in out.items() if not (v > 0)}
    out["_status"] = "PASS" if not bad else "FAIL"; out["_zero_terms"] = list(bad)
    json.dump(out, open(a.out, "w", newline="\n"), indent=1); print(out["_status"], bad, flush=True)
    sys.exit(0 if not bad else 1)


if __name__ == "__main__":
    main()
