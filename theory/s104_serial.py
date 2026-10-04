"""THEORY §104 (eval only; rules fixed in §104 before this ran on any checkpoint): serial causal use of latent slots.
Models: §92 l92_k4_s1-5 (latent86; 5x5 head reads the slot) and §97 g97_{gen4,hyb4}_s1-3 (gen97; text fork). Within one
4-op chunk; n fresh recipient chunks A and donor chunks B (random starts and ops; donors patched at the same slot index).
PRIMARY (resample/donor patch, Literature 5054f476): at slot j (1..3)
  single = slot j's input <- B's slot-j input;
  joint  = also op j's token KV entries <- B's (the positions that produced slot j's input; multiple mediators).
  Scores at the final slot: donor-consistent  = decode == B's state after j ops composed with A's ops j+1..4 (use);
                            recipient-consistent = decode == A's own true final state (recompute / bypass).
SECONDARY: zero / mean replacement of slot j's input; per-slot decode accuracy vs A's truth.
    python theory/s104_serial.py --device cuda:0 --n 2000
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from boxes import gen97 as G  # noqa: E402
from boxes import latent86 as L  # noqa: E402
from boxes import s5  # noqa: E402

K = 4


def comp(S, O):
    w = S.copy(); out = []
    for t in range(O.shape[1]):
        w = s5.COMP[w, O[:, t]]; out.append(w)
    return np.stack(out, 1)


def splice(pastA, pastB, n_last):
    """Replace the last n_last positions of A's KV cache with B's (all layers)."""
    out = []
    for (ka, va), (kb, vb) in zip(pastA, pastB):
        ka = ka.clone(); va = va.clone()
        ka[:, :, -n_last:] = kb[:, :, -n_last:]; va[:, :, -n_last:] = vb[:, :, -n_last:]
        out.append((ka, va))
    return out


def _apply(patch, j, s_in, sB, pA, pB, nop):
    if patch is None or patch[0] != j:
        return s_in, pA
    if patch[1] in ("single", "joint"):
        s_in = sB.to(s_in.dtype)
        if patch[1] == "joint":
            pA = splice(pA, pB, nop)              # the producers of this slot's input: the op's token positions
        return s_in, pA
    return patch[2].to(s_in.dtype), pA


# ---------------- latent86 (§92) replay: segment j = [slot j (j > 0)] + [op j+1 tokens]
def l86_states(m, S, O, dev, patch=None, B=None):
    g = m.g; ops = torch.tensor(O, device=dev); nop = L.NOP; n = len(S)
    xA = torch.cat([m.start_emb(torch.tensor(S, device=dev))[:, None], g.embed(m._ops(ops[:, 0], dev))], 1); pA = None
    if B is not None:
        opsB = torch.tensor(B[1], device=dev)
        xB = torch.cat([m.start_emb(torch.tensor(B[0], device=dev))[:, None], g.embed(m._ops(opsB[:, 0], dev))], 1); pB = None
    logits = []
    for j in range(K):
        h, pA = g(xA, pA)
        if j > 0:
            logits.append(m.head(h[:, 0]).view(n, 5, 5))
        s_in = h[:, -1]; sB = None
        if B is not None:
            hB, pB = g(xB, pB); sB = hB[:, -1]
        s_in, pA = _apply(patch, j, s_in, sB, pA, pB if B is not None else None, nop)
        xA = torch.cat([s_in[:, None]] + ([g.embed(m._ops(ops[:, j + 1], dev))] if j + 1 < K else []), 1)
        if B is not None:
            xB = torch.cat([sB[:, None]] + ([g.embed(m._ops(opsB[:, j + 1], dev))] if j + 1 < K else []), 1)
    h, pA = g(xA, pA); logits.append(m.head(h[:, 0]).view(n, 5, 5))
    let = torch.stack(logits, 1).argmax(-1).cpu().numpy()
    return L.CODE2EL[(let * (5 ** np.arange(5))).sum(-1)]


def l86_inputs(m, S, O, dev):
    _, ins = m.chunk(torch.tensor(S, device=dev), torch.tensor(O, device=dev))
    return ins


# ---------------- gen97 (§97) replay: segment = [start + op1] or [op j+1 tokens], then the slot
def g97_states(m, S, O, dev, patch=None, B=None):
    g = m.g; ops = torch.tensor(O, device=dev); nop = m.optok.shape[1]; n = len(S)
    xA = torch.cat([m.start_emb(m.start_from(els=S, dev=dev)), g.embed(m.optok[ops[:, 0]])], 1); pA = None
    if B is not None:
        opsB = torch.tensor(B[1], device=dev)
        xB = torch.cat([m.start_emb(m.start_from(els=B[0], dev=dev)), g.embed(m.optok[opsB[:, 0]])], 1); pB = None
    decs = []
    for j in range(K):
        h, pA = g(xA, pA); s_in = h[:, -1]; sB = None
        if B is not None:
            hB, pB = g(xB, pB); sB = hB[:, -1]
        s_in, pA = _apply(patch, j, s_in, sB, pA, pB if B is not None else None, nop)
        hs, pA = g(s_in[:, None], pA); decs.append(m.greedy(hs[:, 0], pA))
        if B is not None:
            _, pB = g(sB[:, None], pB)
        if j + 1 < K:
            xA = g.embed(m.optok[ops[:, j + 1]])
            if B is not None:
                xB = g.embed(m.optok[opsB[:, j + 1]])
    toks = torch.stack(decs, 1).cpu().numpy(); el, _ = G.decode_to_el(toks.reshape(-1, G.NST))
    return el.reshape(n, K)


def g97_inputs(m, S, O, dev):
    _, _, ins = m.chunk(m.start_from(els=S, dev=dev), torch.tensor(O, device=dev), None, None)
    return ins


@torch.no_grad()
def evaluate(states, inputs, m, dev, n, bs, seed):
    rng = np.random.default_rng(seed); S = rng.integers(0, 120, n); O = rng.integers(0, 120, (n, K))
    SB = rng.integers(0, 120, n); OB = rng.integers(0, 120, (n, K)); TA = comp(S, O)
    mean_in = torch.cat([inputs(m, S[i:i + bs], O[i:i + bs], dev).float() for i in range(0, n, bs)]).mean(0)
    el = np.concatenate([states(m, S[i:i + bs], O[i:i + bs], dev) for i in range(0, n, bs)])
    res = {"intact_per_slot": (el == TA).mean(0).tolist(), "patch": {}, "secondary": {}}
    for j in range(K - 1):
        donor_target = comp(SB, OB[:, :j + 1])[:, -1]
        for t in range(j + 1, K):
            donor_target = s5.COMP[donor_target, O[:, t]]
        for mode in ("single", "joint"):
            out = np.concatenate([states(m, S[i:i + bs], O[i:i + bs], dev, patch=(j, mode, None), B=(SB[i:i + bs], OB[i:i + bs]))
                                  for i in range(0, n, bs)])
            res["patch"]["slot%d_%s" % (j + 1, mode)] = {"donor_consistent": float((out[:, -1] == donor_target).mean()),
                                                         "recipient_consistent": float((out[:, -1] == TA[:, -1]).mean())}
        for mode in ("zero", "mean"):
            out = []
            for i in range(0, n, bs):
                b = min(bs, n - i)
                v = torch.zeros(b, mean_in.shape[-1], device=dev) if mode == "zero" else mean_in[j][None].expand(b, -1).to(dev)
                out.append(states(m, S[i:i + bs], O[i:i + bs], dev, patch=(j, mode, v)))
            res["secondary"]["slot%d_%s" % (j + 1, mode)] = (np.concatenate(out) == TA).mean(0).tolist()
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cpu"); ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--n", type=int, default=2000); ap.add_argument("--bs", type=int, default=250)
    a = ap.parse_args(); torch.set_num_threads(a.threads); dev = a.device; out = {}
    for s in range(1, 6):
        m = L.LatentRestart(s5.load_gpt2(dev)).to(dev)
        m.load_state_dict(torch.load(os.path.join(ROOT, "boxes", "runs_s5", "l92_k4_s%d" % s, "model.pt"), map_location=dev)); m.eval()
        key = "l92_k4_s%d" % s; out[key] = evaluate(l86_states, l86_inputs, m, dev, a.n, a.bs, 10400 + s)
        print(key, json.dumps(out[key]["patch"]), flush=True)
    for arm in ("gen", "hyb"):
        for s in range(1, 4):
            m = G.Gen97(s5.load_gpt2(dev), arm).to(dev)
            m.load_state_dict(torch.load(os.path.join(ROOT, "boxes", "runs_s5", "g97_%s4_s%d" % (arm, s), "model.pt"), map_location=dev)); m.eval()
            key = "g97_%s4_s%d" % (arm, s); out[key] = evaluate(g97_states, g97_inputs, m, dev, a.n, a.bs, 10410 + s)
            print(key, json.dumps(out[key]["patch"]), flush=True)
    json.dump(out, open(os.path.join(ROOT, "theory", "results", "s104_serial.json"), "w", newline="\n"), indent=1)


if __name__ == "__main__":
    main()
