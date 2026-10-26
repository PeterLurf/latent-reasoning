"""THEORY §234 ANSWER-WRITE (AW_v1.md + AW_draft §2.3 / §2.4; Notes 234-1 / 234-2): the write source = the ':' of " The answer is:" appended to each
§233 LAT-ST pass input; only that last token writes the INJECT state (gamma = 1 from step 1 via --write_mix_start 0 --write_mix_end 0); init = the
s233_askname_s1 trunk. A NEW module: inject_st / inject_st_eval / inject_ow / inject_ow_eval / nested170 / deltamem stay byte-identical. main() below is
inject_st.main() copied with listed edits (scratch/s234_build/mk_inject_aw.py, each asserted to apply once), so `--write_src all --init_trunk ''` is
inject_st exactly (gate G1). Eval: boxes/inject_aw_eval.py (inject_st_eval's parts on the AW inputs + swap_k / swap_1 / probe / ws).
  --write_src answer  pass t's input = [EOT] + enc(" " + text + " " + QUESTION(u, S_min(t, d))) + enc(" The answer is:"); InjectAW.write runs the trunk
                      over every position (the read at block 6, blocks 6-11) and write_seq on the gathered ':' rows only (aw_checks write_at; equal to
                      the one-hot-masked full write in fp64, gate G2). Decoder and aux unchanged (§228 / §233). No think-pass token is decoded or fed.
  --init_trunk P      GPT-2 loaded strict from P before DeltaMem is built (DeltaMem copies block 6's ln_1 / c_attn and block 11's ln_1 from the LOADED
                      trunk; g_add calibrated on it); --init_sha the expected sha256 prefix (asserted). CD-AN = --arm direct --cue all --init_trunk P.
  WS-READOUT          every 1k-bin record (LAT-AW): the trunk's argmax over the 11 number tokens at ':' of pass t vs v_min(t, d) on the first 200 TEST2
                      items of d 1..d_max (Note 234-5 c; held out by construction; no EXCL change, so the training stream is st_plst2_s1's), no_grad, never fed back.
    python -m boxes.inject_aw --arm inject --cue step --write_src answer --init_trunk boxes/runs_fsq/s233_askname_s1/model.pt --init_sha c00f7ec0bd7977fe
        --think_k 8 --aux_step 1 --aux_fade 48000,48000 --aux_reweight --fast_add --cache_lower --write_mix_start 0 --write_mix_end 0 --d_max 2 --steps 12000 --out ..."""
import os
os.environ.setdefault("BOXES_NB", "5")
import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import boxes.deltamem as DM  # noqa: E402
import boxes.nested170 as N7  # noqa: E402
import boxes.ow_words as OW  # noqa: E402
from boxes.nested170 import AUX_W, EOT, NUM_IDS, pad  # noqa: E402
from boxes.inject_ow import (ENC, SFX, SEED_D, SEED_CAL, SEED_MEMCHECK, OWKW, META, tok_ow, tok_nested, direct_logits, sha_file, build_sets,  # noqa: E402, F401
                             InjectOW, load_trunk)
from boxes.inject_st import CUE_FMT, cue_str, tok_st, cue_inputs, st_len  # noqa: E402, F401

COLON = SFX[-1]
AW_FMT = {"all": "", "answer": " | §234 AW: + ' The answer is:' on every pass input; only the ':' state writes (gathered single-token write)"}


class InjectAW(InjectOW):
    """InjectOW whose think write comes from the last real token only (':'). pass_aw also returns that token's logits (WS-READOUT; eval only)."""
    def pass_aw(self, S, x, mask, h_low=None):
        g, mem = self.g, self.mem; B, T = x.shape; D = g.w("wte.weight").shape[1]; hd = D // g.H; assert mem.write_mode == "top", "AW: h_w = block 11's output"
        h = (g.embed(x) + DM.wpe_slice(g, T)[None]) if h_low is None else h_low; h6 = None
        for i in range(0 if h_low is None else mem.layer, g.L):
            if i == mem.layer:
                h6 = h; h = DM.read_fixed(mem, h, S)
            h = DM._block(g, i, h, hd)
        pos = (mask.long().sum(1) - 1) if mask is not None else torch.full((B,), T - 1, device=x.device); ar = torch.arange(B, device=x.device)
        assert bool((x[ar, pos] == COLON).all()), "AW: the write position must be ':'"
        return DM.write_seq(mem, h6[ar, pos][:, None], h[ar, pos][:, None], S, None), h, pos

    def write(self, S, x, mask, h_low=None):
        return self.pass_aw(S, x, mask, h_low)[0]

    def colon_logits(self, h, pos):
        D = h.shape[-1]; hf = Fn.layer_norm(h, (D,), self.g.w("ln_f.weight"), self.g.w("ln_f.bias"))
        return self.g.logits(hf[torch.arange(h.shape[0], device=h.device), pos]).float()


def tok_aw(it, aux_target="value", cue="none", write_src="all"):
    """tok_st, then (write_src answer) " The answer is:" appended to every pass input; the decoder / aux untouched."""
    o = tok_st(it, aux_target, cue)
    if write_src == "answer":
        assert cue == "step" and "think_cues" in o, "AW: --write_src answer needs --cue step"
        o["think_cues"] = [q + SFX for q in o["think_cues"]]
        assert all(q[-1] == COLON for q in o["think_cues"]) and o["dec"] == o["think"] + SFX, "AW: inputs / decoder"
    return o


def build_aw(a, dev):
    """inject_ow.build_model with --init_trunk: GPT-2 loaded strict (sha asserted) BEFORE DeltaMem is built; InjectAW for --write_src answer."""
    g = load_trunk(dev, {"scratch": ""})
    if getattr(a, "init_trunk", ""):
        sha = hashlib.sha256(open(a.init_trunk, "rb").read()).hexdigest(); assert not a.init_sha or sha.startswith(a.init_sha), ("init sha", sha, a.init_sha)
        g.load_state_dict(torch.load(a.init_trunk, map_location=dev, weights_only=True), strict=True); a.init_sha256 = sha
    if a.arm == "direct":
        return None, g, g
    cls = InjectAW if getattr(a, "write_src", "all") == "answer" else InjectOW
    m = cls(g, 6, a.mem_write, 6, 6, None).to(dev); m.nopass = False
    assert torch.equal(m.mem.qkv.weight.detach(), g.w("h.6.attn.c_attn.weight").detach().float().T), "G12: DeltaMem qkv = the loaded block-6 c_attn"
    return m, g, m


@torch.no_grad()
def ws_readout(m, items_by_d, dev, K, bs=50, per_item=False):
    """WS-READOUT: per d, per pass t, the trunk's argmax over the 11 number tokens at ':' vs v_min(t, d); never fed back. per_item: also
    {d: {t: [argmax value per item]}} (the eval part)."""
    out = {}; preds = {}; num = torch.tensor(NUM_IDS, device=dev); was = m.training; m.eval()
    try:
        for d, its in items_by_d.items():
            ok = {t: [] for t in range(1, K + 1)}; pv = {t: [] for t in range(1, K + 1)}
            for i in range(0, len(its), bs):
                ch = its[i:i + bs]; xs, msk = cue_inputs(m, ch, dev); hls = [m.lower(x_) for x_ in xs]; S = m.mem.zero_state(len(ch), dev)
                for t in range(1, K + 1):
                    j = min(t, len(xs)) - 1; S, h, pos = m.pass_aw(S, xs[j], msk, hls[j])
                    y = torch.tensor([int(it["raw"]["values"][min(t, it["d"])]) for it in ch], device=dev)
                    p_ = m.colon_logits(h, pos)[:, num].argmax(-1); ok[t].append((p_ == y).float()); pv[t].append(p_)
            out[str(d)] = {str(t): float(torch.cat(v).mean()) for t, v in ok.items()}; preds[str(d)] = {str(t): torch.cat(v).tolist() for t, v in pv.items()}
    finally:
        m.train(was)
    return (out, preds) if per_item else out
