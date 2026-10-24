"""THEORY §233 STEP-TOLD (ST_v1.md at da924dca + REVIEW_ST fixes; Notes 233-1 / 233-2): LAT-ST and CUED-DIRECT on the TOPO word programs. A NEW module:
boxes/inject_ow.py / inject_ow_eval.py / nested170.py / deltamem.py stay byte-identical (s228_pdir2c_s1 stays valid; no lazy-import skew). Shared code
(build_sets, tok_ow, InjectOW, build_model, direct_logits, ...) is imported from inject_ow; main() below is inject_ow.main() copied with listed edits
(scratchpad mk_inject_st.py, each edit asserted to apply once), so `--cue none` is inject_ow bitwise (gate G1).
  --cue step  (LAT-ST, --arm inject): pass t's think input = [EOT] + enc(" " + text + " " + QUESTION(u, S_min(t, d))) (7 extra tokens; passes t > d hold
              the S_d cue); the decoder ([EOT] + text + " The answer is:") and the aux are §228's. --cache_lower caches blocks 0-5 per DISTINCT input
              (d per step; exact: identical input -> identical rows, gates G2 / G2b).
  --cue all   (CUED-DIRECT, --arm direct): dec = [EOT] + enc(" " + text + sum_{t=1..d} " " + QUESTION(u, S_t)) + " The answer is:" (st_cd_local's form).
  --cue none  inject_ow exactly.
Edits vs inject_ow.main: the --cue flag + asserts; tok_st for items / calibration / mem_check; the loss's per-pass inputs xs[min(t, len(xs)) - 1] with one
lower cache per xs entry; calibration's think as an explicit write loop (InjectN.think's sequence); the 1k-bin keys aux_vacc_by_d_pass (+ _n) when the
cue is on (G10: they pool back to aux_vacc_by_pass); the memory check's eval probe = TEST2's own longest cued d 12 item (the CD-local eval OOM); the
eval module boxes/inject_st_eval.py; args gain inject_ow_sha256 and the cue format.
G-CUE build asserts (tok_st): every cue has no digit character and no number token, every cue is exactly 7 tokens.
    python -m boxes.inject_st --arm inject --cue step --think_k 8 --aux_step 1 --aux_fade 48000,48000 --aux_reweight --fast_add --cache_lower --d_max 2 --steps 12000 --out ...
    python -m boxes.inject_st --arm direct --cue all --d_max 2 --steps 12000 --out ..."""
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
import boxes.nested170 as N7  # noqa: E402
import boxes.ow_words as OW  # noqa: E402
from boxes.nested170 import AUX_W, EOT, NUM_IDS, pad  # noqa: E402
from boxes.inject_ow import (ENC, SFX, SEED_D, SEED_CAL, SEED_MEMCHECK, OWKW, META, tok_ow, tok_nested, direct_logits, sha_file, build_sets,  # noqa: E402, F401
                             build_model)

CUE_LEN = 7
CUE_FMT = {"none": "", "step": " | §233 LAT-ST: think_t = [EOT] + enc(' ' + text + ' ' + QUESTION(u, S_min(t, d))); lower cache per distinct input",
           "all": " | §233 CUED-DIRECT: dec = [EOT] + enc(' ' + text + sum_t ' ' + QUESTION(u, S_t), t = 1..d) + ' The answer is:'"}


def cue_str(it, c):
    """the cue for S_c: the question template with S_c's name (a function of the skeleton only: names, unit, c)."""
    return OW.QUESTION.format(u=it["unit"], A=it["names"]["spine"][c])


def tok_st(it, aux_target="value", cue="none"):
    """§228's tok_ow, then (cue step) think_cues = the pass inputs for c = 1..d, or (cue all) the CUED-DIRECT think / dec. G-CUE asserts."""
    o = tok_ow(it, aux_target)
    if cue == "none":
        return o
    d = it["d"]; text = it["text"]; nums = set(int(v) for v in NUM_IDS)
    for c in range(1, d + 1):
        s = cue_str(it, c); tk = ENC.encode(" " + s)
        assert not any(ch.isdigit() for ch in s) and not (set(tk) & nums) and len(tk) == CUE_LEN, ("G-CUE", s, tk)
    if cue == "step":
        o["think_cues"] = [[EOT] + ENC.encode(" " + text + " " + cue_str(it, c)) for c in range(1, d + 1)]
        assert all(len(q) == len(o["think"]) + CUE_LEN and q[:len(o["think"])] == o["think"] for q in o["think_cues"]), "G-CUE: the text rows precede the cue"
    elif cue == "all":
        th = [EOT] + ENC.encode(" " + text + "".join(" " + cue_str(it, c) for c in range(1, d + 1))); o["think"] = th; o["dec"] = th + SFX
    return o


def st_len(it):
    return max([len(it["dec"])] + [len(q) for q in it.get("think_cues", [])])


def cue_inputs(m, items, dev):
    """-> ([x_1 .. x_Td], mask): x_t = the batch's pass-t think input (item i: its cue min(t, d_i)); Td = max d_i; cue none (no think_cues): [the plain
    think input] (inject_ow's think_inputs, one tensor). Every x_t has the same padding (each item's cued length is fixed): asserted."""
    if "think_cues" not in items[0]:
        x, msk = m.think_inputs(items, dev); return [x], msk
    Td = max(len(it["think_cues"]) for it in items); xs = []; m0 = None
    for t in range(1, Td + 1):
        ids_, mk = pad([it["think_cues"][min(t, len(it["think_cues"])) - 1] for it in items])
        assert m0 is None or (mk == m0).all(), "cue_inputs: the pass inputs must share one padding"
        m0 = mk; xs.append(torch.tensor(ids_, device=dev))
    return xs, torch.tensor(m0, device=dev)
