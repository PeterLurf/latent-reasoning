"""THEORY §105 HANDOFF: §104's donor protocol (theory/s104_serial.py) on §105's masked-fork GEN-4 models
(boxes/runs_s5/g105_mask4_s*), with the model's masked fork on (m.mask = True, so greedy decodes see only the slot's K/V).
Output theory/results/s104_serial_s105.json (read by theory/score_s105.py).
    python theory/s105_donor.py --device cuda:0 --n 2000
"""
import argparse
import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "theory"))
from boxes import gen97 as G  # noqa: E402
from boxes import s5  # noqa: E402
import s104_serial as T  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--bs", type=int, default=200); ap.add_argument("--threads", type=int, default=3)
    a = ap.parse_args(); torch.set_num_threads(a.threads); out = {}
    for s in range(1, 6):
        p = os.path.join(ROOT, "boxes", "runs_s5", "g105_mask4_s%d" % s, "model.pt")
        if not os.path.exists(p):
            continue
        m = G.Gen97(s5.load_gpt2(a.device), "gen").to(a.device); m.load_state_dict(torch.load(p, map_location=a.device)); m.eval(); m.mask = True
        key = "g105_mask4_s%d" % s; out[key] = T.evaluate(T.g97_states, T.g97_inputs, m, a.device, a.n, a.bs, 10500 + s)
        print(key, json.dumps(out[key]["patch"]), flush=True)
        del m
        if a.device.startswith("cuda"):
            torch.cuda.empty_cache()
    json.dump(out, open(os.path.join(ROOT, "theory", "results", "s104_serial_s105.json"), "w", newline="\n"), indent=1)


if __name__ == "__main__":
    main()
