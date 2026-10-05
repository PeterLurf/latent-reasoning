"""THEORY §112 / §110, descriptive (note 112-2): §104's donor protocol, single AND joint patch (slot + op-token KV),
on self-training models (boxes/self110.Gen110, masked fork on). eval.json's donor key has the single patch only; the
joint patch says whether a state that is neither donor- nor recipient-consistent under the single patch is carried
jointly by the slot and the op tokens (multiple mediators, as §92 in 3/5). Eval only; CPU by default (idle priority).
Output theory/results/s112_donor.json (one key per run dir).
    python theory/s112_donor.py --runs s112_f010_s21 s110_s5 s110_s3 --n 500
"""
import argparse
import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "theory"))
from boxes import s5  # noqa: E402
from boxes.self110 import Gen110  # noqa: E402
import s104_serial as T  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", nargs="+", required=True); ap.add_argument("--device", default="cpu")
    ap.add_argument("--n", type=int, default=500); ap.add_argument("--bs", type=int, default=100); ap.add_argument("--threads", type=int, default=3)
    a = ap.parse_args(); torch.set_num_threads(a.threads)
    try:
        import psutil
        psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS if os.name == "nt" else 19)
    except Exception:
        pass
    outp = os.path.join(ROOT, "theory", "results", "s112_donor.json")
    out = json.load(open(outp)) if os.path.exists(outp) else {}
    for run in a.runs:
        p = os.path.join(ROOT, "boxes", "runs_s5", run, "model.pt")
        m = Gen110(s5.load_gpt2(a.device)).to(a.device)
        sd = torch.load(p, map_location=a.device); missing, unexpected = m.load_state_dict(sd, strict=False)
        assert not missing and not unexpected, (missing, unexpected)
        m.eval(); m.mask = True
        out[run] = T.evaluate(T.g97_states, T.g97_inputs, m, a.device, a.n, a.bs, 11200)
        out[run]["n"] = a.n
        print(run, json.dumps(out[run]["intact_per_slot"]), json.dumps(out[run]["patch"]), flush=True)
        json.dump(out, open(outp, "w", newline="\n"), indent=1)
        del m


if __name__ == "__main__":
    main()
