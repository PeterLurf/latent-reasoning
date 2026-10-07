"""§136 comparator (descriptive, Experiments 2026-10-07): fsq120.s136_dim_use on an existing FIXED-width S5 model - the variance share
of z_b in dims >= 8, R^2_state from those dims alone, and the effective rank - so §136's grown runs (eval.json s136) can be read
against fixed d 64 / d 16 runs (§133) and d 8 runs (§128 / §134) measured the same way (probe127's fixed chains, eval / hard).
    python scratch/dimuse136.py <run dir> [device] [--fp32]   -> <run dir>/eval_dimuse136.json"""
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("BOXES_NB", "5")
import torch  # noqa: E402

try:                                                                        # local hard memory cap
    from boxes import memcap  # noqa: E402
except ImportError:
    memcap = None
import boxes.fsq120 as F  # noqa: E402
from boxes import s5  # noqa: E402


def main():
    d = sys.argv[1].rstrip("/"); dev = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else "cuda:0"; fp32 = "--fp32" in sys.argv
    a = json.load(open(os.path.join(d, "args.json"))); dq, lev = int(a.get("d_q", 8)), int(a.get("levels", 8))
    F.D_Q, F.LEV = dq, lev; F.HALF = (lev - 1) / 2; F.SHIFT = math.atanh(F.OFF / F.HALF)
    memcap and memcap.apply(dev)
    amp = None if fp32 or not dev.startswith("cuda") or torch.cuda.get_device_capability(dev)[0] < 8 else torch.bfloat16
    fam = F.Family(a.get("family", "s5")); m = F.FSQ120(s5.load_gpt2(dev), fam).to(dev)
    m.no_round = bool(a.get("no_round")); m.soft_beta = float(a.get("soft_beta") or 0.0); m.carrier_ln = bool(a.get("carrier_ln"))
    m.load_state_dict(torch.load(os.path.join(d, "model.pt"), map_location=dev)); m.p_hard = 1.0; m.eval()
    out = F.s136_dim_use(m, fam, dev, amp, 8)
    out.update({"run": os.path.basename(d), "d_q": dq, "no_round": m.no_round, "soft_beta": m.soft_beta, "eval_dtype": str(amp),
                "device": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu", "note": "fixed-width comparator for §136 dim use (dims >= 8 = 'new')"})
    json.dump(out, open(os.path.join(d, "eval_dimuse136.json"), "w", newline="\n"), indent=1); print("DIMUSE", json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
