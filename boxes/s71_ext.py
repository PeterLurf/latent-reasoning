"""THEORY §71 extended test (descriptive): a trained WC0_k run's model.pt on the 500 length-2,048 sequences
(S5=2048.csv, seed 7103), same chunked inference and settings as the panel. Writes ext2048.json and preds_ext2048.npy.
    python -m boxes.s71_ext boxes/runs_s5/<run> [--device cuda:0]"""
import os
os.environ["BOXES_NB"] = "5"
import argparse  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import boxes.s5 as s5  # noqa: E402
from boxes.train import load_model  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--device", default="cuda:0")
a = ap.parse_args()
args = json.load(open(os.path.join(a.run, "args.json")))
X, Y = s5.load_csv(os.path.join(s5.ROOT, "bench", "data", "s5", "S5=2048.csv")); s5.check_labels(X, Y)
m = load_model(os.path.join(a.run, "model.pt"), a.device).eval()
amp = torch.bfloat16 if torch.cuda.get_device_capability(a.device)[0] >= 8 else None
if args["arm"] == "p3":                                              # final-only: the final state at 2,048
    fin = s5.p3_rollout(m, X, args["k"], a.device, amp, args.get("eval_bs", 500), args.get("latent", 0))
    pred = np.full_like(Y, -1); pred[:, -1] = fin; comp = None
else:
    pred, comp = s5.wc_rollout(m, X, args["k"], a.device, amp, args.get("eval_bs", 500))
np.save(os.path.join(a.run, "preds_ext2048.npy"), pred)
cum = s5.cumulative(pred, Y)
res = {"n": len(X), "L": X.shape[1], "cumulative": {str(t): float(cum[t - 1]) for t in (512, 1024, 2048)},
       "final_state_acc_2048": float((pred[:, -1] == Y[:, -1]).mean()), "compute_2048_per_problem": comp, "dtype": str(amp)}
json.dump(res, open(os.path.join(a.run, "ext2048.json"), "w", newline="\n"), indent=1); print(json.dumps(res))
