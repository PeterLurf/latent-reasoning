"""§211 G1b split (Audit's decision rule 00:2x 10-16): free-running CE at K' = n on 256 items (n 1..8, seed 21108), split n <= 3 / n 4-8, per kept step.
    python scratch/s211_g1b_split.py <run> [<run> ...] --device cuda:1 -> scratch/gate_s211/g1b_split.json"""
import json, sys
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "D:/Latent"); sys.path.insert(0, "D:/Latent/scratch")
import boxes.vbank as VB, boxes.nested170 as N, gate_s211 as G
from boxes import s5
runs = [a for a in sys.argv[1:] if not a.startswith("--") and not a.startswith("cuda")]; dev = sys.argv[sys.argv.index("--device") + 1] if "--device" in sys.argv else "cuda:1"
its = G._train_items(256, 21108, 1, 8); out = {}
for run in runs:
    a = json.load(open("boxes/runs_fsq/%s/args.json" % run)); m = VB.VBankN(s5.load_gpt2(dev), a["mode"]).to(dev); m.eval(); out[run] = {}
    for st in (0, 1000, 2000):
        m.load_state_dict(torch.load("boxes/runs_fsq/%s/ckpt_model_%d.pt" % (run, st), map_location=dev, weights_only=False)["model"]); ce = []
        with torch.no_grad():
            for c0 in range(0, len(its), 64):
                ch = its[c0:c0 + 64]; x, pad, lens = VB.batch_x(ch, dev); y = torch.tensor(N.NUM_IDS[[it["t"] for it in ch]], device=dev)
                r = m.rollout(x, pad, 8, coda_rows={t: torch.arange(len(ch), device=dev) for t in range(1, 9)})
                for i, it in enumerate(ch):
                    ce.append((it["n"], float(F.cross_entropy(m.answer_logits(r["coda"][it["n"]][1][i:i + 1], lens[i:i + 1]), y[i:i + 1]))))
        c = np.array(ce); out[run][str(st)] = {"n<=3": float(c[c[:, 0] <= 3, 1].mean()), "n4-8": float(c[c[:, 0] >= 4, 1].mean()), "all": float(c[:, 1].mean()), "c": float(m.c_anneal)}
        print(run, st, out[run][str(st)], flush=True)
json.dump(out, open("scratch/gate_s211/g1b_split.json", "w", newline="\n"), indent=1)
