"""§250 Note 250-3 $0 PROBE (pre-stated; eval-only): does pass 1 read v_1 under the held-out frames? -> scratch/s250_s1probe/eval_s1probe.json {model: {frame: {acc, n}}}.
Input (one string for both models): [EOT] + enc(' ' + text + ' ' + QUESTION(u, S_1)) + enc(' The answer is:') = the askname P1-ASK+NAME dec (ask_meta.json, X = S_1) = the
AW pass-1 input (think_cues[0]); target v_1 (the item's values[1]); the argmax over the 11 number tokens at ':'.
  (a) s233_askname_s1: the direct trunk (as it is evaluated: IO.direct_logits on that dec, fp32);  (b) lh_fh_s1: pass 1 only from S = 0 (pass_aw's ':' logits, colon_logits).
Items (d 2, DF law): '0-5' = TEST2-DF d 2 (every op in frames 0-5; there is no per-frame set for 0-5), 1,000; '6' / '7' = the eval-only sets of part_frames (Stream(25000 + f),
frames (f,)), 1,000 each; '12'..'15' = build_sets(2)['framesets'][f][2], 1,000 each. Path check: (b)'s preds on TEST2-DF d 2 == lh_fh_s1's eval_ws pass-1 preds (host).
    python -B scratch/s250_s1probe.py [--device cuda:0 --bs 250]"""
import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = "D:/Latent"; sys.path.insert(0, ROOT); os.chdir(ROOT)
from boxes import inject_aw_eval as E  # noqa: E402
from boxes import inject_ow as IO  # noqa: E402
import boxes.inject_st as ST  # noqa: E402
import boxes.ow_words as OW  # noqa: E402
from boxes.nested170 import EOT, NUM_IDS, pad  # noqa: E402


def probe_input(raw):
    return [EOT] + IO.ENC.encode(" " + raw["text"] + " " + ST.cue_str(raw, 1)) + IO.SFX


@torch.no_grad()
def run_direct(g, raws, dev, bs):
    out = []
    for i in range(0, len(raws), bs):
        out += E.pred_of(IO.direct_logits(g, [{"dec": probe_input(r)} for r in raws[i:i + bs]], dev, E.NULLC)).tolist()
    return np.array(out)


@torch.no_grad()
def run_pass1(m, raws, dev, bs):
    num = torch.tensor(NUM_IDS, device=dev); out = []
    for i in range(0, len(raws), bs):
        ids_, mk = pad([probe_input(r) for r in raws[i:i + bs]]); x = torch.tensor(ids_, device=dev); msk = torch.tensor(mk, device=dev)
        _, h, pos = m.pass_aw(m.mem.zero_state(len(ids_), dev), x, msk, m.lower(x)); out += m.colon_logits(h, pos)[:, num].argmax(-1).tolist()
    return np.array(out)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--bs", type=int, default=250); a_ = ap.parse_args(); dev = a_.device
    s = IO.build_sets(mul_s_p=2); items = {"0-5": s["test2"][2]}
    for f in (6, 7):
        sf = OW.Stream(IO.SEED_FRAMESET + f, frames=(f,), mul_s_p=2, **IO.OWKW); items[str(f)] = [sf.draw(2) for _ in range(IO.TEST_N)]
    for f in (12, 13, 14, 15):
        items[str(f)] = s["framesets"][f][2]
    for k, v in items.items():
        assert len(v) == 1000 and all(it["d"] == 2 for it in v), k
        assert k == "0-5" and all(set(it["frames"]) <= set(range(6)) for it in v) or k != "0-5" and all(set(it["frames"]) == {int(k)} for it in v), k
    out = {"def": __doc__.split("\n")[0], "items_sha": {k: OW.set_sha(v) for k, v in items.items()}, "bs": a_.bs, "device": torch.cuda.get_device_name(dev)}
    a, m, g = E.load(os.path.join(ROOT, "boxes/runs_fsq/s233_askname_s1"), dev); assert m is None, "askname is a direct run"
    out["s233_askname_s1"] = {k: {"acc": float((run_direct(g, v, dev, a_.bs) == np.array([it["values"][1] for it in v])).mean()), "n": len(v)} for k, v in items.items()}
    del g; torch.cuda.empty_cache()
    a, m, g = E.load(os.path.join(ROOT, "boxes/runs_fsq/lh_fh_s1"), dev); res = {}; preds = {}
    for k, v in items.items():
        preds[k] = run_pass1(m, v, dev, a_.bs); res[k] = {"acc": float((preds[k] == np.array([it["values"][1] for it in v])).mean()), "n": len(v)}
    out["lh_fh_s1_pass1_S0"] = res
    ws = json.load(open(os.path.join(ROOT, "boxes/runs_fsq/lh_fh_s1/eval_ws.json")))["items"]["2"]
    assert ws["sig"] == [it["sig"] for it in s["test2"][2]], "eval_ws item order"
    out["path_check_pass1_eq_host_ws_d2"] = bool((preds["0-5"] == np.array(ws["pred_by_pass"]["1"])).all())
    os.makedirs(os.path.join(ROOT, "scratch/s250_s1probe"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "scratch/s250_s1probe/eval_s1probe.json"), "w", newline="\n"), indent=1); print("S1PROBE done; path check", out["path_check_pass1_eq_host_ws_d2"])


if __name__ == "__main__":
    main()
