"""THEORY §98: evaluate the released ICoT-SI GPT-2 checkpoints (Deng et al. 2405.14838; repo
da03/Internalize_CoT_Step_by_Step, commit c2e3891d; Google Drive folders from its README) on the bench items
bench/data/mult_{n}x{n}_test.txt, with our harness (boxes.core.GPT2KV + Meter, full-vocabulary LM head).
Input format as the repo's data.py / generate.py: ' {src} <|endoftext|> ' tokenised, cut after the first EOS; the model
then generates until two more EOS tokens (DoubleEOSStoppingCriteria). Decoding here is GREEDY; the released generate()
passes do_sample=True (recorded in eval.json). Correct = the digits after '####' equal the product's digits.
Weights: base_model.transformer.* -> GPT2KV keys; base_model.lm_head.weight is checked equal to wte (tied).
    python -m boxes.icotsi98 --task 4x4 --ckpt ext/icotsi/4x4/state_dict.bin --out boxes/runs_mult/icotsi_4x4
"""
import os
os.environ["BOXES_NB"] = "5"
import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from boxes import s5  # noqa: E402,F401
from boxes import core as bc  # noqa: E402
from boxes.mult98 import MeterR, load  # noqa: E402

EOS = 50256
DRIVE = {"4x4": "1cCqJmmwJA7wg0Q64f51WgASSnLVJ4ahO", "5x5": "1r80h5rLgAa1NpAzIknuvHB92nDT2j0Hz"}
REPO_COMMIT = "c2e3891d8d577beeaff03fa3b076d1e4db96cfe8"


def load_icot(path, dev):
    sd = torch.load(path, map_location="cpu", weights_only=True)
    m = {k[len("base_model.transformer."):]: v for k, v in sd.items() if k.startswith("base_model.transformer.")}
    assert torch.equal(sd["base_model.lm_head.weight"], m["wte.weight"]), "lm_head is not tied to wte"
    return bc.GPT2KV(m).to(dev)


def src_ids(line):
    src = line.strip().split("||")[0]
    return bc.tok(" " + src + " ") + [EOS]            # == HF tokenizer(' {} {} '.format(src, eos)) cut after the first EOS


@torch.no_grad()
def generate(g, ids, max_new, meter=None):
    """Greedy decode for a batch of equal-length inputs; stop a row after it has produced two EOS."""
    x = torch.tensor(ids); h, past = g(g.embed(x))
    if meter:
        meter.step(x.shape[1], 0, 1)
    out = []; eos_n = torch.zeros(len(ids), dtype=torch.long); t = g.logits(h[:, -1]).argmax(-1)
    for _ in range(max_new):
        t = torch.where(eos_n >= 2, torch.full_like(t, EOS), t); out.append(t); eos_n += (t == EOS).long()
        if (eos_n >= 2).all():
            break
        ctx = past[0][0].shape[2]; h, past = g(g.embed(t)[:, None], past)
        if meter:
            meter.step(1, ctx, 1)
        t = g.logits(h[:, 0]).argmax(-1)
    return torch.stack(out, 1).numpy()


def answer_digits(tokens):
    txt = bc.ENC.decode([int(t) for t in tokens if t != EOS])
    if "####" not in txt:
        return None
    return txt.split("####", 1)[1].split()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--task", required=True, choices=["4x4", "5x5"])
    ap.add_argument("--ckpt", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--bs", type=int, default=100); ap.add_argument("--max_new", type=int, default=48)
    ap.add_argument("--threads", type=int, default=3); ap.add_argument("--test_n", type=int, default=None)
    a = ap.parse_args(); torch.set_num_threads(a.threads); dev = "cpu"
    path = os.path.join(ROOT, "bench", "data", f"mult_{a.task}_test.txt")
    lines = open(path).read().splitlines()[:a.test_n]; D = load(path, limit=a.test_n)
    g = load_icot(a.ckpt, dev).eval(); t0 = time.time(); ok = []; preds = []; meter = MeterR(g)
    ids = [src_ids(l) for l in lines]; assert len({len(i) for i in ids}) == 1
    for s in range(0, len(ids), a.bs):
        dec = generate(g, ids[s:s + a.bs], a.max_new, meter if s == 0 else None)
        for i, row in enumerate(dec):
            d = answer_digits(row); truth = [str(x) for x in D["prod"][s + i]]
            ok.append(d == truth); preds.append(bc.ENC.decode([int(t) for t in row if t != EOS]))
    sha = hashlib.sha256(open(a.ckpt, "rb").read()).hexdigest()
    res = {"arm": "icotsi", "task": a.task, "acc_test": float(np.mean(ok)), "test_items": len(ok), "gpu": "cpu",
           "compute": meter.as_dict(),
           "checkpoint": {"path": a.ckpt, "sha256": sha, "drive_folder": DRIVE[a.task], "repo": "da03/Internalize_CoT_Step_by_Step",
                          "repo_commit": REPO_COMMIT, "config": json.load(open(os.path.join(os.path.dirname(a.ckpt), "config.json")))},
           "decoding": "greedy, stop after two EOS (the released generate() passes do_sample=True)",
           "training": {"epochs": 200, "flops": None, "note": "released checkpoint; recipe per README/Literature e206d92d (about 200 epochs over 808k problems), not metered"},
           "repo_log_acc": {"4x4": 1.0, "5x5": 0.996}[a.task], "eval_sec": round(time.time() - t0)}
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "eval.json"), "w", newline="\n"), indent=1)
    json.dump(preds, open(os.path.join(a.out, "preds.json"), "w", newline="\n"))
    print("EVAL", a.task, res["acc_test"], "flops", res["compute"]["flops"], "passes", res["compute"]["sequential_passes"], flush=True)


if __name__ == "__main__":
    main()
