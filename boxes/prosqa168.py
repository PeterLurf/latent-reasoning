"""THEORY §168 DEEP-ProsQA step 1 (a5d7ff5e; scorer theory/score_s168.py): the single-pass (DIRECT) boundary on deep ProsQA.
GPT-2 small (pretrained, boxes.s5.load_gpt2) reads the graph + question in ONE context and answers in Coconut's no-CoT format (PROTOCOL8 P8-0a):
    prefix = question + "\\n";  target = "### " + answer + <EOS>   (CE on the target tokens only)
Data: prosqa/p8data: --data std -> p8_train.json / p8_test.json; --data na -> p8_na_train.json / p8_na_test.json (+ p8_ctl.json, the std control, for both).
Recipe: 24k steps, bs 64 (micro-batched; --auto_micro picks micro at the longest train item), AdamW 1e-4 wd 0.01, warm-up 100 + cosine; ckpts / resume.
eval.json: candidate accuracy (log p of the correct "### <root> is a <target>." > the negative's, as prosqa/eval_p8.py --mode nocot; chance 0.5):
    acc_by_depth {"3".."14"} (depth = len(steps)) on the test file, acc (all), n_by_depth, acc_ctl (p8_ctl.json), margins summary.
    python -m boxes.prosqa168 --data std --seed 1 --out boxes/runs_fsq/s168_direct_std_s1"""
import os
os.environ["BOXES_NB"] = "5"
import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import tiktoken  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from boxes import s5  # noqa: E402
from boxes.fsq120 import driver_version, env_info, lr_at  # noqa: E402

ENC = tiktoken.get_encoding("gpt2"); EOS = 50256
FILES = {"std": ("p8_train.json", "p8_test.json"), "na": ("p8_na_train.json", "p8_na_test.json")}


def items(path):
    """-> list of (prefix ids, correct answer ids, negative answer ids, depth); answers as eval_p8 / analyze.parse ("### " + answer, no EOS)."""
    out = []
    for x in json.load(open(path)):
        sym = x["idx_to_symbol"]; tgt, neg = sym[int(x["target"])], sym[int(x["neg_target"])]; ans = x["answer"]
        a_neg = re.sub(rf"\b{tgt}\.$", f"{neg}.", ans); assert a_neg != ans
        out.append((ENC.encode(x["question"] + "\n"), ENC.encode("### " + ans), ENC.encode("### " + a_neg), len(x["steps"])))
    return out


def train_batch(D, idx, dev):
    """padded ids (B, T) and labels (B, T) (-100 off the target): target = answer + EOS."""
    seqs = [D[i][0] + D[i][1] + [EOS] for i in idx]; T = max(len(s) for s in seqs); ids = np.full((len(idx), T), EOS); lab = np.full((len(idx), T), -100)
    for r, (i, s) in enumerate(zip(idx, seqs)):
        ids[r, :len(s)] = s; n0 = len(D[i][0]); lab[r, n0:len(s)] = s[n0:]
    return torch.tensor(ids, device=dev), torch.tensor(lab, device=dev)


def loss_fn(g, ids, lab, amp):
    with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
        h, _ = g(g.embed(ids), None)
    lg = g.logits(h[:, :-1]).float(); return Fn.cross_entropy(lg.reshape(-1, lg.shape[-1]), lab[:, 1:].reshape(-1), ignore_index=-100)


@torch.no_grad()
def candidate_margins(g, D, dev, amp, bs=16):
    """log p(correct answer) - log p(negative) per item (answer tokens only, given the prefix)."""
    out = []
    for s in range(0, len(D), bs):
        chunk = D[s:s + bs]; seqs, spans = [], []
        for pre, a1, a2, _ in chunk:
            for a in (a1, a2):
                seqs.append(pre + a); spans.append((len(pre), len(pre) + len(a)))
        T = max(len(q) for q in seqs); ids = np.full((len(seqs), T), EOS)
        for r, q in enumerate(seqs):
            ids[r, :len(q)] = q
        x = torch.tensor(ids, device=dev)
        with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
            h, _ = g(g.embed(x), None)
        lp = g.logits(h).float().log_softmax(-1); sc = []
        for r, (a, b) in enumerate(spans):
            tok = x[r, a:b]; sc.append(float(lp[r, a - 1:b - 1].gather(-1, tok[:, None]).sum()))
        out.extend(sc[k] - sc[k + 1] for k in range(0, len(sc), 2))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True, choices=["std", "na"]); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--steps", type=int, default=24000); ap.add_argument("--bs", type=int, default=64); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--micro", type=int, default=1); ap.add_argument("--auto_micro", action="store_true"); ap.add_argument("--mem_budget", type=float, default=21.0)
    ap.add_argument("--mem_check", action="store_true"); ap.add_argument("--ckpt_every", type=int, default=2000); ap.add_argument("--resume", default=None)
    ap.add_argument("--log_every", type=int, default=100); ap.add_argument("--eval_bs", type=int, default=16); ap.add_argument("--train_n", type=int, default=0, help="smokes only")
    ap.add_argument("--test_n", type=int, default=0, help="smokes only"); ap.add_argument("--data_root", default=os.path.join("prosqa", "p8data"))
    ap.add_argument("--fp32", action="store_true"); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--out", required=True)
    a = ap.parse_args(); a.arm = "direct"; os.makedirs(a.out, exist_ok=True); dev = a.device; torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    amp = None if (a.fp32 or not dev.startswith("cuda") or torch.cuda.get_device_capability(dev)[0] < 8) else torch.bfloat16
    root = a.data_root if os.path.isabs(a.data_root) else os.path.join(ROOT, a.data_root); ftr, fte = FILES[a.data]
    Dtr = items(os.path.join(root, ftr)); Dte = items(os.path.join(root, fte)); Dctl = items(os.path.join(root, "p8_ctl.json"))
    if a.train_n:
        Dtr = Dtr[:a.train_n]
    if a.test_n:
        Dte, Dctl = Dte[:a.test_n], Dctl[:a.test_n]
    g = s5.load_gpt2(dev); longest = int(np.argmax([len(x[0]) + len(x[1]) for x in Dtr]))

    def mem_check(micro):
        torch.cuda.reset_peak_memory_stats(dev); ids, lab = train_batch(Dtr, [longest] * (a.bs // micro), dev); g.train(); loss_fn(g, ids, lab, amp).backward(); g.zero_grad(set_to_none=True)
        return torch.cuda.max_memory_allocated(dev) / 2 ** 30
    if a.mem_check or (a.auto_micro and dev.startswith("cuda") and not a.resume):
        mc = []
        while True:
            pk = mem_check(a.micro); mc.append({"micro": a.micro, "peak_gib": pk}); print("MEM", a.micro, pk, flush=True)
            if pk <= a.mem_budget or a.micro >= a.bs or a.mem_check:
                break
            a.micro *= 2
        json.dump({"checks": mc, "micro": a.micro, "peak_gib": mc[-1]["peak_gib"], "longest_tokens": len(Dtr[longest][0]) + len(Dtr[longest][1]) + 1}, open(os.path.join(a.out, "mem_check.json"), "w", newline="\n"), indent=1)
        if a.mem_check:
            return
        assert mc[-1]["peak_gib"] <= a.mem_budget
    info = {**vars(a), "train_items": len(Dtr), "test_items": len(Dte), "format": "Coconut no-CoT: question + '\\n' -> '### ' + answer + EOS (CE on the target only)",
            "train_gpu": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu", "train_driver": driver_version(), **env_info()}
    json.dump(info, open(os.path.join(a.out, ("args_resume_%d.json" % int(time.time())) if a.resume else "args.json"), "w", newline="\n"), indent=1)
    opt = torch.optim.AdamW(g.parameters(), lr=a.lr, weight_decay=0.01); step0 = 1; hist = []; n_nonfinite = 0
    log = open(os.path.join(a.out, "log.jsonl"), "a" if a.resume else "w", newline="\n"); t0 = time.time()
    if a.resume:
        ck = torch.load(a.resume, map_location=dev, weights_only=False); g.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); rng.bit_generator.state = ck["np_rng"]
        step0 = ck["step"] + 1; t0 = time.time() - ck["elapsed"]; n_nonfinite = ck["n_nonfinite"]; log.write(json.dumps({"step": ck["step"], "resumed": a.resume}) + "\n"); log.flush()
    for step in range(step0, a.steps + 1):
        for gr in opt.param_groups:
            gr["lr"] = lr_at(a.lr, gr, step, a.steps)
        g.train(); idx = rng.integers(0, len(Dtr), a.bs); opt.zero_grad(); tot = 0.0; ch = a.bs // a.micro
        for k in range(a.micro):
            ids, lab = train_batch(Dtr, idx[k * ch:(k + 1) * ch], dev); lk = loss_fn(g, ids, lab, amp); (lk / a.micro).backward(); tot += float(lk.item()) / a.micro
        gn = torch.nn.utils.clip_grad_norm_(g.parameters(), 1.0)
        if not (math.isfinite(tot) and bool(torch.isfinite(gn))):
            n_nonfinite += 1; opt.zero_grad(); log.write(json.dumps({"step": step, "nonfinite": 1}) + "\n"); log.flush()
        else:
            opt.step(); hist.append(tot)
        if step % a.log_every == 0 or step == a.steps:
            rec = {"step": step, "loss": float(np.mean(hist)) if hist else None, "sec": round(time.time() - t0)}; hist = []
            log.write(json.dumps(rec) + "\n"); log.flush(); print(rec, flush=True)
        if a.ckpt_every and step % a.ckpt_every == 0 and step < a.steps:
            cp = os.path.join(a.out, "ckpt_last.pt")
            if os.path.exists(cp):
                os.replace(cp, os.path.join(a.out, "ckpt_prev.pt"))
            torch.save({"model": g.state_dict(), "opt": opt.state_dict(), "np_rng": rng.bit_generator.state, "step": step, "elapsed": time.time() - t0, "n_nonfinite": n_nonfinite}, cp)
    train_sec = round(time.time() - t0); torch.save(g.state_dict(), os.path.join(a.out, "model.pt")); g.eval(); t1 = time.time()
    M = candidate_margins(g, Dte, dev, amp, a.eval_bs); dep = np.array([x[3] for x in Dte]); Mc = candidate_margins(g, Dctl, dev, amp, a.eval_bs)
    res = {"arm": "direct", "data": a.data, "seed": a.seed, "steps": a.steps, "acc": float((M > 0).mean()), "n": int(len(M)),
           "acc_by_depth": {str(d): float((M[dep == d] > 0).mean()) for d in range(3, 15) if (dep == d).any()},
           "n_by_depth": {str(d): int((dep == d).sum()) for d in range(3, 15) if (dep == d).any()},
           "margin_median_by_depth": {str(d): float(np.median(M[dep == d])) for d in range(3, 15) if (dep == d).any()},
           "acc_ctl": float((Mc > 0).mean()), "n_ctl": int(len(Mc)), "ctl_file": "p8_ctl.json (the std control, for both arms)",
           "train_sec": train_sec, "eval_sec": round(time.time() - t1), "n_nonfinite": n_nonfinite, "micro": a.micro, "eval_dtype": str(amp),
           "gpu": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu", "train_gpu": info["train_gpu"], "train_driver": info["train_driver"],
           "acc_def": "candidate accuracy: log p('### ' + correct answer) > log p(the negative's), given question + '\\n' (as prosqa/eval_p8.py --mode nocot)"}
    json.dump(res, open(os.path.join(a.out, "eval.json"), "w", newline="\n"), indent=1)
    for ck_ in ("ckpt_last.pt", "ckpt_prev.pt"):
        if os.path.exists(os.path.join(a.out, ck_)):
            os.remove(os.path.join(a.out, ck_))
    print("EVAL", json.dumps({"acc": res["acc"], "acc_by_depth": res["acc_by_depth"], "acc_ctl": res["acc_ctl"]}), flush=True); print("DONE", a.out, flush=True)


if __name__ == "__main__":
    main()
