"""Model adapters for the benchmark harness. Each wraps existing project code; nothing under prosqa/ is modified.

Spec strings (the part after "?" is optional, key=value pairs joined by "&"):
    coconut:<ckpt>?K=6&mask=0     Coconut GPT-2 checkpoint, K continuous thoughts (prosqa/gpt2c.py rollout).
                                  mask=1 for models trained with the answer mask (PROTOCOL6 addendum G, PROTOCOL8 arms).
    nocot:<ckpt>                  answer scored directly after the question (one pass).
    cot:<ckpt>?max_new=220        greedy text chain-of-thought, then both answers scored (prosqa/eval_p8.py).
    f1s:<run_dir>?P=16            per-entity label-channel model (THEORY §54b/§61, prosqa/ail_gpt2.py), P passes.
    gen:<ckpt>?max_new=256        greedy generation for generate tasks; the answer is the text after the last "####"/"###".
    shortcut                      the task's first no-reasoning baseline, scored per item (choice tasks).
    chance                        a coin flip per item (seeded).

predict(item) returns {"margin": float} (choice) or {"text": str} (generate), and "compute": {...} with latent passes
or generated tokens, so accuracy can be compared at matched compute.
"""
import os
import sys
from urllib.parse import parse_qs

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "prosqa"))


def parse_spec(spec):
    kind, _, rest = spec.partition(":")
    path, _, q = rest.partition("?")
    opts = {k: v[0] for k, v in parse_qs(q).items()}
    return kind, path, opts


class Adapter:
    kinds = ("choice",)

    def supports(self, item):
        return item["kind"] in self.kinds


class Coconut(Adapter):
    def __init__(self, ckpt, device, K="6", mask="0"):
        from gpt2c import GPT2C, load_state
        self.m = GPT2C(load_state(ckpt, device)); self.m.mask_answer = mask == "1"; self.K = int(K)

    @torch.no_grad()
    def predict(self, item):
        from analyze import parse
        P = parse(item["raw"])
        lp, _, _ = self.m.rollout(P["ids"], self.K, P["answers"])
        return {"margin": float(lp[0, 0] - lp[0, 1]), "compute": {"latent_passes": self.K, "gen_tokens": 0}}


class TextCoT(Adapter):
    def __init__(self, ckpt, device, max_new="220", cot=True):
        from gpt2c import GPT2C, load_state
        self.m = GPT2C(load_state(ckpt, device)); self.max_new = int(max_new); self.cot = cot

    @torch.no_grad()
    def predict(self, item):
        from analyze import parse
        from eval_p8 import generate, score
        P = parse(item["raw"])
        ids = generate(self.m, P["ids"], self.max_new) if self.cot else P["ids"]
        lp = score(self.m, ids, P["answers"])
        return {"margin": lp[0] - lp[1], "compute": {"latent_passes": 0, "gen_tokens": len(ids) - len(P["ids"])}}


class F1S(Adapter):
    def __init__(self, run_dir, device, P="16"):
        from ail_gpt2 import AIL
        ck = torch.load(os.path.join(run_dir, "model.pt"), map_location="cpu", weights_only=False)
        init = ck["args"]["init"]
        if not os.path.exists(init):                          # trained on a Vast host: use the local copy
            init = os.path.join(ROOT, "prosqa", "ckpt", os.path.basename(init))
        sd = torch.load(init, map_location="cpu", weights_only=True)
        self.m = AIL(sd, "mono"); self.m.load_state_dict(ck["model"]); self.m.to(device).eval()
        self.m.mode, self.m.tau = "hard", None; self.P = int(P); self.dev = device

    @torch.no_grad()
    def predict(self, item):
        from ail_gpt2 import collate
        x = item["raw"]; b = collate([x], self.dev)
        lab, logit, _ = self.m.run(b, self.P)
        t, n = float(lab[0, x["target"]]), float(lab[0, x["neg_target"]])
        margin = (t - n) if t != n else float(logit[0, x["target"]] - logit[0, x["neg_target"]]) * 1e-3
        return {"margin": margin, "compute": {"latent_passes": self.P, "gen_tokens": 0}}


class Generate(Adapter):
    kinds = ("generate",)

    def __init__(self, ckpt, device, max_new="256"):
        import tiktoken
        from gpt2c import GPT2C, load_state
        self.m = GPT2C(load_state(ckpt, device)); self.max_new = int(max_new); self.enc = tiktoken.get_encoding("gpt2")

    @torch.no_grad()
    def predict(self, item):
        from eval_p8 import EOS
        dev = self.m.wte.device
        seq = self.enc.encode(item["raw"]["question"] + "\n"); n0 = len(seq)
        for _ in range(self.max_new):
            _, logits, _ = self.m.forward(self.m.embed(torch.tensor(seq, device=dev))[None])
            t = int(logits[0, -1].argmax())
            if t == EOS:
                break
            seq.append(t)
        text = self.enc.decode(seq[n0:])
        for mark in ("####", "###"):
            if mark in text:
                text = text.rsplit(mark, 1)[1]; break
        return {"text": text, "compute": {"latent_passes": 0, "gen_tokens": len(seq) - n0}}


class Baseline(Adapter):
    def __init__(self, task, which, seed=0):
        self.task, self.which, self.rng = task, which, np.random.default_rng(seed)

    def predict(self, item):
        if self.which == "chance":
            s = float(self.rng.integers(2))
        else:
            s = next(iter(self.task.baselines.values()))(item["raw"])
        return {"score": s, "compute": {"latent_passes": 0, "gen_tokens": 0}}


def load(spec, task, device="cpu"):
    kind, path, opts = parse_spec(spec)
    if kind == "coconut":
        return Coconut(path, device, **opts)
    if kind == "nocot":
        return TextCoT(path, device, cot=False)
    if kind == "cot":
        return TextCoT(path, device, **opts)
    if kind == "f1s":
        return F1S(path, device, **opts)
    if kind == "gen":
        return Generate(path, device, **opts)
    if kind in ("shortcut", "chance"):
        if kind == "shortcut" and not task.baselines:
            raise SystemExit(f"task {task.name} has no no-reasoning baseline")
        return Baseline(task, kind)
    raise SystemExit(f"unknown model spec {spec!r}")
