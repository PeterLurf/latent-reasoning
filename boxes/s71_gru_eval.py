"""THEORY §71: predictions of a trained GRU (word-problem train_gru, 1 layer, defaults) on the first 2,000 sequences of
S5=512.csv. Runs inside the word-problem repo on Linux. The tokenizer and the tokenisation come from the repo's own
get_dataset / tokenize (so ids and the [BOS] prefix are exactly as in training).
    PYTHONPATH=$PWD python <this file> --ckpt gru_ckpt/best_<seed>.pt --out preds_test.npy
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

import src.main as wp
from sfirah.rnn import GRUTokenClassifier

ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=2000); ap.add_argument("--bs", type=int, default=100)
a = ap.parse_args()
d = wp.get_dataset(group="S5", k=128, strict_len=True, train_size=1.0, data_dir=Path("data"))
tok, n_vocab = d["tokenizer"], d["n_vocab"]
X, Y = [], []
with open("data/S5=512.csv", newline="") as f:
    for i, r in enumerate(csv.DictReader(f)):
        if i >= a.n:
            break
        X.append(r["input"]); Y.append([int(v) for v in r["target"].split()])
m = GRUTokenClassifier(batch_first=True, bias=True, dropout=0.0, d_embedding=512, d_hidden=2048, n_layers=1, n_vocab=n_vocab,
                       weight_scale=1.0)
m.load_state_dict(torch.load(a.ckpt, map_location="cpu")); m.cuda().eval()
inv = {tok.convert_tokens_to_ids(str(i)): i for i in range(120)}
P = []
with torch.no_grad():
    for s in range(0, len(X), a.bs):
        ids = tok(X[s:s + a.bs], return_tensors="pt", padding=True)["input_ids"].cuda()
        out = m(ids).argmax(-1)[:, 1:].cpu().numpy()
        P.append(np.vectorize(lambda v: inv.get(int(v), -1))(out))
P = np.concatenate(P); Ya = np.array(Y); np.save(a.out, P)
cum = np.cumprod(P == Ya, axis=1).mean(0)
print(json.dumps({"n_vocab": n_vocab, "ids_identity": all(k == v for k, v in inv.items()),
                  "cumulative": {t: float(cum[t - 1]) for t in (128, 256, 512)}, "final_512": float((P[:, -1] == Ya[:, -1]).mean())}))
