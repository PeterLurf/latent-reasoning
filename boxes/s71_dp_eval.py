"""THEORY §71: predictions of a trained DeltaProduct (automl/DeltaProduct state_tracking, run unchanged) on the first
2,000 sequences of S5=512.csv, for the shared scorer. Runs inside the DeltaProduct repo's state_tracking dir on Linux:
    PYTHONPATH=$PWD python <this file> --ckpt checkpoints/<run dir> --test data/S5=512.csv --out preds_test.npy
Tokenizer rebuilt exactly as main.get_dataset: WordLevel over the element strings added in integer order, then the
special tokens; input = [BOS] + elements; the model's output at position t (t >= 1) predicts the label of element t.
"""
import argparse
import csv
import json

import numpy as np
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from src.gated_delta_product import GatedDeltaProduct
from fla.models import GatedDeltaProductConfig

SPECIAL = ["[PAD]", "[BOS]", "[UNK]", "[EOS]", "[SEP]", "[CLS]", "[MASK]"]

ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--test", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--n", type=int, default=2000); ap.add_argument("--bs", type=int, default=250)
a = ap.parse_args()
X, Y = [], []
with open(a.test, newline="") as f:
    for i, r in enumerate(csv.DictReader(f)):
        if i >= a.n:
            break
        X.append(r["input"].split()); Y.append([int(v) for v in r["target"].split()])
tok = Tokenizer(WordLevel()); tok.add_tokens([str(i) for i in range(120)]); tok.add_special_tokens(SPECIAL)
ids = {str(i): tok.token_to_id(str(i)) for i in range(120)}; bos = tok.token_to_id("[BOS]")
inv = {v: int(k) for k, v in ids.items()}
conf = GatedDeltaProductConfig(**{k: v for k, v in json.load(open(f"{a.ckpt}/config.json")).items() if k not in ("model_type", "architectures", "transformers_version")})
m = GatedDeltaProduct(conf); m.load_state_dict(torch.load(f"{a.ckpt}/pytorch_model.bin", map_location="cpu")); m.to(torch.bfloat16).cuda().eval()
preds = []
with torch.no_grad():
    for s in range(0, len(X), a.bs):
        x = torch.tensor([[bos] + [ids[t] for t in row] for row in X[s:s + a.bs]], device="cuda")
        out = m(x).argmax(-1)[:, 1:].cpu().numpy()
        preds.append(np.vectorize(lambda v: inv.get(int(v), -1))(out))
P = np.concatenate(preds); Ya = np.array(Y)
np.save(a.out, P)
cum = np.cumprod(P == Ya, axis=1).mean(0)
print(json.dumps({"tokenizer_ids_are_identity": all(ids[str(i)] == i for i in range(120)), "bos": bos,
                  "cumulative": {t: float(cum[t - 1]) for t in (128, 256, 512)}, "final_512": float((P[:, -1] == Ya[:, -1]).mean())}))
