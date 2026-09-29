"""Exploratory (addendum G follow-up): accuracy of the bypass-blocked model with K' = 0, 1, 2, 3 thoughts, by path length.
K' = 1 feeds only the final hidden state of the prefix pass (at <start-latent>) as the single thought."""
import json, sys
sys.path.insert(0, "prosqa")
import torch
from analyze import kprime
from gpt2c import GPT2C, load_state
ckpt, out, dev = sys.argv[1], sys.argv[2], sys.argv[3]
torch.backends.cuda.matmul.allow_tf32 = False
m = GPT2C(load_state(ckpt, dev)); m.mask_answer = True
data = json.load(open("prosqa/data/prosqa_test.json"))
json.dump(kprime(m, data, 500, budgets=(0, 1, 2, 3)), open(out, "w"), indent=1)
