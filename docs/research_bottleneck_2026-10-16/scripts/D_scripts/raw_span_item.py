import sys
sys.path.insert(0, "D:/Latent"); sys.path.insert(0, __import__("os").path.dirname(__file__))
import numpy as np, boxes.nested170 as N
from local_rule_cov import raw_states, handle_index
def spans(root):
    out = []
    for z in raw_states(root)[:-1]:
        h = handle_index(z); alive = [i for i, s in enumerate(z) if s != "V"]; j = alive.index(h)
        out.append(alive[min(len(alive) - 1, j + 2)] - alive[max(0, j - 2)] + 1)
    return out
rng = np.random.default_rng(31500)
mx = 33
for n in (9, 10, 12, 14, 16):
    it = [max(spans(N.generate(rng, n)[0])) > mx for _ in range(1000)]
    print(n, "items with some step needing a raw span > train max 33:", round(float(np.mean(it)), 3))
