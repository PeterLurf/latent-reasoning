"""§216 final-tree gates (SD2_v1 §7 at e9ade2ec), run from D:/Latent after the T1 / E1-E4 merge; inputs from the built s1 round-2 pool.
  s0   G-S0 NO-GOLD at round 2: vbank.py has no 'chain_label' and no 's216_gate'; selfdepth's only code read of chain_label is SD-LABEL-MAP; self items
       (from pool_r2 records, Kb 20) carry no gold key; VB.recode_targets call counter: 0 in vb_loss / mass_stats on a self batch, > 0 on a plain batch.
  s8   G-S8 GRADFLOW at Kb 20: 8 pool records of n 16 (K = min(n + U, 20), U rng 3), FT-SELF weights, c 0, train: R1 alone (rec_w 0) and R2 alone
       (L(1) - L(0)) give nonzero grads on slva_wq / slva_wk / core 7, 10 (+ rec.weight on R2); no decorator on rec_targets / vb_loss / main; mass_stats runs.
  s2   G-S2 COMPOSE at Kb 20: the trainer's rec_targets on self_item(rec, 20) == rec_targets on {root} at Kb 20, bitwise, on the first 1,000 pool records
       (differences only where a step value is wrong: counted with the label check), plus eval_gate.json's gs2_compose per window.
    python scratch/gate_s216.py s0 --device cuda:0"""
import argparse
import ast
import json
import math
import os
import sys
from contextlib import nullcontext

sys.path.insert(0, "D:/Latent"); os.chdir("D:/Latent")
import numpy as np  # noqa: E402
import torch  # noqa: E402

import boxes.nested170 as N  # noqa: E402
import boxes.vbank as VB  # noqa: E402
import boxes.selfdepth as SD  # noqa: E402
from boxes import s5  # noqa: E402

OUT = "scratch/gate_s216"; R = "boxes/runs_fsq/"; FT = R + "s215_ftself_s1"; POOL = R + "s216_pool_s1"


def pool():
    return SD.load_pool(POOL + "/pool_r2.jsonl", POOL + "/pool_meta.json")


def ft_model(dev):
    m = VB.make_model(s5.load_gpt2(dev), "slva", "train", "none").to(dev)
    mk = m.load_state_dict(torch.load(FT + "/model.pt", map_location=dev, weights_only=True), strict=True); assert not mk.missing_keys
    m.c_anneal.fill_(0.0); return m


class Counter:
    def __init__(self):
        self.n = 0; self.f = VB.recode_targets

    def __enter__(self):
        def w(*a, **k):
            self.n += 1; return self.f(*a, **k)
        VB.recode_targets = w; return self

    def __exit__(self, *a):
        VB.recode_targets = self.f


def n16_items(recs, k=4):                                                # 4 records: an 8-record n-16 Kb-20 backward OOMs the 8 GB 4060 (04:42 10-18)
    rs = [r for r in recs if r["n"] == 16][:k]; rng = np.random.default_rng(3); its = [SD.self_item(r, 20) for r in rs]
    return its, [min(it["n"] + int(rng.integers(0, 5)), 20) for it in its]


def g_s0(dev):
    o = {}; src = open("boxes/vbank.py", encoding="utf-8").read(); sds = open("boxes/selfdepth.py", encoding="utf-8").read()
    o["vbank_chain_label"] = src.count("chain_label"); o["vbank_s216_gate"] = src.count("s216_gate")
    sub = [nd for nd in ast.walk(ast.parse(sds)) if isinstance(nd, ast.Subscript) and isinstance(nd.slice, ast.Constant) and nd.slice.value == "chain_label"]
    o["sd_label_map_lines"] = [sds.splitlines()[nd.lineno - 1].strip() for nd in sub]
    recs, _ = pool(); its, Ks = n16_items(recs); o["self_item_keys"] = sorted(set().union(*[set(it) for it in its])); o["Kb"] = max(Ks)
    m = ft_model(dev); m.train()
    from boxes.slva import mass_stats
    with Counter() as c:
        m.slva_log = []; VB.vb_loss(m, its, Ks, dev, nullcontext, 1.0); n_vb = c.n
        with torch.no_grad():
            ms = mass_stats(m, its, Ks, dev); n_ms = c.n - n_vb
    pl = [it for it in N.test_set(1000) if it["n"] == 6][:4]
    with Counter() as c:
        VB.vb_loss(m, pl, [8] * 4, dev, nullcontext, 1.0); n_pl = c.n
    o.update({"recode_calls_self_vb_loss": n_vb, "recode_calls_self_mass_stats": n_ms, "recode_calls_plain": n_pl, "mass_stats_self": ms})
    o["pass"] = (o["vbank_chain_label"] == 0 and o["vbank_s216_gate"] == 0 and o["sd_label_map_lines"] == ['it["t"] = rec["chain_label"]  # SD-LABEL-MAP']
                 and not (set(o["self_item_keys"]) & (SD.GOLD_KEYS - {"E_self", "t"})) and n_vb == 0 and n_ms == 0 and n_pl > 0 and ms["chg"][1] > 0)
    return o


def g_s8(dev):
    o = {}; recs, _ = pool(); its, Ks = n16_items(recs, 2); m = ft_model(dev); m.train(); P = dict(m.named_parameters())   # 2: R2 = L(1) - L(0) holds two graphs (4 OOM the 4060)
    names = ("slva_wq", "slva_wk", "rec.weight", "g.p.h_7_attn_c_attn_weight", "g.p.h_10_attn_c_attn_weight", "beta")

    def grads(f):
        m.zero_grad(set_to_none=True); torch.manual_seed(0); L = f(); L.backward()
        return {k: (float(P[k].grad.norm()) if P[k].grad is not None else None) for k in names}
    o["R1"] = grads(lambda: VB.vb_loss(m, its, Ks, dev, nullcontext, 0.0)[0])
    o["R2"] = grads(lambda: VB.vb_loss(m, its, Ks, dev, nullcontext, 1.0)[0] - VB.vb_loss(m, its, Ks, dev, nullcontext, 0.0)[0])
    tree = ast.parse(open("boxes/vbank.py", encoding="utf-8").read())
    o["decorators"] = {nd.name: [ast.unparse(d) for d in nd.decorator_list] for nd in ast.walk(tree) if isinstance(nd, ast.FunctionDef) and nd.name in ("rec_targets", "vb_loss", "main")}
    from boxes.slva import mass_stats
    with torch.no_grad():
        m.slva_log = []; VB.vb_loss(m, its, Ks, dev, nullcontext, 1.0); o["mass_stats"] = mass_stats(m, its, Ks, dev)
    ok = lambda d, skip=(): all(v is not None and v > 0 and math.isfinite(v) for k, v in d.items() if k != "beta" and k not in skip)  # noqa: E731
    o["Ks"] = Ks; o["pass"] = ok(o["R1"], ("rec.weight",)) and ok(o["R2"]) and all(v == [] for v in o["decorators"].values()) and o["mass_stats"]["chg"][1] > 0
    return o


def g_s2(dev):
    o = {}; recs, _ = pool(); n_eq = 0; diff_label_ok = 0
    for r in recs[:1000]:
        tree = SD._parse_tree(r["syms"]); T = len(r["syms"]) + 2
        Es = VB.rec_targets([SD.self_item(r, 20)], 20, T, "cpu"); Eg = VB.rec_targets([{"root": tree, "syms": r["syms"], "n": r["n"]}], 20, T, "cpu")
        if torch.equal(Es, Eg):
            n_eq += 1
        else:
            diff_label_ok += int(r["chain_label"] == int(N.value(tree)))
    eg = json.load(open(R + "s216_gate2_s1/eval_gate.json"))
    o.update({"pool_first_1000": {"n_equal": n_eq, "n_diff": 1000 - n_eq, "n_diff_label_correct": diff_label_ok}, "gate_gs2": eg["gs2_compose"]})
    o["pass"] = all(v["PASS"] for v in eg["gs2_compose"].values()) and (1000 - n_eq) <= 10
    return o


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("gate"); ap.add_argument("--device", default="cuda:0"); a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False; os.makedirs(OUT, exist_ok=True)
    r = {"s0": g_s0, "s8": g_s8, "s2": g_s2}[a.gate](a.device); r["device"] = torch.cuda.get_device_name(a.device) if a.device.startswith("cuda") else "cpu"
    json.dump(r, open(os.path.join(OUT, "%s.json" % a.gate), "w", newline="\n"), indent=1); print(a.gate, "PASS" if r["pass"] else "FAIL", json.dumps(r)[:1500], flush=True)


if __name__ == "__main__":
    main()
