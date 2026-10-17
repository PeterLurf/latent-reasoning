"""PROOFS_SELFDEPTH check 4 ($0, CPU, 1 thread; imports the draft's compose.py read-only, never calls its main()):
 (a) chain-exact (213-33's string-level definition: rerender(dec) == render(true one-step reduction)) does NOT imply COMPOSE-VALID:
     a decoder that puts the value on the span's FIRST slot is chain-exact at every step and fails SPAN / ORDER on every item;
 (b) on COMPOSE-VALID chains, E_self == recode_targets(root) at every t  <=>  every step value is correct; and E_self == gold => label == gold
     (so state precision pi^E <= value precision pi on the same denominator); compensated chains (some step wrong, final right) exist;
 (c) the padded slot count T of vbank's --mem_check batch (the 64 longest of 512 train_items at n_max 8, rng 0) - the T the 11.28 GiB was measured at;
 (d) the analytic item-space bound: max_s P(render = s) <= (1/11) * 40^-n, so expected duplicate pairs among D draws <= D^2/2 * that."""
import os, sys, json
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, "D:/Latent"); sys.path.insert(0, "D:/Latent/docs/research_selfdepth_2026-10-17/scripts")
import numpy as np
import torch
torch.set_num_threads(1)
import boxes.nested170 as N
import boxes.vbank as VB
import compose as C

out = {}
# (a)
def value_first_all(cur):
    d = list(C.perfect(cur)); T = len(d); voids = [j for j in range(1, T - 1) if d[j] == VB.VOID]; vj = voids[-1] + 1
    d[voids[0]], d[vj] = d[vj], VB.VOID
    return d
rng = np.random.default_rng(31700); a = {"items": 0, "string_chain_exact_all_steps": 0, "compose_valid": 0, "keep": 0}
for _ in range(1000):
    n = int(rng.integers(2, 13)); root, y = N.generate(rng, n); cur = N._copy(root); ok = True
    while cur.op is not None:                                          # string-level chain-exactness of the corrupted decode
        dec = value_first_all(cur); s_ = VB.rerender(dec); nxt = N._copy(cur)
        path = N._leftmost_ready(nxt); h = nxt
        for p_ in path:
            h = getattr(h, p_)
        h.v = N.apply(h.op, h.l.v, h.r.v); h.op = h.l = h.r = None
        ok = ok and s_ == N.render(nxt); cur = nxt
    r = C.chain_compose(N.render(root), value_first_all)
    a["items"] += 1; a["string_chain_exact_all_steps"] += ok; a["compose_valid"] += r["compose_valid"]; a["keep"] += r["keep"]
out["a_chain_exact_not_compose_valid"] = a; print("(a)", a, flush=True)

# (b) single wrong value at one random step (the model-like error), then exact steps
def one_wrong(kstep):
    st = {"k": 0}
    def dec_fn(cur):
        d = list(C.perfect(cur)); st["k"] += 1
        if st["k"] == kstep:
            T = len(d); voids = [j for j in range(1, T - 1) if d[j] == VB.VOID]; vj = voids[-1] + 1
            d[vj] = VB.VAL0 + (d[vj] - VB.VAL0 + 1 + int(rng.integers(0, N.P - 1))) % N.P
        return d
    return dec_fn
N.INV[0] = 0                                                           # a '/ 0' handle can only follow a corrupted value (as compose.py's fixtures)
b = {"items": 0, "cv": 0, "E_eq_gold": 0, "E_eq_gold_label_wrong": 0, "label_right_E_wrong(compensated)": 0}
for _ in range(3000):
    n = int(rng.integers(2, 13)); root, y = N.generate(rng, n); r = C.chain_compose(N.render(root), one_wrong(int(rng.integers(1, n + 1))))
    b["items"] += 1
    if r["compose_valid"]:
        b["cv"] += 1; eq = r["E_self"] == VB.recode_targets(root); b["E_eq_gold"] += eq
        b["E_eq_gold_label_wrong"] += bool(eq and r["label"] != y); b["label_right_E_wrong(compensated)"] += bool((not eq) and r["label"] == y)
out["b_one_wrong_step"] = b; print("(b)", b, flush=True)
N.INV.pop(0, None)

# (c)
big = sorted(N.train_items(np.random.default_rng(0), 8 * 64, 8, set()), key=lambda it: -len(it["syms"]))[:64]
Tb = max(len(it["syms"]) for it in big) + 2; Tmin = min(len(it["syms"]) for it in big) + 2
out["c_mem_check_batch_T"] = {"padded_T": Tb, "min_T_in_batch": Tmin}
def mem(Kb, T, Tref):
    return 2 + 0.58 * Kb * T / Tref + 0.0258 * (T / Tref) ** 2 * sum(t + 1 for t in range(1, Kb + 1))
out["c_mem_projection"] = {f"K{K}_T{T}": {"Tref33": round(mem(K, T, 33), 2), "Tref_actual": round(mem(K, T, Tb), 2)} for K, T in ((16, 47), (14, 47), (20, 61), (18, 61))}
print("(c)", out["c_mem_check_batch_T"], out["c_mem_projection"], flush=True)

# (d)
pairs_min = min(len(v) for v in N.PAIRS.values())
dbound = {}
for n in (4, 5, 9, 12, 16, 20):
    pmax = (1 / 11) * (1 / (4 * pairs_min)) ** n; D = 135000 if n >= 9 else (5000 if n == 4 else 50000)
    dbound[n] = {"pmax_bound": pmax, "draws": D, "E_dup_pairs_bound": D * D / 2 * pmax}
out["d_item_space"] = {"min_PAIRS": pairs_min, "bound": dbound}; print("(d)", out["d_item_space"], flush=True)
json.dump(out, open(sys.argv[1], "w", newline="\n"), indent=1)
