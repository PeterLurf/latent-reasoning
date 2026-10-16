"""VB-2G Stage 0, note 211-15 C (user-approved Stage 0, $0, eval-only): D §1.2 RESTART-SPLIT and PREFIX-FACTORIAL on INJECT nested170 models with §175's
aux decoder (§190 s1 / s2, §193 s1 / s2), with the same eval_rs.json / eval_pf.json contracts as boxes/vbank_diag.
Decode at pass t: aux_greedy(S_t, t, L) (the model's own prompt; §190's --aux_nopass honoured); EXACT iff the generated ids == ids(cot_steps[t]) followed by
EOT. Answers: InjectN.dec_logits after K' passes (intact K' = n + 2; restart = the item reduced t times run fresh at K' = r + 2). LONG-STEP(m): chains
re-rooted after every pass (one pass, aux decode at t = 1 vs the next step), stopping at the first inexact step. PF: vbank_diag.pf_items (same seeds),
S(k, s) = P(correct at K' = k + r + 2 | the aux decode at k == Q). The model's own eval settings: bf16 autocast on capability >= 8, --bs (default 250),
deltamem.FAST_ADD from args; card / bs / dtype recorded.
    python scratch/inject_rspf.py s190_rmsnopass_s1 --part rs --device cuda:0"""
import argparse
import json
import os
import sys
import time
from contextlib import nullcontext

sys.path.insert(0, "D:/Latent")
import numpy as np  # noqa: E402
import torch  # noqa: E402

import boxes.deltamem as DM  # noqa: E402
import boxes.halt173 as H  # noqa: E402
import boxes.nested170 as N  # noqa: E402
from boxes.vbank_diag import reduce_k, n_ops, pf_items  # noqa: E402

EOT = N.EOT


class Runner:
    def __init__(self, m, dev, amp, bs):
        self.m, self.dev, self.bs = m, dev, bs; self.ctx = (lambda: torch.autocast("cuda", dtype=amp)) if amp is not None else nullcontext

    @torch.no_grad()
    def roll(self, items, K, decode_at=(), answer_at=()):
        """-> (answers {K': (b,) argmax value}, exact {t: (b,) bool}) for one batch of items."""
        m = self.m; x, msk = m.think_inputs(items, self.dev); ans, ex = {}, {}
        with self.ctx():
            S = m.mem.zero_state(len(items), self.dev)
            if 0 in answer_at:
                ans[0] = m.dec_logits(S, items, self.dev)[:, N.NUM_IDS].argmax(-1).cpu().numpy()
            for t in range(1, K + 1):
                S = m.write(S, x, msk)
                if t in decode_at:
                    tg = [N.ids(N.cot_steps(it["root"])[t]) + [EOT] if t < len(N.cot_steps(it["root"])) else None for it in items]
                    L = max(len(z) for z in tg if z is not None) if any(z is not None for z in tg) else 1
                    gen = m.aux_greedy(S, t, L); ex[t] = np.array([z is not None and list(gen[i, :len(z)]) == z for i, z in enumerate(tg)])
                if t in answer_at:
                    ans[t] = m.dec_logits(S, items, self.dev)[:, N.NUM_IDS].argmax(-1).cpu().numpy()
        return ans, ex


def mk(root):
    s_ = N.render(root); return {"syms": s_, "root": root, "t": N.value(root), "depth": N.depth(s_), "n": n_ops(root)}


def part_rs(R, TE, bs):
    Nn = np.array([it["n"] for it in TE]); cells = {}; jobs = {r: [] for r in (2, 4, 6, 8)}; intact = {}
    for n in range(3, 17):
        cuts = {n - r: r for r in (2, 4, 6, 8) if n - r >= 1}; idx = np.nonzero(Nn == n)[0]; dec = {r: [0, 0] for r in cuts.values()}
        for c0 in range(0, len(idx), bs):
            ch = idx[c0:c0 + bs]; its = [TE[j] for j in ch]; ans, ex = R.roll(its, n + 2, decode_at=tuple(cuts), answer_at=(n + 2,))
            ok = ans[n + 2] == np.array([it["t"] for it in its])
            for k_, j in enumerate(ch):
                intact[int(j)] = bool(ok[k_])
                for t, r in cuts.items():
                    dec[r][0] += 1
                    if ex[t][k_]:
                        dec[r][1] += 1; jobs[r].append((int(j), t))
        for r, (tot, n_ex) in dec.items():
            cells["%d_%d" % (n, r)] = {"n": n, "r": r, "t": n - r, "exact_rate": n_ex / tot if tot else None}
        print("RS intact n", n, round(float(np.mean([intact[int(j)] for j in idx])), 3), flush=True)
    rest = {}
    for r, js in jobs.items():
        for c0 in range(0, len(js), bs):
            part = js[c0:c0 + bs]; its = [mk(reduce_k(TE[j]["root"], t)) for j, t in part]
            ans, _ = R.roll(its, r + 2, answer_at=(r + 2,)); a_ = ans[r + 2]
            for k_, (j, t) in enumerate(part):
                rest[(j, t)] = bool(a_[k_] == TE[j]["t"])
    for key, c_ in cells.items():
        n, r, t = c_["n"], c_["r"], c_["t"]; js = [j for (j, tt) in jobs[r] if tt == t and TE[j]["n"] == n]
        c_.update({"n_items": len(js), "intact": float(np.mean([intact[j] for j in js])) if js else None, "restart": float(np.mean([rest[(j, t)] for j in js])) if js else None,
                   "unparse_rate": None, "intact_all": float(np.mean([intact[int(j)] for j in np.nonzero(Nn == n)[0]]))})
    ls = long_step(R, TE, bs)
    return {"cells": cells, "long_step": ls, "def": "D §1.2 RS on INJECT: decode = aux_greedy at pass t, exact = ids(cot_steps[t]) + EOT; intact at K' = n + 2; restart = the item "
            "reduced t times run fresh at K' = r + 2; unparse_rate is not defined for a greedy token decode (null); exact_rate = exact / all items of n"}


def long_step(R, TE, bs):
    cur = {j: N._copy(it["root"]) for j, it in enumerate(TE)}; st = {}
    while cur:
        idx = sorted(cur, key=lambda j: len(N.render(cur[j]))); nxt = {}
        for c0 in range(0, len(idx), bs):
            part = idx[c0:c0 + bs]; its = [mk(cur[j]) for j in part]; _, ex = R.roll(its, 1, decode_at=(1,))
            for k_, j in enumerate(part):
                mo = n_ops(cur[j]); s_ = st.setdefault(str(mo), [0, 0]); s_[0] += 1; s_[1] += int(ex[1][k_])
                red = reduce_k(cur[j], 1)
                if ex[1][k_] and red.op is not None:
                    nxt[j] = red
        cur = nxt
    return {k: {"n_steps": v[0], "exact": v[1] / v[0], "n_exact": v[1]} for k, v in sorted(st.items(), key=lambda kv: int(kv[0]))}


def part_pf(R, TE, bs):
    grid, S0 = {}, {}
    for r in (2, 3):
        items = pf_items(r, 20000); ok0 = []
        for c0 in range(0, len(items), bs):
            its = [mk(reduce_k(it["root"], it["k"])) for it in items[c0:c0 + bs]]; ans, _ = R.roll(its, r + 2, answer_at=(r + 2,))
            ok0 += list(ans[r + 2] == np.array([it["t"] for it in its]))
        S0[str(r)] = float(np.mean(ok0))
        for k in range(1, 13):
            ks = [it for it in items if it["k"] == k]; K = k + r + 2
            for c0 in range(0, len(ks), bs):
                its = [mk(it["root"]) for it in ks[c0:c0 + bs]]; ans, ex = R.roll(its, K, decode_at=(k,), answer_at=(K,))
                ok = ans[K] == np.array([it["t"] for it in its])
                for i_, it in enumerate(ks[c0:c0 + bs]):
                    c_ = grid.setdefault("%d_%d_%d" % (k, it["s"], r), {"n_items": 0, "n_correct": 0, "n_draws": 0, "r": r, "k": k, "s": it["s"]}); c_["n_draws"] += 1
                    if ex[k][i_]:
                        c_["n_items"] += 1; c_["n_correct"] += int(ok[i_])
        print("PF r", r, "S0", S0[str(r)], flush=True)
    for c_ in grid.values():
        c_["S"] = c_["n_correct"] / c_["n_items"] if c_["n_items"] else None; c_["exact_rate"] = c_["n_items"] / c_["n_draws"]
    return {"grid": grid, "S0": S0, "def": "D §1.2 PF on INJECT: vbank_diag.pf_items (seeds 21402 / 21403); exact at k = aux_greedy(S_k, k) == ids(Q) + EOT; S0 = fresh Q at K' = r + 2"}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--part", required=True, choices=["rs", "pf"]); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--bs", type=int, default=250); a = ap.parse_args(); dev = a.device; rd = os.path.join("D:/Latent/boxes/runs_fsq", a.run); t0 = time.time()
    kind, m, ra = H.load_model(rd, dev); assert kind == "nested" and ra.get("aux_step"), "needs an INJECT run with §175's aux decoder"
    DM.FAST_ADD = bool(ra.get("fast_add")); DM.CHUNK_DELTA = 0; DM.READ_CLAMP = None
    amp = torch.bfloat16 if (dev.startswith("cuda") and torch.cuda.get_device_capability(dev)[0] >= 8) else None
    R = Runner(m, dev, amp, a.bs); TE = N.test_set(int(ra.get("per_n") or 1000))
    res = {"rs": part_rs, "pf": part_pf}[a.part](R, TE, a.bs)
    res.update({"run": a.run, "card": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu", "bs": a.bs, "eval_bs": a.bs,
                "dtype": "bf16 autocast trunk, fp32 memory branch" if amp else "fp32", "fast_add": DM.FAST_ADD, "aux_nopass": bool(ra.get("aux_nopass")), "sec": round(time.time() - t0)})
    fo = os.path.join(rd, "eval_%s.json" % a.part); json.dump(res, open(fo + ".tmp", "w", newline="\n"), indent=1); os.replace(fo + ".tmp", fo); print("DONE", fo, res["sec"], flush=True)


if __name__ == "__main__":
    main()
