"""THEORY §114 (notes 114-1): looped depth per op on 4x4 multiplication, in §107's harness (accel99 MARKOV segments
[a, *, op j-1's carried latent (or the chunk start text), op j] -> slot j, fresh cache per segment, full own-segment forks;
K-CURR 1 / 2 / 4 at 2k / 6k; eligible from 6k; valid-split early stop; 24k cap).
Looped block: GPT-2 layers 4-7 run r times over the segment's positions (and over every slot / fork token); iteration
t >= 2 takes h_out(t-1) + h3 (input injection; h3 = the layer-3 output of the same positions); tied base weights plus a
per-(iteration, layer) LoRA (rank 8, scale 1, on attn c_attn and mlp c_fc; B zero-initialised, so r = 1 is GPT-2 exactly);
layers 8-11 run once on the final iteration. Test r > 8 reuses iteration 8's LoRA (114-1).
Per-iteration targets (114-1, unbiased 2-sample estimator): each training step draws r ~ U{1..8} (loopr; r = 1 for loop1)
and u ~ U{1..r-1}; the slot + fork CE is computed for the network truncated after iteration r (the chain's carried latent
is iteration r's) and after iteration u (layers 8-11 on iteration u's output, then that slot and fork through the same
truncated network); per slot loss = (CE_r + (r - 1) CE_u) / r, an unbiased estimate of the mean over iterations 1..r.
The segment trunk (layers 0-3 and the loop iterations) is shared between the two targets.
Arms: loopr (test r = 8), loop1 (r = 1 always; same code path).
Loss/null ratio (111-1 analogue for multiplication): every 100 steps, the same batch's loss with every carried latent
zeroed (no grad) is the model-relative null; ratio = loss / null (k >= 2 only). early_escape = ratio < 0.9 at some log
within 1k steps after k = 2 starts.
Eval (eval.json): §107's (test_acc, per_slot_decode_acc, removal zero / mean of every carried latent, steps_to_threshold,
curve) + donor (§104 single patch of the carried latent at slots 1-3, n = 2000, random donors and same-a donors; the
final answer scored donor-consistent = S_j^B + a_A sum_{t>j} b_t^A 10^(t-1), recipient-consistent = a_A b_A),
acc_by_r {1, 2, 4, 8, 12, 16}, ratio_per_1k / early_escape, compute (Meter FLOPs scaled to the looped depth, layer depth).
    python -m boxes.loop114 --arm loopr --seed 1 --out boxes/runs_mult/a114_loopr_4x4_s1
"""
import os
os.environ["BOXES_NB"] = "5"
import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from boxes import s5  # noqa: E402
from boxes.accel99 import ARMS, MultGen99, config, latent_batch, rollout_safe  # noqa: E402
from boxes.mult98 import DIGIT_IDS, HASH, TOK, MeterR, gen_mean_inputs, gen_rollout, load  # noqa: E402

N_LORA, RANK, LOOP_LAYERS = 8, 8, (4, 5, 6, 7)


def depth(r):
    return 8 + 4 * r


def mstep(meter, T, ctx, rows, dep):
    """Meter step for a forward of `dep` layers (Meter assumes 12): the body FLOPs are scaled by dep / 12."""
    if meter is None:
        return
    f0 = meter.flops; meter.step(T, ctx, rows); body = meter.flops - f0 - 2 * 768 * 50257 * rows
    meter.flops += body * (dep / 12 - 1)


class Looper(nn.Module):
    def __init__(self, g):
        super().__init__()
        self.g = g; self.lora = nn.ParameterDict()
        for t in range(1, N_LORA + 1):
            for i in LOOP_LAYERS:
                for nm, din, dout in (("attn", 768, 3 * 768), ("fc", 768, 4 * 768)):
                    self.lora[f"t{t}_l{i}_{nm}_A"] = nn.Parameter(torch.randn(din, RANK) / math.sqrt(din))
                    self.lora[f"t{t}_l{i}_{nm}_B"] = nn.Parameter(torch.zeros(RANK, dout))

    def layer(self, h, i, t, kv, P0):
        g = self.g; B, T, D = h.shape; H = 12; hd = D // H; pre = f"h.{i}."; lt = None if t is None else min(t, N_LORA)
        a = F.layer_norm(h, (D,), g.w(pre + "ln_1.weight"), g.w(pre + "ln_1.bias"))
        qkv = a @ g.w(pre + "attn.c_attn.weight") + g.w(pre + "attn.c_attn.bias")
        if lt is not None:
            qkv = qkv + (a @ self.lora[f"t{lt}_l{i}_attn_A"].to(a.dtype)) @ self.lora[f"t{lt}_l{i}_attn_B"].to(a.dtype)
        q, k, v = (z.view(B, T, H, hd).transpose(1, 2) for z in qkv.split(D, -1))
        if kv is not None:
            k = torch.cat([kv[0], k], 2); v = torch.cat([kv[1], v], 2)
        if P0 == 0:
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=torch.ones(T, P0 + T, dtype=torch.bool, device=h.device).tril(P0))
        h = h + o.transpose(1, 2).reshape(B, T, D) @ g.w(pre + "attn.c_proj.weight") + g.w(pre + "attn.c_proj.bias")
        a = F.layer_norm(h, (D,), g.w(pre + "ln_2.weight"), g.w(pre + "ln_2.bias"))
        fc = a @ g.w(pre + "mlp.c_fc.weight") + g.w(pre + "mlp.c_fc.bias")
        if lt is not None:
            fc = fc + (a @ self.lora[f"t{lt}_l{i}_fc_A"].to(a.dtype)) @ self.lora[f"t{lt}_l{i}_fc_B"].to(a.dtype)
        h = h + F.gelu(fc, approximate="tanh") @ g.w(pre + "mlp.c_proj.weight") + g.w(pre + "mlp.c_proj.bias")
        return h, (k, v)

    @staticmethod
    def sched(r):
        return [(i, None) for i in range(4)] + [(i, t) for t in range(1, r + 1) for i in LOOP_LAYERS] + [(i, None) for i in range(8, 12)]

    def fwd(self, x, past, r):
        """New tokens x (B, T, D) through the network truncated after iteration r, appended after `past` (a list aligned
        with sched(r)). Returns ln_f output and the new past."""
        g = self.g; P0 = 0 if past is None else past[0][0].shape[2]; T = x.shape[1]
        h = x + g.w("wpe.weight")[P0:P0 + T][None]; new = []; h3 = None
        for n, (i, t) in enumerate(self.sched(r)):
            if t is not None and t >= 2 and i == LOOP_LAYERS[0]:
                h = h + h3
            h, kv = self.layer(h, i, t, None if past is None else past[n], P0); new.append(kv)
            if i == 3:
                h3 = h
        return F.layer_norm(h, (768,), g.w("ln_f.weight"), g.w("ln_f.bias")), new

    def trunk(self, x, R, branches):
        """A fresh segment (positions from 0) through layers 0-3 and loop iterations 1..R (shared), with layers 8-11 run
        for each iteration in `branches`. Returns {t: (ln_f output, past aligned with sched(t))}."""
        g = self.g; T = x.shape[1]; h = x + g.w("wpe.weight")[:T][None]; c03 = []
        for i in range(4):
            h, kv = self.layer(h, i, None, None, 0); c03.append(kv)
        h3 = h; cl = []; out = {}
        for t in range(1, R + 1):
            if t >= 2:
                h = h + h3
            for i in LOOP_LAYERS:
                h, kv = self.layer(h, i, t, None, 0); cl.append(kv)
            if t in branches:
                hb = h; cb = []
                for i in range(8, 12):
                    hb, kv = self.layer(hb, i, None, None, 0); cb.append(kv)
                out[t] = (F.layer_norm(hb, (768,), g.w("ln_f.weight"), g.w("ln_f.bias")), c03 + list(cl) + cb)
        return out


class Loop114(MultGen99):
    def __init__(self, g):
        super().__init__(g, "genm", std=False)
        self.markov = True; self.lp = Looper(g); self.r = 1; self.u = None; self.override = None

    def fork_tf_l(self, hslot, past, tgt, r):
        g = self.g
        hf = self.lp.fwd(g.embed(tgt[:, :-1]), past, r)[0] if tgt.shape[1] > 1 else hslot[:, None][:, :0]
        lg = g.logits(torch.cat([hslot[:, None], hf], 1)).float()
        return F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1), reduction="none").view(tgt.shape).sum(-1).mean()

    def fork_greedy_l(self, hslot, past, L, r, meter=None):
        g = self.g; t = g.logits(hslot).argmax(-1); out = [t]
        if meter:
            meter.step(0, 0, 1)
        for _ in range(L - 1):
            ctx = past[0][0].shape[2]; h, past = self.lp.fwd(g.embed(t)[:, None], past, r); mstep(meter, 1, ctx, 1, depth(r))
            t = g.logits(h[:, 0]).argmax(-1); out.append(t)
        return torch.stack(out, 1)

    def chunk(self, prefix, start, ops, step_tgt=None, ans_tgt=None, step_len=None, ans_len=None, decode="all",
              ablate=None, mean_in=None, meter=None, meter_x=None, snap=False, collect=False):
        """accel99's MARKOV chunk with the looped block (self.r) and, in training, the second per-iteration target self.u."""
        B, c = ops.shape; g = self.g; R = self.r; U = self.u if step_tgt is not None else None
        x = torch.cat([g.embed(prefix)] + ([g.embed(start)] if start is not None else []) + [g.embed(self.digtok[ops[:, 0]])[:, None]], 1)
        losses, decs, ins = [], [], []; ans = None
        for j in range(c):
            br = self.lp.trunk(x, R, {R} | ({U} if U else set())); mstep(meter, x.shape[1], 0, 0, depth(R))
            out_R, past_R = br[R]; s_in = out_R[:, -1]
            if ablate in ("zero", "mean") and j < c - 1:
                s_in = torch.zeros_like(s_in) if ablate == "zero" else mean_in[j][None].expand_as(s_in).to(s_in.dtype)
            if self.override is not None and self.override[0] == j:
                s_in = self.override[1].to(s_in.dtype)
            ins.append(s_in)
            final = j == c - 1 and (ans_tgt is not None or ans_len is not None)
            ctx = past_R[0][0].shape[2]; hs, past_s = self.lp.fwd(s_in[:, None], past_R, R); mstep(meter, 1, ctx, 0, depth(R)); hslot = hs[:, 0]
            if step_tgt is not None:
                tgt = torch.cat([step_tgt[j], ans_tgt], 1) if (final and ans_tgt is not None) else step_tgt[j]
                ce = self.fork_tf_l(hslot, past_s, tgt, R)
                if U:
                    out_U, past_U = br[U]; hu, past_u = self.lp.fwd(out_U[:, -1][:, None], past_U, U)
                    ce = (ce + (R - 1) * self.fork_tf_l(hu[:, 0], past_u, tgt, U)) / R
                losses.append(ce)
            if final and ans_len is not None:
                d = self.fork_greedy_l(hslot, past_s, step_len[j] + ans_len, R, meter)
                decs.append(d[:, :step_len[j]]); ans = d[:, step_len[j]:]
            elif step_len is not None and (decode == "all" or j == c - 1):
                decs.append(self.fork_greedy_l(hslot, past_s, step_len[j], R, meter if (snap and j == c - 1) else meter_x))
            if j + 1 < c:
                x = torch.cat([g.embed(prefix), s_in[:, None].to(g.embed(prefix).dtype), g.embed(self.digtok[ops[:, j + 1]])[:, None]], 1)
        loss = torch.stack(losses).mean() if losses else None
        return loss, decs, ans, torch.stack(ins, 1)


def to_num(toks):
    d = {int(t): i for i, t in enumerate(DIGIT_IDS)}
    try:
        return int("".join(str(d[int(t)]) for t in toks[::-1]))
    except KeyError:
        return None


@torch.no_grad()
def donor(m, dev, amp, n=2000, bs=200, seed=11401, same_a=False):
    """§104 single patch of the carried latent at slot j (1..3) of the 4-op chunk; scored on the final answer."""
    from boxes.donor100 import problems, steps_of  # noqa: F401  (the dataset format, token-exact against the test file)
    rng = np.random.default_rng(seed); A = problems(rng, n); Bp = problems(rng, n, a=A["a"] if same_a else None)
    T = lambda x: torch.tensor(x, device=dev)  # noqa: E731
    pre = lambda P, s: T(np.concatenate([DIGIT_IDS[P["A"][s]], np.full((len(P["A"][s]), 1), TOK["*"])], 1))  # noqa: E731
    sl_ = [A["steps"][j].shape[1] for j in range(4)]; la = A["ans"].shape[1]; res = {}
    aA, aB = A["a"].astype(object), Bp["a"].astype(object); bA, bB = A["B"].astype(object), Bp["B"].astype(object)
    PA = aA * A["b"].astype(object)
    for j in range(3):                                                      # 0-based: slot j+1's carried latent
        got = []
        for s0 in range(0, n, bs):
            s = slice(s0, min(n, s0 + bs))
            with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
                _, _, _, insB = m.chunk(pre(Bp, s), None, T(Bp["B"][s]))
                m.override = (j, insB[:, j]); _, _, an, _ = m.chunk(pre(A, s), None, T(A["B"][s]), step_len=sl_, ans_len=la); m.override = None
            ta = an.cpu().numpy(); ok = (ta[:, :len(HASH)] == np.array(HASH)).all(1)
            got += [to_num(r[len(HASH):]) if o else None for r, o in zip(ta, ok)]
        SB = sum(aB * bB[:, t] * 10 ** t for t in range(j + 1)); PD = SB + sum(aA * bA[:, t] * 10 ** t for t in range(j + 1, 4))
        res[f"slot{j + 1}"] = {"donor_consistent": float(np.mean([g_ is not None and g_ == v for g_, v in zip(got, PD)])),
                               "recipient_consistent": float(np.mean([g_ is not None and g_ == v for g_, v in zip(got, PA)])),
                               "unparsed": float(np.mean([g_ is None for g_ in got])), "chance_overlap": float(np.mean(PD == PA))}
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--arm", required=True, choices=["loopr", "loop1"]); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--steps", type=int, default=24000); ap.add_argument("--bs", type=int, default=64); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--r_max", type=int, default=8); ap.add_argument("--test_r", type=int, default=None, help="default: r_max (loopr), 1 (loop1)")
    ap.add_argument("--acc_r", default="1,2,4,8,12,16"); ap.add_argument("--log_every", type=int, default=1000); ap.add_argument("--eval_bs", type=int, default=250)
    ap.add_argument("--final_eval_bs", type=int, default=100); ap.add_argument("--donor_n", type=int, default=2000)
    ap.add_argument("--val_n", type=int, default=None, help="smoke only"); ap.add_argument("--test_n", type=int, default=None, help="smoke only")
    ap.add_argument("--train_limit", type=int, default=0, help="smoke only"); ap.add_argument("--sched_scale", type=float, default=1.0, help="smoke only")
    ap.add_argument("--train_dir", default=os.path.join(ROOT, "bench", "data", "mult_train"))
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--out", required=True)
    a = ap.parse_args(); dev = a.device; torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); t_start = time.time()
    amp = torch.bfloat16 if dev.startswith("cuda") and torch.cuda.get_device_capability(dev)[0] >= 8 else None
    Dtr = load(os.path.join(a.train_dir, "4x4_train.txt"), limit=a.train_limit or None)
    Dva = load(os.path.join(a.train_dir, "4x4_valid.txt"), limit=a.val_n)
    Dte = load(os.path.join(ROOT, "bench", "data", "mult_4x4_test.txt"), limit=a.test_n)
    sc = a.sched_scale; elig = int(ARMS["markov"] * sc); cfg = lambda s: config("markov", s / sc)  # noqa: E731
    test_r = a.test_r or (a.r_max if a.arm == "loopr" else 1)
    os.makedirs(a.out, exist_ok=True); json.dump({**vars(a), "eligible_from": elig, "test_r_used": test_r}, open(os.path.join(a.out, "args.json"), "w", newline="\n"), indent=1)
    m = Loop114(s5.load_gpt2(dev)).to(dev); opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.01)
    log = open(os.path.join(a.out, "log.jsonl"), "w", newline="\n"); t0 = time.time(); hist = []; curve = []; ratios = []; flops = 0.0
    stop_step, hit = a.steps, None; base = {}
    for step in range(1, a.steps + 1):
        for gr in opt.param_groups:
            gr["lr"] = a.lr * min(1, step / 100) * 0.5 * (1 + math.cos(math.pi * step / a.steps))
        _, L, _ = cfg(step)
        R = int(rng.integers(1, a.r_max + 1)) if a.arm == "loopr" else 1; U = int(rng.integers(1, R)) if R > 1 else None
        m.r, m.u = R, U
        prefix, start, ops, st, ans = latent_batch(Dtr, rng, a.bs, dev, L)
        with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
            loss, _, _, _ = m.chunk(prefix, start, ops, step_tgt=st, ans_tgt=ans)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); hist.append(loss.item())
        if L not in base:                                                   # FLOPs: a 12-layer forward of this chunk shape, scaled by depth
            mt = MeterR(m.g); m.eval(); m.r, m.u = 1, None
            with torch.no_grad():
                m.chunk(prefix[:1], None if start is None else start[:1], ops[:1], step_tgt=[x[:1] for x in st], ans_tgt=None if ans is None else ans[:1], meter=mt)
            base[L] = mt.flops + 2 * 768 * 50257 * sum(x.shape[1] for x in st); m.train()
        flops += 3 * base[L] * a.bs * (depth(R) + (depth(U) if U else 0)) / 12
        if step % 100 == 0 or step % a.log_every == 0:
            rec = {"step": step, "loss": float(np.mean(hist)), "sec": round(time.time() - t0), "k_train": L, "r": R}; hist = []
            if L >= 2:                                                      # model-relative null: the same batch with carried latents zeroed
                m.eval(); m.r, m.u = R, None
                with torch.no_grad(), torch.autocast("cuda", dtype=amp, enabled=amp is not None):
                    l1, _, _, _ = m.chunk(prefix, start, ops, step_tgt=st, ans_tgt=ans)
                    l0, _, _, _ = m.chunk(prefix, start, ops, step_tgt=st, ans_tgt=ans, ablate="zero")
                m.train(); rec["loss_eval"] = float(l1); rec["null"] = float(l0); rec["ratio"] = float(l1) / max(1e-9, float(l0))
                ratios.append((step, rec["ratio"]))
            if step % a.log_every == 0:
                m.eval(); m.r, m.u = test_r, None; per, ok, _ = gen_rollout(m, Dva, dev, amp, a.eval_bs); m.train()
                e = {"step": step, "per_slot": per.mean(0).tolist(), "val_acc": float(ok.mean()), "k_train": L, "eligible": step >= elig}
                curve.append(e); rec.update({k: e[k] for k in ("per_slot", "val_acc", "eligible")})
                if e["eligible"] and min(e["per_slot"]) >= 0.95 and e["val_acc"] >= 0.95:
                    hit = step
            log.write(json.dumps(rec) + "\n"); log.flush(); print(rec, flush=True)
            if hit:
                stop_step = step; break
    wall = round(time.time() - t_start); train_sec = round(time.time() - t0)
    torch.save(m.state_dict(), os.path.join(a.out, "model.pt"))
    del opt; m.zero_grad(set_to_none=True); m.eval()
    if dev.startswith("cuda"):
        torch.cuda.empty_cache()
    m.r, m.u = test_r, None; bs = a.final_eval_bs
    meter = MeterR(m.g); per, ok, _ = rollout_safe(m, Dte, dev, amp, bs, meter=meter); mu = gen_mean_inputs(m, Dtr, dev, amp)
    k2 = 2000 * sc; win = [q for s_, q in ratios if k2 < s_ <= k2 + 1000 * sc]
    res = {"arm": a.arm, "task": "4x4", "seed": a.seed, "test_r": test_r, "gpu": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu",
           "eval_dtype": str(amp), "eligible_from": elig, "steps_to_threshold": hit, "stopped_step": stop_step, "curve": curve,
           "test_acc": float(ok.mean()), "per_slot_decode_acc": per.mean(0).tolist(), "removal": {},
           "compute": {**meter.as_dict(), "layer_depth": depth(test_r)},
           "training": {"flops": flops, "wall_sec": wall, "train_sec": train_sec,
                        "flops_convention": "sum over updates of a 12-layer Meter forward of the chunk shape x (depth(r) + depth(u)) / 12 x 3 x bs (an upper bound: the trunk is shared)"},
           "ratio_per_1k": [[s_, q] for s_, q in ratios if s_ % 1000 == 0], "early_escape": bool(win) and min(win) < 0.9,
           "early_window_min_ratio": min(win) if win else None, "escape_step": next((s_ for s_, q in ratios if q < 0.9), None),
           "null_convention": "loss with every carried latent zeroed on the same batch (no grad), test-mode r = the batch's r"}
    for ab in ("zero", "mean"):
        _, o_, _ = rollout_safe(m, Dte, dev, amp, bs, ablate=ab, mean_in=mu); res["removal"][ab] = {"answer_acc": float(o_.mean())}
    res["donor"] = donor(m, dev, amp, n=a.donor_n, bs=bs); res["donor_same_a"] = donor(m, dev, amp, n=a.donor_n, bs=bs, same_a=True)
    res["acc_by_r"] = {}
    for r in [int(x) for x in a.acc_r.split(",")]:
        m.r = r; mt = MeterR(m.g); p_, o_, _ = rollout_safe(m, Dte, dev, amp, bs, meter=mt)
        res["acc_by_r"][str(r)] = {"test_acc": float(o_.mean()), "per_slot": p_.mean(0).tolist(), "flops": mt.flops, "layer_depth": depth(r)}
    m.r = test_r
    json.dump(res, open(os.path.join(a.out, "eval.json"), "w", newline="\n"), indent=1)
    print("EVAL", json.dumps({k: res[k] for k in ("steps_to_threshold", "stopped_step", "test_acc")}), flush=True); print("DONE", a.out, flush=True)


if __name__ == "__main__":
    main()
