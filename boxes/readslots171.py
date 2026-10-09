"""THEORY §171 READ-SLOTS (registration 2026-10-09 01:54 EDT, note 171-1; scorer theory/score_s171.py): the input is encoded ONCE, then K think passes in
which M learned slot tokens read it and write §155 R's DeltaMem-R (6 + 6 heads after block 6, top writes, gamma 2k -> 6k, pass-start reads, W_o fade-in,
g_add calibrated at step 0), the ONLY cross-pass state.
    encode       = GPT-2 over [EOT, " a_1 .. a_4 * b_1 .. b_n"] (LSB first) with NO memory read; its per-layer KV (P0 = na + nb + 2 positions) is kept for
                   the item. It is GPT-2 itself and trained: gradients from every slot pass and from the decoder flow into it.
    think pass t = M slot embeddings (new parameters, wte(" .") + N(0, 0.02^2)) at positions P0 .. P0 + M - 1; every layer attends to the input KV and
                   causally to the slots; the read on the pass-start state entering block 6; §155's top writes from the slot positions. The slot KV is
                   discarded after the pass (each pass builds a fresh slot cache; the input KV is never extended).
    decoder      = [" ="] + the answer digits after the input KV (positions P0 ..): reads the memory, writes nothing, attends to no slot position
                   (teacher-forced in training, greedy at test). K = 0 is DIRECT by construction (the input, " =", an empty memory).
                   --blind_decoder (note 171-1 BLIND): the decoder is [EOT, " ="] + digits with the memory and NO input KV.
Training: FINAL only, §161's 4 x n mix (n ~ U{2..5} per batch; gen_operands minus the 4x4 bench test, both orders), 165-1's curriculum, 48k, AdamW 1e-4
wd 0.01, warm-up 100 + cosine, clip 1.0; ckpts / resume; --auto_micro (4x5 training + the 4x16 eval at K' = 4K). As boxes/inject169.py otherwise.
eval.json: test_acc / acc_by_column (4x4 bench, K' = K), ext_4x6 / 4x7 / 4x8 / 4x16 (§161's seeds), compute / compute_4x8 (Meter), s171 {acc_by_kprime
{0, 1, K/2, K, 2K, 4K}, ext_4x8_by_kprime {K, 2K, 4K}, acc_by_size_kprime, skip_pass {t: acc}, read_map, read_entropy, read_order_rho, compute_4x8,
mem_norm_by_pass, spikes, n_nonfinite}.
  read_map {4x4 / 4x8: {t: {eot, a_i, star, b_j, slots}}}: the slots' attention mass per input position at pass t, mean over heads, blocks 7-12, the M slots
      and 200 test items ("slots" = the mass on slot positions).
  read_entropy {4x4 / 4x8: {t: nats}}: per head / block (7-12) / slot / item, the entropy of the attention renormalised over the input positions; the mean.
  read_order_rho {4x4 / 4x8: rho | None}: Spearman rho (average ranks) between the pass index t and the pass's peak answer column c (1-based), where column
      c's mass = the MEAN over its digit pairs (a_i, b_j), i + j = c - 1 (0-based i, j), of mass(a_i) + mass(b_j) (a mean, so middle columns' larger pair
      counts do not favour them). None when the peak column is constant over passes. peak_column_by_pass is kept beside it.
    python -m boxes.readslots171 --think_k 8 --seed 1 --out boxes/runs_fsq/s171_k8_4x4_s1"""
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
import torch.nn.functional as Fn  # noqa: E402
from torch.utils.checkpoint import checkpoint  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from boxes import s5  # noqa: E402
from boxes import core as bc  # noqa: E402
import boxes.deltamem as DM  # noqa: E402
from boxes.fsq120 import EOT, make_opt, lr_at, driver_version, env_info  # noqa: E402
from boxes.multcol import DTOK, gen_operands, load_bench  # noqa: E402
from boxes.deltas5 import fwd_read, write_grad_norm  # noqa: E402
from boxes.fixedchunk import DT, STAR, EQ, PAD, product_digits, acc_report, draw, cmax, amp_ctx  # noqa: E402

CAP_FROM = 6                                                                # read_map / read_entropy: blocks 7-12 (0-based 6..11)


def inp_ids(A, B):
    """the encoded input: [EOT, a digits, " *", b digits] -> (N, na + nb + 2)."""
    return np.concatenate([np.full((len(A), 1), EOT), DT[A], np.full((len(A), 1), STAR), DT[B]], 1)


def run(g, x, past=None, mem=None, S=None, write=False, cap=None):
    """GPT-2 over x (B, T, D) placed after `past` (per layer (k, v), length P0; None = from position 0, causal). mem / S: the read on S entering block
    mem.layer; write: §155's top writes from these T positions -> S_new. Returns (hidden after ln_f, this call's per-layer (k, v) INCLUDING past, S_new).
    cap: list -> per block >= CAP_FROM the attention probabilities (B, H, T, P0 + T) (fp32, no grad)."""
    B, T, D = x.shape; hd = D // g.H; P0 = 0 if past is None else past[0][0].shape[2]; h = x + g.w("wpe.weight")[P0:P0 + T][None]; h6 = None; kv = []
    mask = None if past is None else torch.ones(T, P0 + T, dtype=torch.bool, device=x.device).tril(P0)
    for i in range(g.L):
        if mem is not None and i == mem.layer:
            h6 = h; h = DM.read_fixed(mem, h, S)
        pre = f"h.{i}."
        a = Fn.layer_norm(h, (D,), g.w(pre + "ln_1.weight"), g.w(pre + "ln_1.bias"))
        q, k, v = (a @ g.w(pre + "attn.c_attn.weight") + g.w(pre + "attn.c_attn.bias")).split(D, -1)
        q, k, v = (t.view(B, T, g.H, hd).transpose(1, 2) for t in (q, k, v))
        if past is not None:
            k = torch.cat([past[i][0], k], 2); v = torch.cat([past[i][1], v], 2)
        kv.append((k, v))
        o = Fn.scaled_dot_product_attention(q, k, v, is_causal=True) if mask is None else Fn.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        if cap is not None and i >= CAP_FROM:
            with torch.no_grad():
                sc_ = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(hd)
                if mask is not None:
                    sc_ = sc_.masked_fill(~mask, float("-inf"))
                cap.append(sc_.softmax(-1))
        h = h + o.transpose(1, 2).reshape(B, T, D) @ g.w(pre + "attn.c_proj.weight") + g.w(pre + "attn.c_proj.bias")
        a = Fn.layer_norm(h, (D,), g.w(pre + "ln_2.weight"), g.w(pre + "ln_2.bias"))
        mm = Fn.gelu(a @ g.w(pre + "mlp.c_fc.weight") + g.w(pre + "mlp.c_fc.bias"), approximate="tanh")
        h = h + mm @ g.w(pre + "mlp.c_proj.weight") + g.w(pre + "mlp.c_proj.bias")
    S_new = DM.write_seq(mem, h6, h, S) if write else None                  # top writes: h_w = the residual after block 11, pre ln_f
    return Fn.layer_norm(h, (D,), g.w("ln_f.weight"), g.w("ln_f.bias")), kv, S_new


class ReadSlots(nn.Module):
    def __init__(self, g, n_slots=8, layer=6, write="top", blind=False):
        super().__init__(); assert write == "top"; self.g = g; self.mem = DM.DeltaMem(g, layer, write); self.blind = blind
        with torch.no_grad():
            e = g.w("wte.weight")[PAD].float()
        self.slots = nn.Parameter(e[None].repeat(n_slots, 1) + 0.02 * torch.randn(n_slots, e.shape[0], device=e.device))

    def encode(self, A, B, dev):
        """the input's per-layer KV (no memory read); a tuple so no pass can extend it."""
        _, kv, _ = run(self.g, self.g.embed(torch.tensor(inp_ids(A, B), device=dev)))
        return tuple(kv)

    def slot_pass(self, S, kv, cap=None):
        P0 = kv[0][0].shape[2]; x = self.slots[None].expand(S.shape[0], -1, -1)
        _, kv2, S2 = run(self.g, x, kv, self.mem, S, write=True, cap=cap)
        assert kv2[0][0].shape[2] == P0 + self.slots.shape[0] and kv[0][0].shape[2] == P0   # the slot KV lives only in kv2, dropped here
        return S2

    def think(self, S, kv, K, skip=(), override=None, ckpt=False, cap=None):
        for t in range(1, K + 1):
            if t not in skip:
                if ckpt and torch.is_grad_enabled():
                    S = checkpoint(self.slot_pass, S, kv, use_reentrant=False)
                else:
                    S = self.slot_pass(S, kv, None if cap is None else cap.setdefault(t, []))
            if override is not None and override[0] == t:
                S = override[1]
        return S

    def dec_hidden(self, S, kv, toks):
        """toks (B, T) starting with " =" -> hidden at those positions; the read on S, no write; the input KV only (none if blind)."""
        if self.blind:
            x = torch.cat([torch.full((toks.shape[0], 1), EOT, device=toks.device), toks], 1)
            return fwd_read(self.g, self.g.embed(x), self.mem, S)[:, 1:]
        return run(self.g, self.g.embed(toks), kv, self.mem, S)[0]

    def dec_logits(self, S, kv, ans):
        toks = torch.cat([torch.full((ans.shape[0], 1), EQ, device=ans.device), ans[:, :-1]], 1)
        return self.g.logits(self.dec_hidden(S, kv, toks)).float()

    @torch.no_grad()
    def greedy_ans(self, S, kv, L):
        toks = torch.full((S.shape[0], 1), EQ, device=S.device); lut = {int(t): i for i, t in enumerate(DTOK)}
        for _ in range(L):
            nt = self.g.logits(self.dec_hidden(S, kv, toks)[:, -1]).float().argmax(-1); toks = torch.cat([toks, nt[:, None]], 1)
        return np.vectorize(lambda t: lut.get(int(t), -1))(toks[:, 1:].cpu().numpy())


@torch.no_grad()
def predict(m, A, B, K, dev, amp, bs, skip=()):
    N = len(A); L = A.shape[1] + B.shape[1]; out = np.full((N, L), -1)
    for i in range(0, N, bs):
        sl = slice(i, i + bs)
        with amp_ctx(amp):
            kv = m.encode(A[sl], B[sl], dev); S = m.think(m.mem.zero_state(len(A[sl]), dev), kv, K, skip=skip); out[sl] = m.greedy_ans(S, kv, L)
    return out


def compute_rs(m, na, nb, K):
    """Meter per item (§98 convention): the input encoded once (na + nb + 2 tokens, no rows); K slot passes of M tokens after the input KV; the decoder's
    " =" pass (1 row; blind: [EOT, " ="], 2 tokens from position 0) then na + nb - 1 KV-cached single-token passes (1 row each); plus the DeltaMem-R branch
    (reads at the slot and decoder tokens, writes at the slot tokens; as inject169.compute_inj)."""
    mt = bc.Meter(m.g); Tin = na + nb + 2; M = m.slots.shape[0]; mt.step(Tin, 0, 0); tok = 0
    for _ in range(K):
        mt.step(M, Tin, 0); tok += M
    if m.blind:
        mt.step(2, 0, 1); ctx = 2; tok_r = 2
    else:
        mt.step(1, Tin, 1); ctx = Tin + 1; tok_r = 1
    for j in range(na + nb - 1):
        mt.step(1, ctx + j, 1); tok_r += 1
    nb_ = sum(p_.numel() for n_, p_ in m.mem.named_parameters() if not n_.startswith("s_")); D2 = DM.DH ** 2
    fb = 2 * nb_ * (tok + tok_r) + (DM.H_ADD + DM.H_DEL) * 2 * D2 * (tok + tok_r) + (DM.H_ADD * 2 + DM.H_DEL * 8) * D2 * tok; d = mt.as_dict()
    return {"flops": d["flops"] + fb, "flops_trunk": d["flops"], "flops_branch": float(fb), "sequential_passes": d["seq_passes"], "lm_head_rows": na + nb,
            "tokens_processed": Tin + tok + tok_r, "think_passes": K, "n_slots": M, "blind_decoder": m.blind,
            "convention": "§98 (boxes.core.Meter); input encoded once; K slot passes after its KV; decoder ' =' + (na + nb - 1) KV-cached passes"}


def _spearman(x, y):
    def rk(v):
        v = np.asarray(v, float); o = np.argsort(v, kind="stable"); r = np.empty(len(v)); r[o] = np.arange(len(v), dtype=float)
        for u in np.unique(v):
            r[v == u] = r[v == u].mean()
        return r
    rx, ry = rk(x), rk(y)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


@torch.no_grad()
def read_stats(m, A, B, K, dev, amp, bs):
    """read_map / read_entropy / read_order_rho at one operand size (see the module docstring)."""
    na, nb = A.shape[1], B.shape[1]; P0 = na + nb + 2; names = ["eot"] + ["a_%d" % (i + 1) for i in range(na)] + ["star"] + ["b_%d" % (j + 1) for j in range(nb)]
    mass = {t: [] for t in range(1, K + 1)}; ent = {t: [] for t in range(1, K + 1)}
    for i in range(0, len(A), bs):
        sl = slice(i, i + bs); cap = {}
        with amp_ctx(amp):
            kv = m.encode(A[sl], B[sl], dev); m.think(m.mem.zero_state(len(A[sl]), dev), kv, K, cap=cap)
        for t in range(1, K + 1):
            p = torch.stack(cap[t], 1)                                      # (b, blocks, H, M, P0 + M)
            mass[t].append(p.mean(dim=(1, 2, 3)).cpu().numpy())            # (b, P0 + M)
            pin = p[..., :P0]; pin = pin / pin.sum(-1, keepdim=True).clamp_min(1e-12)
            ent[t].append((-(pin * pin.clamp_min(1e-30).log()).sum(-1)).mean(dim=(1, 2, 3)).cpu().numpy())
    rmap, rent, peak = {}, {}, []
    for t in range(1, K + 1):
        mv = np.concatenate(mass[t]).mean(0); rmap[str(t)] = {nm: float(mv[k]) for k, nm in enumerate(names)}; rmap[str(t)]["slots"] = float(mv[P0:].sum())
        rent[str(t)] = float(np.concatenate(ent[t]).mean())
        ma, mb = mv[1:1 + na], mv[2 + na:2 + na + nb]
        col = [np.mean([ma[i_] + mb[c - i_] for i_ in range(na) if 0 <= c - i_ < nb]) for c in range(na + nb - 1)]
        peak.append(int(np.argmax(col)) + 1)
    return rmap, rent, (_spearman(list(range(1, K + 1)), peak) if K >= 2 else None), peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--think_k", type=int, required=True); ap.add_argument("--n_slots", type=int, default=8); ap.add_argument("--blind_decoder", action="store_true")
    ap.add_argument("--shape_mix", default="2,3,4,5"); ap.add_argument("--steps", type=int, default=48000)
    ap.add_argument("--bs", type=int, default=64); ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--micro", type=int, default=1); ap.add_argument("--ckpt_passes", action="store_true", help="activation checkpointing around each slot pass")
    ap.add_argument("--mem_write", default="top", choices=["top"]); ap.add_argument("--write_mix_start", type=int, default=2000); ap.add_argument("--write_mix_end", type=int, default=6000)
    ap.add_argument("--ckpt_every", type=int, default=2000); ap.add_argument("--resume", default=None); ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--eval_bs", type=int, default=250); ap.add_argument("--test_n", type=int, default=1000); ap.add_argument("--ext_n", type=int, default=1000)
    ap.add_argument("--read_n", type=int, default=200); ap.add_argument("--auto_micro", action="store_true"); ap.add_argument("--mem_budget", type=float, default=21.0)
    ap.add_argument("--eval_at", type=int, default=0, help="also evaluate test_acc / ext_4x8 at this step -> eval_<k>k.json")
    ap.add_argument("--mem_check", action="store_true"); ap.add_argument("--sched_scale", type=float, default=1.0, help="smokes only"); ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--bench", default=os.path.join("bench", "data", "mult_4x4_test.txt")); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--out", required=True)
    a = ap.parse_args(); a.arm = "readslots"; os.makedirs(a.out, exist_ok=True); dev = a.device; sc = a.sched_scale; K = a.think_k
    SHM = [int(x) for x in a.shape_mix.split(",") if x.strip()]; torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    amp = None if (a.fp32 or not dev.startswith("cuda") or torch.cuda.get_device_capability(dev)[0] < 8) else torch.bfloat16
    teA, teB = load_bench(a.bench if os.path.isabs(a.bench) else os.path.join(ROOT, a.bench))
    EXCL = {(x.tobytes(), y.tobytes()) for x, y in zip(teA, teB)} | {(y.tobytes(), x.tobytes()) for x, y in zip(teA, teB)}; teA, teB = teA[:a.test_n], teB[:a.test_n]
    m = ReadSlots(s5.load_gpt2(dev), a.n_slots, 6, a.mem_write, a.blind_decoder).to(dev)

    def train_loss(A, B, c):
        kv = m.encode(A, B, dev); S = m.think(m.mem.zero_state(len(A), dev), kv, K, ckpt=a.ckpt_passes); P = product_digits(A, B); ans = torch.tensor(DT[P], device=dev)
        lg = m.dec_logits(S, kv, ans); ce = P.shape[1] if c >= 8 else min(c, P.shape[1])
        return Fn.cross_entropy(lg[:, :ce].reshape(-1, lg.shape[-1]), ans[:, :ce].reshape(-1))

    def mem_check(micro):
        torch.cuda.reset_peak_memory_stats(dev); r_ = np.random.default_rng(0); A, B = gen_operands(r_, a.bs // micro, 4), gen_operands(r_, a.bs // micro, max(SHM)); m.train()
        with amp_ctx(amp):
            l_ = train_loss(A, B, 8)
        l_.backward(); m.zero_grad(set_to_none=True); m.eval()
        predict(m, gen_operands(r_, a.eval_bs, 4), gen_operands(r_, a.eval_bs, 16), 4 * K, dev, amp, a.eval_bs); m.train()
        return torch.cuda.max_memory_allocated(dev) / 2 ** 30
    if a.mem_check or (a.auto_micro and dev.startswith("cuda") and not a.resume):
        mc = []
        while True:
            pk = mem_check(a.micro); mc.append({"micro": a.micro, "peak_gib": pk}); print("MEM", a.micro, pk, flush=True)
            if pk <= a.mem_budget or a.micro >= a.bs or a.mem_check:
                break
            a.micro *= 2
        json.dump({"checks": mc, "micro": a.micro, "peak_gib": mc[-1]["peak_gib"], "train_shape": "4x%d" % max(SHM), "eval_shape": "4x16 at K' = %d" % (4 * K)},
                  open(os.path.join(a.out, "mem_check.json"), "w", newline="\n"), indent=1)
        if a.mem_check:
            return
        assert mc[-1]["peak_gib"] <= a.mem_budget
    info = {**vars(a), "train_gpu": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu", "train_driver": driver_version(), **env_info(),
            "encode_input": "[EOT, a * b] once, no memory read (trained)", "think_input": "%d learned slots after the input KV; fresh slot cache per pass" % a.n_slots,
            "decoder_input": "[EOT, ' ='] + answer digits, no input KV (read-only)" if a.blind_decoder else "[' ='] + answer digits after the input KV (read-only)"}
    json.dump(info, open(os.path.join(a.out, ("args_resume_%d.json" % int(time.time())) if a.resume else "args.json"), "w", newline="\n"), indent=1)
    opt = make_opt(m, a.lr); step0 = 1; n_nonfinite = 0; wg_hist = []; wwin = []; wlast = []; hist = []
    log = open(os.path.join(a.out, "log.jsonl"), "a" if a.resume else "w", newline="\n"); t0 = time.time()
    if a.resume:
        ck = torch.load(a.resume, map_location=dev, weights_only=False); m.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); rng.bit_generator.state = ck["np_rng"]
        torch.set_rng_state(ck["torch_rng"].cpu()); step0 = ck["step"] + 1; n_nonfinite = ck["n_nonfinite"]; wg_hist = ck["wg_hist"]; t0 = time.time() - ck["elapsed"]
        if dev.startswith("cuda") and ck.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(ck["cuda_rng"].cpu(), device=dev)
        log.write(json.dumps({"step": ck["step"], "resumed": a.resume}) + "\n"); log.flush()
    elif K > 0:                                                             # note 147-2: g_add = median ||read_del|| / ||read_add|| at the think hand-offs
        r_ = np.random.default_rng(17101); A0, B0 = gen_operands(r_, 128, 4), gen_operands(r_, 128, 4); m.mem.norms = []
        with torch.no_grad(), amp_ctx(amp):
            m.think(m.mem.zero_state(128, dev), m.encode(A0, B0, dev), min(K, 4))
        nr = m.mem.norms[1:]; m.mem.norms = None; r0 = float(torch.stack([d_ / a_.clamp_min(1e-12) for d_, a_ in nr]).median()) if nr else 1.0
        m.mem.s_add.data.fill_(math.log(max(r0, 1e-12))); json.dump({"g_add_init": r0}, open(os.path.join(a.out, "s171_init.json"), "w", newline="\n"))
    for step in range(step0, a.steps + 1):
        for gr in opt.param_groups:
            gr["lr"] = lr_at(a.lr, gr, step, a.steps)
        m.train(); m.mem.fade.fill_(min(1.0, step / max(1, int(500 * sc))))
        m.mem.gamma.fill_(0.0 if step <= a.write_mix_start * sc else min(1.0, (step - a.write_mix_start * sc) / max(1.0, (a.write_mix_end - a.write_mix_start) * sc)))
        c = int(rng.integers(1, cmax(step / sc) + 1)); A, B = draw(rng, a.bs, SHM, EXCL); opt.zero_grad(); tot = 0.0; ch = a.bs // a.micro
        for k_ in range(a.micro):
            sl = slice(k_ * ch, (k_ + 1) * ch)
            with amp_ctx(amp):
                lk = train_loss(A[sl], B[sl], c)
            (lk / a.micro).backward(); tot += float(lk.item()) / a.micro
        wwin.append(write_grad_norm(m)); gn = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        if not (math.isfinite(tot) and bool(torch.isfinite(gn))):
            log.write(json.dumps({"step": step, "nonfinite": 1}) + "\n"); log.flush(); n_nonfinite += 1; opt.zero_grad()
        else:
            opt.step(); hist.append(tot)
        if a.eval_at and step == a.eval_at:
            m.eval(); P_ = product_digits(teA, teB); xa, xb = gen_operands(np.random.default_rng(14004), a.ext_n, 4), gen_operands(np.random.default_rng(14005), a.ext_n, 8)
            e_ = {"step": step, "test_acc": float((predict(m, teA, teB, K, dev, amp, a.eval_bs) == P_).all(1).mean()),
                  "ext_4x8": {"test_acc": float((predict(m, xa, xb, K, dev, amp, a.eval_bs) == product_digits(xa, xb)).all(1).mean()), "seeds": [14004, 14005]}}
            json.dump(e_, open(os.path.join(a.out, "eval_%dk.json" % (step // 1000)), "w", newline="\n"), indent=1); m.train()
        if step % a.log_every == 0 or step == a.steps:
            md_ = float(np.median(wlast)) if wlast else 0.0; mx_ = float(np.max(wwin)); late = step > a.write_mix_start * sc
            w_rec = {"step": step, "mean": float(np.mean(wwin)), "max": mx_, "median_prev_1k": md_, "spike_100x": bool(late and mx_ > 100 * md_ > 0),
                     "spike_100x_floor": bool(late and mx_ > 100 * md_ > 0 and mx_ > 1.0)}
            wlast.extend(wwin); del wlast[:-int(1000 * sc) or None]; wwin = []; wg_hist.append(w_rec)
            rec = {"step": step, "loss": float(np.mean(hist)) if hist else None, "c_max": cmax(step / sc), "sec": round(time.time() - t0), "g_add": float(torch.exp(m.mem.s_add)),
                   "g_del": float(torch.exp(m.mem.s_del)), "fade": float(m.mem.fade), "gamma": float(m.mem.gamma), "write_grad_norm": w_rec["mean"], "write_grad_spike_100x_floor": w_rec["spike_100x_floor"]}
            hist = []; log.write(json.dumps(rec) + "\n"); log.flush(); print(rec, flush=True)
        if a.ckpt_every and step % a.ckpt_every == 0 and step < a.steps:
            cp = os.path.join(a.out, "ckpt_last.pt")
            if os.path.exists(cp):
                os.replace(cp, os.path.join(a.out, "ckpt_prev.pt"))
            torch.save({"model": m.state_dict(), "opt": opt.state_dict(), "np_rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state(dev) if dev.startswith("cuda") else None, "step": step, "n_nonfinite": n_nonfinite, "wg_hist": wg_hist,
                        "elapsed": time.time() - t0}, cp)
    train_sec = round(time.time() - t0); torch.save(m.state_dict(), os.path.join(a.out, "model.pt")); m.eval(); t1 = time.time(); bs = a.eval_bs
    Pte = product_digits(teA, teB); res = acc_report(predict(m, teA, teB, K, dev, amp, bs), Pte)
    res.update({"arm": "readslots", "think_k": K, "n_slots": a.n_slots, "blind_decoder": a.blind_decoder, "seed": a.seed, "train_sec": train_sec, "n_nonfinite": n_nonfinite,
                "eval_dtype": str(amp), "micro": a.micro, "gpu": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu", "train_gpu": info["train_gpu"],
                "train_driver": info["train_driver"]})
    xs = {4: (teA, teB)}
    for nb_, s1_, s2_ in ((6, 14016, 14017), (7, 14018, 14019), (8, 14004, 14005), (16, 14006, 14007)):
        xA, xB = gen_operands(np.random.default_rng(s1_), a.ext_n, 4), gen_operands(np.random.default_rng(s2_), a.ext_n, nb_); xs[nb_] = (xA, xB)
        res["ext_4x%d" % nb_] = {**acc_report(predict(m, xA, xB, K, dev, amp, bs), product_digits(xA, xB)), "seeds": [s1_, s2_]}
    res["compute"] = compute_rs(m, 4, 4, K); res["compute_4x8"] = compute_rs(m, 4, 8, K)
    kps = sorted({0, 1, K // 2, K, 2 * K, 4 * K}); s = {"acc_by_kprime": {}, "ext_4x8_by_kprime": {}, "acc_by_size_kprime": {}, "skip_pass": {}, "write_grad_norm": wg_hist,
                                                         "write_grad_spikes_floor": [x_ for x_ in wg_hist if x_["spike_100x_floor"]], "n_nonfinite": n_nonfinite,
                                                         "gamma_final": float(m.mem.gamma), "g_add": float(torch.exp(m.mem.s_add)), "g_del": float(torch.exp(m.mem.s_del)),
                                                         "compute_4x8": res["compute_4x8"]}
    for nb_ in (4, 6, 8, 16):
        Pn = product_digits(*xs[nb_]); s["acc_by_size_kprime"]["4x%d" % nb_] = {str(kp): float((predict(m, *xs[nb_], kp, dev, amp, bs) == Pn).all(1).mean()) for kp in kps}
    s["acc_by_kprime"] = dict(s["acc_by_size_kprime"]["4x4"]); s["ext_4x8_by_kprime"] = {str(kp): s["acc_by_size_kprime"]["4x8"][str(kp)] for kp in sorted({K, 2 * K, 4 * K})}
    for t in (sorted({t for t in (1, K // 4, K // 2, 3 * K // 4, K) if t >= 1}) if K else []):
        s["skip_pass"][str(t)] = float((predict(m, teA, teB, K, dev, amp, bs, skip=(t,)) == Pte).all(1).mean())
    if K:
        s["read_map"], s["read_entropy"], s["read_order_rho"], s["peak_column_by_pass"] = {}, {}, {}, {}
        for nb_ in (4, 8):
            rA, rB = xs[nb_][0][:a.read_n], xs[nb_][1][:a.read_n]
            s["read_map"]["4x%d" % nb_], s["read_entropy"]["4x%d" % nb_], s["read_order_rho"]["4x%d" % nb_], s["peak_column_by_pass"]["4x%d" % nb_] = \
                read_stats(m, rA, rB, K, dev, amp, min(bs, 100))
        s["read_n"] = int(min(a.read_n, len(teA))); mn = {}
        with torch.no_grad():                                               # as §169: the memory's norms per pass up to 4K
            nA, nB = teA[:256], teB[:256]; S = m.mem.zero_state(len(nA), dev)
            with amp_ctx(amp):
                kv = m.encode(nA, nB, dev)
            for t in range(1, 4 * K + 1):
                with amp_ctx(amp):
                    S = m.slot_pass(S, kv)
                mn[str(t)] = {"additive": float(S[:, :DM.H_ADD].float().flatten(1).norm(dim=1).median()), "delta": float(S[:, DM.H_ADD:].float().flatten(1).norm(dim=1).median())}
        s["mem_norm_by_pass"] = mn
    res["s171"] = s; res["eval_sec"] = round(time.time() - t1)
    json.dump(res, open(os.path.join(a.out, "eval.json"), "w", newline="\n"), indent=1)
    for ck_ in ("ckpt_last.pt", "ckpt_prev.pt"):
        if os.path.exists(os.path.join(a.out, ck_)):
            os.remove(os.path.join(a.out, ck_))
    print("EVAL", json.dumps({"test_acc": res["test_acc"], "acc_by_kprime": s["acc_by_kprime"]}), flush=True); print("DONE", a.out, flush=True)


if __name__ == "__main__":
    main()
