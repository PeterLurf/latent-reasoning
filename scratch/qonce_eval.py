"""§208 QUESTION-ONCE eval-only analyses (QONCE draft 1c2dc4d3 §5 / §12.3 / §12.4 G6; note 208-1: a NEW file, no module edit until §208 lands).
Replay machinery (fp32): DM.write_seq is wrapped (not edited) to record each think pass's write inputs (h6, hw, mask, pass-start S). From them:
  incr_s   = the pass's additive increment, sum_tok phi(k) v^T (FAST_ADD's masked einsum)              -> S_add(s) = S_add(s-1) + incr_s
  (P_s, C_s) = the pass's delta map S_del -> S_del P_s + C_s, from DM.delta_chunk run on 0 and on I  -> exact (affine in the start state)
G6 (gate) on a model dir: per-item maps reproduce S_K; a batch of one repeated item has mean map == item map; transplant t = 1 == clean(r) bitwise;
transplant r = d == clean(d); SWAP(t >= 1) == clean(pre) bitwise (P-Q2).        python scratch/qonce_eval.py <run dir> g6 [--device cuda:1] [--n 64]"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("BOXES_NB", "5")
ROOT = "D:/Latent"; sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import boxes.deltamem as DM  # noqa: E402
import boxes.gsm8k_data as G  # noqa: E402
import boxes.gsm8k_factor as GF  # noqa: E402
from boxes import s5  # noqa: E402
from boxes.gsm8k_stepaux import Fwd, ans_ok, greedy_after  # noqa: E402
from boxes.prosqa168s2 import ANS, ENC, EOS, EOT, ProsMem  # noqa: E402

DH = DM.DH


def build(run, dev):
    a_ = json.load(open(os.path.join(run, "args.json"))); g = s5.load_gpt2(dev)
    new = (a_.get("aux_target") or "string") != "string" or bool(a_.get("aux_nopass")) or bool(a_.get("add_norm"))
    m = (GF.FactorMem(g, a_.get("add_norm") or None, bool(a_.get("aux_nopass")), a_.get("aux_target") == "factor", a_.get("seed", 1)) if new else ProsMem(g)).to(dev)
    m.load_state_dict(torch.load(os.path.join(run, "model.pt"), map_location=dev, weights_only=False)); m.eval()
    meta = json.load(open(os.path.join(G.AUG, "s200_test_meta.json"), encoding="utf-8"))
    if a_.get("ans_bottleneck"):
        m.set_ab_qo(True, bool(a_.get("question_once")), [EOT] + ENC.encode(meta["filler"][a_.get("qo_filler", "k16")]))
    DM.FAST_ADD = bool(a_.get("fast_add")); DM.CHUNK_DELTA = int(a_.get("chunk_delta", 0) or 0)
    return g, m, a_


class Recorder:
    """wraps DM.write_seq while active: records (h6, hw, S_start, mask) per call (one call = one think pass for one batch)."""
    def __init__(self):
        self.rec = None; self.orig = DM.write_seq

    def __enter__(self):
        self.rec = []; orig = self.orig

        def w(mem, h6, hw, S, mask=None):
            self.rec.append((h6.detach(), hw.detach(), S.detach(), mask)); return orig(mem, h6, hw, S, mask)
        DM.write_seq = w; return self

    def __exit__(self, *e):
        DM.write_seq = self.orig


def write_terms(mem, h6, hw, mask):
    """write_seq's k / v / alpha / beta for a recorded pass (the same code, fp32)."""
    H_ADD, H_DEL = mem.h_add, mem.h_del
    with torch.autocast("cuda", enabled=False):
        B, T, D = h6.shape; a6 = mem.ln(h6.float()); aw = mem.ln_w(hw.float()); gm = mem.gamma
        u = (1 - gm) * a6 + gm * aw; _, k, v = mem.qkv(u).split(D, -1); k, v = (t_.view(B, T, H_ADD + H_DEL, DH) for t_ in (k, v))
        g_ = mem.gate(u); al = torch.sigmoid(g_[..., :H_DEL]); be = 2 * torch.sigmoid(g_[..., H_DEL:])
        pk = F.elu(k[:, :, :H_ADD]) + 1
        if mask is not None:
            pk = pk * mask[:, :, None, None].to(pk.dtype)
    return pk, v[:, :, :H_ADD], F.normalize(k[:, :, H_ADD:], dim=-1), v[:, :, H_ADD:], al, be


@torch.no_grad()
def pass_maps(mem, rec):
    """-> incr (B, H_ADD, DH, DH), P (B, H_DEL, DH, DH), C (B, H_DEL, DH, DH) for one recorded pass."""
    h6, hw, S, mask = rec; pk, va, kd, vd, al, be = write_terms(mem, h6, hw, mask); B = h6.shape[0]
    incr = torch.einsum("bthk,bthv->bhkv", pk, va)
    Z = torch.zeros(B, mem.h_del, DH, DH, device=h6.device); I = torch.eye(DH, device=h6.device).expand(B, mem.h_del, DH, DH).contiguous()
    run = (lambda S0: DM.delta_chunk(S0, kd, vd, al, be, mask, DM.CHUNK_DELTA)) if DM.CHUNK_DELTA else None
    assert run is not None, "pass_maps: CHUNK_DELTA runs only (the §208 cell's setting)"
    C = run(Z); P = run(I) - C
    return incr, P, C


def split(mem, S):
    return S[:, :mem.h_add].float(), S[:, mem.h_add:mem.h_add + mem.h_del].float()


def compose(maps):
    """[(incr, P, C)] in pass order -> (sum incr, P_total, C_total) with S_del -> ((S P1 + C1) P2 + C2) ..."""
    inc, P, C = maps[0]
    for i_, P_, C_ in maps[1:]:
        inc = inc + i_; P = P @ P_; C = C @ P_ + C_
    return inc, P, C


@torch.no_grad()
def think_record(m, its, K, dev, cache=True):
    """clean think of its with recording -> (S_K, [S after pass t for t = 0..K], [per-pass maps])."""
    x, msk = m.think_inputs(its, dev)
    with Recorder() as R:
        SK = m.think(m.mem.zero_state(len(its), dev), x, msk, K, cache=cache)
    states = [r[2] for r in R.rec] + [SK]; maps = [pass_maps(m.mem, r) for r in R.rec]
    return SK, states, maps


@torch.no_grad()
def live_from(m, S, its, t0, K, dev):
    """passes t0+1..K run live from state S (pass_inputs of its: the filler under QO)."""
    x, msk = m.think_inputs(its, dev); hl = m.lower(x) if True else None; pin = m.pass_inputs(x, msk, hl)
    for t in range(t0 + 1, K + 1):
        S = m.write(S, *pin(t))
    return S


def answers(g, m, S, n, dev, ac, bs=64):
    return greedy_after(Fwd(g, m.mem, S), [[EOT, ANS]] * n, dev, ac, bs)


def rel(a, b):
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-30))


def g6(run, dev, n):
    T0 = time.time(); g, m, A = build(run, dev); K = int(A["think_k"]); ac = lambda: torch.autocast("cuda", enabled=False)  # noqa: E731
    R, _ = G.load_aug("test", filter_bad=False); its = [G.encode(r, ENC, EOS) for r in R[:n]]; out = {"run": run, "n": len(its), "K": K, "dtype": "fp32", "device": dev}
    SK, st, maps = think_record(m, its, K, dev); Sa_K, Sd_K = split(m.mem, SK)
    ident = {}
    for k in (1, 2, 3, 4, 6, 8):
        inc, P, C = compose(maps[k:]); Sa_k, Sd_k = split(m.mem, st[k])
        ident[str(k)] = {"add_rel": rel(Sa_k + inc, Sa_K), "del_rel": rel(Sd_k @ P + C, Sd_K)}
    out["per_item_maps_reproduce_SK"] = ident; ok1 = all(v["add_rel"] <= 1e-5 and v["del_rel"] <= 1e-5 for v in ident.values())
    rep = [its[0]] * 8; SKr, str_, mr = think_record(m, rep, K, dev); inc, P, C = compose(mr[2:]); Sa2, Sd2 = split(m.mem, str_[2])
    mean_rel = {"add": rel(Sa2 + inc.mean(0, keepdim=True), split(m.mem, SKr)[0]), "del": rel(Sd2 @ P.mean(0, keepdim=True) + C.mean(0, keepdim=True), split(m.mem, SKr)[1]),
                "maps_identical_across_rows": bool(torch.equal(P[0], P[-1]) and torch.equal(C[0], C[-1]))}
    out["repeated_item_mean_map_eq_item_map"] = mean_rel; ok2 = mean_rel["add"] <= 1e-5 and mean_rel["del"] <= 1e-5
    # transplant identities and SWAP
    r_, d_ = its[:n // 2], its[n // 2:n // 2 * 2]; SKr_, str_r, mr_ = think_record(m, r_, K, dev); SKd_, str_d, md_ = think_record(m, d_, K, dev)
    Pr = answers(g, m, SKr_, len(r_), dev, ac); Pd = answers(g, m, SKd_, len(d_), dev, ac)

    def hybrid(t, rec_states, don_maps):
        Sa1, Sd1 = split(m.mem, rec_states[1])
        if t == 1:
            return rec_states[1].clone()
        inc, P, C = compose(don_maps[1:t]); return torch.cat([Sa1 + inc, Sd1 @ P + C], 1)
    H1 = live_from(m, hybrid(1, str_r, md_), r_, 1, K, dev)
    Hrr = live_from(m, hybrid(3, str_d, md_), d_, 3, K, dev)
    out["transplant_t1_eq_clean_r"] = {"state_bitwise": bool(torch.equal(H1, SKr_)), "answers_equal": answers(g, m, H1, len(r_), dev, ac) == Pr}
    out["transplant_r_eq_d_eq_clean_d"] = {"state_rel": rel(Hrr, SKd_), "answers_equal": answers(g, m, Hrr, len(d_), dev, ac) == Pd}
    sw = {}
    for t in (1, 2, 5):                                                      # SWAP(t; r -> d): passes 1..t on r, then d's pass inputs (the filler under QO)
        x, msk = m.think_inputs(r_, dev)
        with torch.no_grad():
            S = m.think(m.mem.zero_state(len(r_), dev), x, msk, t, cache=True)
        S = live_from(m, S, d_, t, K, dev); sw[str(t)] = bool(torch.equal(S, SKr_))
    out["swap_eq_clean_pre_bitwise"] = sw
    ok3 = out["transplant_t1_eq_clean_r"]["state_bitwise"] and out["transplant_t1_eq_clean_r"]["answers_equal"] and out["transplant_r_eq_d_eq_clean_d"]["state_rel"] <= 1e-5 \
        and all(sw.values())
    out["pass"] = bool(ok1 and ok2 and ok3); out["sec"] = round(time.time() - T0)
    os.makedirs("scratch/gate_s208", exist_ok=True); json.dump(out, open("scratch/gate_s208/g6.json", "w", newline="\n"), indent=1); print(json.dumps(out, indent=1)); return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("what"); ap.add_argument("--device", default="cuda:1"); ap.add_argument("--n", type=int, default=64)
    a = ap.parse_args(); torch.set_num_threads(3)
    if a.what == "g6":
        g6(a.run, a.device, a.n)
