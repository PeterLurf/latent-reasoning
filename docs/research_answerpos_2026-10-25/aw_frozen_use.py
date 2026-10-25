"""ANSWER-WRITE $0 gate G-AW1 FROZEN-USE (AW_v1 §2; REVIEW_AW fix 1), rule committed in AW_v1 before this script is run. Can the FROZEN askname trunk
compose with a value supplied through the block-6 read (A1), given only a fitted read-out W_o?
Model, built as the arm builds it (aw_gate): pretrained GPT-2, s233_askname_s1 loaded strict (sha logged), torch.manual_seed(1), InjectOW(g, 6, "top", 6, 6,
None) - DeltaMem copied from the loaded trunk; FAST_ADD. Everything frozen except mem.wo (weight + bias), mem.s_add, mem.s_del; gamma = 1, fade = 1.
S_1 = the AW pass-1 write: the single-token write (gamma = 1) of u1 = mem.ln_w(h_w) at ':' of x_1 = [EOT] + enc(" " + text + " " + QUESTION(u, S_1)) +
" The answer is:", computed once at init (W_o = 0, read == 0), no gradient into it.
  MAIN    decoder read at K' = 1: dec = [EOT] + enc(" " + text) + " The answer is:" (tok_ow's dec) with read_fixed(S_1) at block 6 -> CE on v_2's number
          token; d = 2 items. Blocks 0-5 of dec are cached per item (identical to fwd_read: the read enters block 6), so a step runs blocks 6-11.
  (a)     the MAIN-fitted W_o read with S := 0 (control: <= majority + 0.05).
  (b)     LP2's path, a separately fitted W_o: the aux prompt [EOT, ' Step', ' 1', ' :'] + R_step[1] teacher-forced, CE at v_1's value token
          (control: >= 0.80 - the harness can read S_1).
  desc    MAIN with a rank-32 W_o (A B, init A ~ N(0, 0.02), B = 0); SWAP: S_1 from a same-skeleton donor (OW.redraw, rng 23482) on the test items ->
          the share answering cf (v_2 recomputed from the donor's v_1: inject_ow.cf(rec, don, 1, 'a' / 'b')), own and donor-final. T1 is not run.
Items: d = 2, OW.Stream(23481, EXCL = build_sets()["excl"] + the aw_gate items' sigs, frames 0-5, topo, dag); 8,000; split by sig 75 / 12.5 / 12.5
(train 6,000 / val 1,000 / test 1,000; rng 0). Fit: Adam lr 1e-3, bs 64, <= 2,000 steps, val every 100, early stop (patience 5 checks), best-val weights.
Rule (AW_v1 §2 / REVIEW fix 1): PASS iff MAIN test v_2 acc >= 0.50 with (a) <= majority + 0.05 and (b) >= 0.80; FAIL iff MAIN <= majority + 0.10 with valid
controls (-> to the user before P-2); between -> GO; a control fails -> UNREADABLE (fix once, re-run once). Theory scores; Experiments writes, not reads.
    python -B docs/research_answerpos_2026-10-25/aw_frozen_use.py [--device cuda:0] [--smoke]     -> aw_frozen_use.json (smoke: N 400, 60 steps, no reading)"""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2"); os.environ.setdefault("MKL_NUM_THREADS", "2"); os.environ.setdefault("BOXES_NB", "5")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
import boxes.deltamem as DM  # noqa: E402
import boxes.inject_ow as IO  # noqa: E402
import boxes.ow_words as OW  # noqa: E402
from boxes.nested170 import EOT, NUM_IDS, pad  # noqa: E402

ENC = IO.ENC; SFX = IO.SFX; P = 11; SEED_ITEMS, SEED_DONOR = 23481, 23482
INIT = os.path.join(ROOT, "boxes", "runs_fsq", "s233_askname_s1", "model.pt")
TRAINABLE = ("mem.wo.weight", "mem.wo.bias", "mem.s_add", "mem.s_del")


def build(dev):
    g = IO.load_trunk(dev, {"scratch": ""}); sha = hashlib.sha256(open(INIT, "rb").read()).hexdigest()
    g.load_state_dict(torch.load(INIT, map_location=dev, weights_only=True), strict=True); g.eval(); torch.manual_seed(1)
    m = IO.InjectOW(g, 6, "top", 6, 6, None).to(dev); m.eval(); DM.FAST_ADD = True
    for n_, p_ in m.named_parameters():
        p_.requires_grad_(n_ in TRAINABLE)
    assert sorted(n_ for n_, p_ in m.named_parameters() if p_.requires_grad) == sorted(TRAINABLE)
    return m, g, sha


def items(n, excl):
    s = OW.Stream(SEED_ITEMS, excl, frames=OW.FRAME_TRAIN, **IO.OWKW); out = []
    while len(out) < n:
        it = s.draw(2); t = IO.tok_ow(it); sp = it["names"]["spine"]
        x1 = [EOT] + ENC.encode(" " + it["text"] + " " + OW.QUESTION.format(u=it["unit"], A=sp[1])) + SFX
        assert x1[:len(t["think"])] == t["think"] and t["dec"] == t["think"] + SFX
        out.append({"it": it, "tok": t, "x1": x1})
    return out


@torch.no_grad()
def write_s1(m, g, rows, dev, bs):
    """S_1 per item (fp32, dev): the gamma = 1 single-token write of u1 at ':' of x_1 into the zero state, at init (fade 0: the read adds exactly 0)."""
    mem = m.mem; out = []; mem.fade.fill_(0.0); mem.gamma.fill_(1.0)
    for i in range(0, len(rows), bs):
        ch = rows[i:i + bs]; B = len(ch); seqs = [r["x1"] for r in ch]; ids_, _ = pad(seqs); x = torch.tensor(ids_, device=dev); pos = torch.tensor([len(q) - 1 for q in seqs], device=dev)
        cap = {}; S0 = mem.zero_state(B, dev); DM.gpt_forward_mem_r(g, g.embed(x), mem, S0, cap=cap); ar = torch.arange(B, device=dev)
        assert float(cap["read"].abs().max()) == 0.0
        out.append(DM.write_seq(mem, cap["h6"][ar, pos][:, None], cap["hw"][ar, pos][:, None], S0, None))
    return torch.cat(out)


@torch.no_grad()
def lower(g, seqs, dev, bs):
    """blocks 0-5 of each sequence (right-padded per batch) -> list of (h6 (T_i, D), last index)."""
    out = []
    for i in range(0, len(seqs), bs):
        ch = seqs[i:i + bs]; ids_, _ = pad(ch); x = torch.tensor(ids_, device=dev); h = DM.gpt_lower(g, g.embed(x), 6)
        out += [(h[r, :len(q)].clone(), len(q) - 1) for r, q in enumerate(ch)]
    return out


def upper(m, g, H, S):
    """fwd_read from block 6 on: read_fixed(S) entering block 6, blocks 6-11, ln_f -> (B, T, D)."""
    D = H.shape[-1]; hd = D // g.H; h = DM.read_fixed(m.mem, H, S)
    for i in range(6, g.L):
        h = DM._block(g, i, h, hd)
    return F.layer_norm(h, (D,), g.w("ln_f.weight"), g.w("ln_f.bias"))


def batch(L, idx, dev):
    T = max(L[j][0].shape[0] for j in idx); D = L[idx[0]][0].shape[1]; H = torch.zeros(len(idx), T, D, device=dev)
    for r, j in enumerate(idx):
        H[r, :L[j][0].shape[0]] = L[j][0]
    return H, torch.tensor([L[j][1] for j in idx], device=dev)


def logits_at(m, g, L, S, idx, dev, pos_override=None):
    H, last = batch(L, idx, dev); h = upper(m, g, H, S[idx]); pos = last if pos_override is None else pos_override
    return g.logits(h[torch.arange(len(idx), device=dev), pos]).float()


class LowRank(torch.nn.Module):
    """W_o = A B (rank r) + bias, swapped in for mem.wo during the descriptive rank fit (A ~ N(0, 0.02), B = 0: W_o = 0 at the start, as the full fit)."""
    def __init__(self, d_out, d_in, r, dev):
        super().__init__(); gen = torch.Generator(device=dev).manual_seed(7)
        self.A = torch.nn.Parameter(torch.randn(d_out, r, device=dev, generator=gen) * 0.02); self.B = torch.nn.Parameter(torch.zeros(r, d_in, device=dev))
        self.bias = torch.nn.Parameter(torch.zeros(d_out, device=dev))

    def forward(self, x):
        return F.linear(x, self.A @ self.B, self.bias)


def fit(m, g, L, S, y_tok, tr, va, dev, steps, bs, s_init, pos=None):
    """Adam on the trainable set of mem.wo (+ s_add, s_del reset to their init values), val acc every 100 steps (argmax over the 11 number tokens),
    early stop (patience 5 checks), best-val state restored. mem.wo may be a LowRank (descriptive)."""
    mem = m.mem; num = torch.tensor(NUM_IDS, device=dev)
    with torch.no_grad():
        if isinstance(mem.wo, torch.nn.Linear):
            mem.wo.weight.zero_(); mem.wo.bias.zero_()
        mem.s_add.copy_(s_init[0]); mem.s_del.copy_(s_init[1]); mem.fade.fill_(1.0); mem.gamma.fill_(1.0)
    params = [p_ for p_ in m.parameters() if p_.requires_grad]
    opt = torch.optim.Adam(params, lr=1e-3); rng = np.random.default_rng(0); best = (-1.0, None, 0); bad = 0; hist = []

    def lg_of(idx):
        return logits_at(m, g, L, S, idx, dev, None if pos is None else pos[idx])

    def acc(idx):
        with torch.no_grad():
            ok = [(lg_of(idx[i:i + 128])[:, num].argmax(-1) == torch.tensor(y_tok[idx[i:i + 128]], device=dev)).float() for i in range(0, len(idx), 128)]
            return float(torch.cat(ok).mean())
    for st in range(1, steps + 1):
        j = rng.choice(tr, bs, replace=False); loss = F.cross_entropy(lg_of(j), num[torch.tensor(y_tok[j], device=dev)])
        opt.zero_grad(); loss.backward(); opt.step()
        if st % 100 == 0 or st == steps:
            v = acc(va); hist.append([st, float(loss.detach()), v])
            if v > best[0]:
                best = (v, [p_.detach().clone() for p_ in params], st); bad = 0
            else:
                bad += 1
                if bad >= 5:
                    break
    with torch.no_grad():
        for p_, b_ in zip(params, best[1]):
            p_.copy_(b_)
    return {"best_val": best[0], "best_step": best[2], "hist": hist}, lg_of, acc


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--bs", type=int, default=64)
    a = ap.parse_args(); dev = a.device; n, steps = (400, 60) if a.smoke else (8000, 2000); t0 = time.time()
    torch.backends.cuda.matmul.allow_tf32 = False
    m, g, sha = build(dev); sets = IO.build_sets(); excl = set(sets["excl"]); s_init = (m.mem.s_add.detach().clone(), m.mem.s_del.detach().clone())
    try:                                                                    # the G-AW0 items stay out (a different seed; asserted disjoint below)
        excl |= {s_ for s_ in json.load(open(os.path.join(HERE, "aw_gate_askname.json")))["test_item_sig"]}
    except (OSError, KeyError):
        pass
    rows = items(n, excl); sigs = [r["it"]["sig"] for r in rows]; us = sorted(set(sigs)); perm = np.random.default_rng(0).permutation(len(us)); rank = {us[j]: k for k, j in enumerate(perm)}
    rr = np.array([rank[s_] for s_ in sigs]) / len(us); tr, va, te = [np.where(c)[0] for c in (rr < 0.75, (rr >= 0.75) & (rr < 0.875), rr >= 0.875)]
    S1 = write_s1(m, g, rows, dev, 32); Ld = lower(g, [r["tok"]["dec"] for r in rows], dev, 32)
    y1 = np.array([int(r["it"]["values"][1]) for r in rows]); y2 = np.array([int(r["it"]["values"][2]) for r in rows])
    maj = float(np.bincount(y2[te], minlength=P).max() / len(te)); res = {"gate": "G-AW1 FROZEN-USE (AW_v1 §2)", "init_sha256": sha, "n": n, "split": [len(tr), len(va), len(te)],
                                                                    "seed_items": SEED_ITEMS, "trainable": list(TRAINABLE), "chance_majority_v2_test": maj, "chance_uniform": 1 / P}
    fm, lg_main, acc_main = fit(m, g, Ld, S1, y2, tr, va, dev, steps, a.bs, s_init); res["main"] = {**fm, "test_acc_v2": acc_main(te)}
    with torch.no_grad():                                                   # per-item test answers (MAIN) + control (a) with the MAIN W_o
        num = torch.tensor(NUM_IDS, device=dev); pm = np.concatenate([lg_main(te[i:i + 128])[:, num].argmax(-1).cpu().numpy() for i in range(0, len(te), 128)])
        Z = torch.zeros_like(S1); pz = np.concatenate([logits_at(m, g, Ld, Z, te[i:i + 128], dev)[:, num].argmax(-1).cpu().numpy() for i in range(0, len(te), 128)])
    res["main"]["test_pred"] = pm.tolist(); res["control_a_S0"] = {"test_acc_v2": float((pz == y2[te]).mean()), "test_pred": pz.tolist()}
    swap = {"n": 0}                                                         # descriptive SWAP with the MAIN W_o (same-skeleton donors)
    rd = np.random.default_rng(SEED_DONOR); dons = [OW.redraw(rd, rows[j]["it"]) for j in te]; keep = [k for k, d_ in enumerate(dons) if d_ is not None]
    if keep:
        drows = [{"it": dons[k], "tok": IO.tok_ow(dons[k]), "x1": [EOT] + ENC.encode(" " + dons[k]["text"] + " " + OW.QUESTION.format(u=dons[k]["unit"], A=dons[k]["names"]["spine"][1])) + SFX} for k in keep]
        Sd = write_s1(m, g, drows, dev, 32)
        with torch.no_grad():
            m.mem.fade.fill_(1.0); Ssw = S1.clone(); Ssw[te[keep]] = Sd
            ps = np.concatenate([logits_at(m, g, Ld, Ssw, te[keep][i:i + 128], dev)[:, num].argmax(-1).cpu().numpy() for i in range(0, len(keep), 128)])
        rec = [rows[te[k]]["it"] for k in keep]; ca = np.array([IO.cf(r_, d_, 1, "a") for r_, d_ in zip(rec, [dons[k] for k in keep])]); cb = np.array([IO.cf(r_, d_, 1, "b") for r_, d_ in zip(rec, [dons[k] for k in keep])])
        own = y2[te[keep]]; dfin = np.array([int(dons[k]["answer"]) for k in keep]); chg = ca != own
        swap = {"n": len(keep), "n_cf_changed": int(chg.sum()), "follow_cf_a_changed": float((ps[chg] == ca[chg]).mean()) if chg.any() else None,
                "follow_cf_b_changed": float((ps[chg] == cb[chg]).mean()) if chg.any() else None, "own_changed": float((ps[chg] == own[chg]).mean()) if chg.any() else None,
                "donor_final": float((ps == dfin).mean()), "pred": ps.tolist(), "cf_a": ca.tolist(), "cf_b": cb.tolist()}
    res["swap_desc"] = swap
    wo_full = m.mem.wo; m.mem.wo = LowRank(wo_full.weight.shape[0], wo_full.weight.shape[1], 32, dev)
    fr, _, acc_r = fit(m, g, Ld, S1, y2, tr, va, dev, steps, a.bs, s_init); res["rank32_desc"] = {**fr, "test_acc_v2": acc_r(te)}; m.mem.wo = wo_full
    aux_seqs = [m.aux_prompt(1) + r["tok"]["aux"][0][:-1] for r in rows]; La = lower(g, aux_seqs, dev, 32)   # control (b): LP2's path (target v_1 at its value token)
    vpos = np.array([len(m.aux_prompt(1)) - 1 + r["tok"]["vidx"][0] for r in rows]); assert all(aux_seqs[k][vpos[k] + 1] == NUM_IDS[y1[k]] for k in range(len(rows)))
    fb, _, acc_b = fit(m, g, La, S1, y1, tr, va, dev, steps, a.bs, s_init, pos=torch.tensor(vpos, device=dev)); res["control_b_lp2_path"] = {**fb, "test_acc_v1": acc_b(te)}
    mt = res["main"]["test_acc_v2"]; ca_ok = res["control_a_S0"]["test_acc_v2"] <= maj + 0.05; cb_ok = res["control_b_lp2_path"]["test_acc_v1"] >= 0.80
    res["verdict"] = ("SMOKE (plumbing only)" if a.smoke else "UNREADABLE (control)" if not (ca_ok and cb_ok) else "PASS" if mt >= 0.50 else "FAIL" if mt <= maj + 0.10 else "GO (between)")
    res["rule"] = "PASS iff MAIN v_2 >= 0.50 with (a) <= majority + 0.05 and (b) >= 0.80; FAIL iff MAIN <= majority + 0.10 (to the user); between -> GO"
    res["sec"] = round(time.time() - t0); res["device"] = torch.cuda.get_device_name(dev) if str(dev).startswith("cuda") else "cpu"
    out = os.path.join(HERE, "aw_frozen_use_smoke.json" if a.smoke else "aw_frozen_use.json")
    with open(out, "w", newline="\n") as f:
        json.dump(res, f); f.write("\n")
    print("VERDICT", res["verdict"], "->", out, res["sec"], "s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
