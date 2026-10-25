"""ANSWER-WRITE (AW) $0 local gate G-AW0 (AW_draft.md sec. 2), pre-stated there BEFORE this script is run on the init model. CPU (or a free local card),
2 threads, idle priority, fp32, no_grad. Audit's condition: v_1 linearly decodable at the arm's ACTUAL write source, on the arm's ACTUAL init, >= 0.80.
Model, built as the arm builds it: pretrained GPT-2 (IO.load_trunk), the init trunk loaded strict (default boxes/runs_fsq/s233_askname_s1/model.pt; its
sha256 is logged), torch.manual_seed(1), then IO.InjectOW(g, 6, "top", 6, 6, None) (DeltaMem copies block 6's ln_1 / c_attn and block 11's ln_1 from
the LOADED trunk). S = zero state; fade = 0 (its step-0 value): the read adds fade * W_o(...) = 0 exactly (asserted).
Items: N d = 2 items, OW.Stream(23471, EXCL = build_sets()["excl"], frames 0-5, topo, dag) (seed unused elsewhere, grep 10-25). Probe protocol = Audit's
answer-position probe (231-1 family): split by sig 60 / 20 / 20 (rng 0), standardised multinomial logistic regression, LBFGS 200 iterations, L2 wd in
{1e-4, 1e-3, 1e-2} chosen on val; test acc; chance = test majority share and 1 / 11. Per-item test predictions are stored.
Pass-1 input x_1 = [EOT] + enc(" " + text + " " + QUESTION(u, S_1)) + enc(" The answer is:"); the write position is its last token ':'.
Features (all from the arm's own forward DM.gpt_forward_mem_r with cap):
  u1      = mem.ln_w(h_w) at ':'   (gamma = 1: the write source from write_mix_end = 6000 on)            BINDING  -> v_1
  u0      = mem.ln(h_6) at ':'     (gamma = 0: the source up to write_mix_start = 2000)                  descriptive
  uhalf   = 0.5 u0 + 0.5 u1        (gamma = 0.5, step 4000)                                              descriptive
  s1      = the single-token write of u1 into the zero state (12 x 64 x 64)                             descriptive
  name    = u1 at S_1's name token in the cue (the "cheaper" 0.855 source)                              descriptive
  l0      = wte + wpe at ':'                                                                             control (<= majority + 0.05)
  v0ctl   = u1 at the '.' ending S_0's GIVEN sentence -> v_0                                             control (>= 0.80, 231-2's bar)
  v0ctl_u0 = u0 at the same '.' -> v_0                                                                  descriptive
  iv_u1   = u1 at ':' of the pass-2 job in text form (S_1's sentence replaced by GIVEN, cue S_2) -> v_2  descriptive
  behaviour: the trunk's own argmax over the 11 number tokens at ':' (x_1 -> v_1; iv -> v_2)               descriptive
Rule (AW_draft sec. 2): PASS iff u1|v1 test >= 0.80 and l0|v1 <= majority + 0.05 and v0ctl|v0 >= 0.80 and read == 0. u1 < 0.80 with valid controls ->
FAIL (AW stops at $0). A control fails -> UNREADABLE (fix once, re-run once).
    python -B docs/research_answerpos_2026-10-25/aw_gate.py                       (binding: askname init, N 3000 -> aw_gate_askname.json)
    python -B docs/research_answerpos_2026-10-25/aw_gate.py --init base|pdir2c     (descriptive references -> aw_gate_<init>.json)
    python -B docs/research_answerpos_2026-10-25/aw_gate.py --smoke                (base GPT-2, N 48: plumbing only -> aw_gate_smoke.json)
"""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2"); os.environ.setdefault("MKL_NUM_THREADS", "2"); os.environ.setdefault("BOXES_NB", "5")
try:
    import psutil
    psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS if os.name == "nt" else 19)
except Exception:  # noqa: BLE001
    pass
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

torch.set_num_threads(2)
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
import boxes.deltamem as DM  # noqa: E402
import boxes.inject_ow as IO  # noqa: E402
import boxes.ow_words as OW  # noqa: E402
from boxes.nested170 import EOT, NUM_IDS, pad  # noqa: E402

ENC = IO.ENC; SFX = IO.SFX; P = 11; SEED_ITEMS = 23471
INITS = {"askname": os.path.join(ROOT, "boxes", "runs_fsq", "s233_askname_s1", "model.pt"),
         "pdir2c": os.path.join(ROOT, "boxes", "runs_fsq", "s228_pdir2c_s1", "model.pt"), "base": None}
_sp = importlib.util.spec_from_file_location("st_ask_name_local", os.path.join(ROOT, "docs", "research_steptold_2026-10-24", "st_ask_name_local.py"))
AN = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(AN)


def fit_probe(X, y, tr, va, te):
    """Audit's fit_probe (scratch/audit_answerpos_probe.py), CPU."""
    mu = X[tr].mean(0, keepdim=True); sd = X[tr].std(0, keepdim=True) + 1e-6; Z = (X - mu) / sd; Y = torch.tensor(y); best = None
    for wd in (1e-4, 1e-3, 1e-2):
        W = torch.zeros(Z.shape[1], P, requires_grad=True); b = torch.zeros(P, requires_grad=True)
        opt = torch.optim.LBFGS([W, b], lr=1, max_iter=200, history_size=20, line_search_fn="strong_wolfe")

        def clo():
            opt.zero_grad(); l_ = F.cross_entropy(Z[tr] @ W + b, Y[tr]) + 0.5 * wd * (W ** 2).sum(); l_.backward(); return l_
        with torch.enable_grad():
            opt.step(clo)
        with torch.no_grad():
            pr = (Z @ W + b).argmax(-1); va_acc = float((pr[va] == Y[va]).float().mean())
            if best is None or va_acc > best[0]:
                best = (va_acc, wd, pr[te].numpy(), float((pr[tr] == Y[tr]).float().mean()))
    yt = y[te]
    return {"test_acc": float((best[2] == yt).mean()), "val_acc": best[0], "train_acc": best[3], "wd": best[1], "n_test": int(len(te)),
            "chance_majority": float(np.bincount(yt, minlength=P).max() / len(yt)), "chance_uniform": 1 / P, "test_pred": best[2].tolist(), "feature_dim": int(X.shape[1])}


def find(seq, pat):
    hit = [i for i in range(len(seq) - len(pat) + 1) if seq[i:i + len(pat)] == pat]
    return hit[0] if len(hit) == 1 else None


def build_items(n, excl):
    s = OW.Stream(SEED_ITEMS, excl, frames=OW.FRAME_TRAIN, **IO.OWKW); rows = []; skipped = {"name": 0, "s0": 0, "iv": 0}
    while len(rows) < n:
        it = s.draw(2); sp = it["names"]["spine"]; u = it["unit"]; th = [EOT] + ENC.encode(" " + it["text"])
        x1 = [EOT] + ENC.encode(" " + it["text"] + " " + OW.QUESTION.format(u=u, A=sp[1])) + SFX
        assert x1[:len(th)] == th and x1[-1] == SFX[-1]
        npos = len(th) + len(ENC.encode(" How many %s does %s" % (u, sp[1]))) - 1                 # S_1's (single) name token in the cue
        if x1[npos] != ENC.encode(" " + sp[1])[-1]:
            skipped["name"] += 1; continue
        g0 = find(x1, ENC.encode(" " + OW.GIVEN.format(A=sp[0], v=it["values"][0], u=u)))          # S_0's GIVEN sentence (unique), ending '.'
        if g0 is None:
            skipped["s0"] += 1; continue
        s0end = g0 + len(ENC.encode(" " + OW.GIVEN.format(A=sp[0], v=it["values"][0], u=u))) - 1
        try:
            a, b = AN.s1_sentence(it)
        except AssertionError:
            skipped["iv"] += 1; continue
        lead = " " if a > 0 and it["text"][a] == " " else ""; t2 = it["text"][:a] + lead + OW.GIVEN.format(A=sp[1], v=it["values"][1], u=u) + it["text"][b:]
        xiv = [EOT] + ENC.encode(" " + t2 + " " + OW.QUESTION.format(u=u, A=sp[2])) + SFX
        rows.append({"it": it, "x1": x1, "npos": npos, "s0end": s0end, "xiv": xiv})
    return rows, skipped


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--init", default="askname", choices=sorted(INITS)); ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--bs", type=int, default=32); a = ap.parse_args()
    if a.smoke:
        a.init, a.n = "base", 48
    out = os.path.join(HERE, "aw_gate_smoke.json" if a.smoke else "aw_gate_%s.json" % a.init); t0 = time.time()
    g = IO.load_trunk("cpu", {"scratch": ""}); sha = None
    if INITS[a.init]:
        sha = hashlib.sha256(open(INITS[a.init], "rb").read()).hexdigest()
        g.load_state_dict(torch.load(INITS[a.init], map_location="cpu", weights_only=True), strict=True)
    g.eval(); torch.manual_seed(1); m = IO.InjectOW(g, 6, "top", 6, 6, None); mem = m.mem; mem.eval(); DM.FAST_ADD = True
    assert float(mem.fade) == 0.0 and float(mem.gamma) == 0.0
    sets = IO.build_sets(); rows, skipped = build_items(a.n, sets["excl"]); n = len(rows)
    feats = {k: [] for k in ("u1", "u0", "uhalf", "s1", "name", "l0", "v0ctl", "v0ctl_u0", "iv_u1")}; beh1 = []; behiv = []; read_max = 0.0
    for i in range(0, n, a.bs):
        ch = rows[i:i + a.bs]; B = len(ch); ar = torch.arange(B)
        for key in ("x1", "xiv"):
            seqs = [r[key] for r in ch]; ids_, _ = pad(seqs); x = torch.tensor(ids_); pos = torch.tensor([len(q) - 1 for q in seqs]); cap = {}
            S0 = mem.zero_state(B, "cpu"); hf, _ = DM.gpt_forward_mem_r(g, g.embed(x), mem, S0, cap=cap)
            read_max = max(read_max, float(cap["read"].abs().max()))
            h6, hw = cap["h6"].float(), cap["hw"].float(); u1 = mem.ln_w(hw); u0 = mem.ln(h6)
            beh = g.logits(hf[ar, pos])[:, NUM_IDS].argmax(-1).numpy()
            if key == "xiv":
                feats["iv_u1"].append(u1[ar, pos]); behiv.append(beh); continue
            feats["u1"].append(u1[ar, pos]); feats["u0"].append(u0[ar, pos]); feats["uhalf"].append(0.5 * u0[ar, pos] + 0.5 * u1[ar, pos])
            mem.gamma.fill_(1.0); s1 = DM.write_seq(mem, h6[ar, pos][:, None], hw[ar, pos][:, None], S0, None); mem.gamma.fill_(0.0)
            feats["s1"].append(s1.flatten(1)); feats["name"].append(u1[ar, torch.tensor([r["npos"] for r in ch])])
            feats["l0"].append((g.embed(x) + DM.wpe_slice(g, x.shape[1])[None])[ar, pos]); feats["v0ctl"].append(u1[ar, torch.tensor([r["s0end"] for r in ch])])
            feats["v0ctl_u0"].append(u0[ar, torch.tensor([r["s0end"] for r in ch])])
            beh1.append(beh)
        if i % (a.bs * 10) == 0:
            print("fwd", i + B, "/", n, round(time.time() - t0), "s", flush=True)
    X = {k: torch.cat(v).float() for k, v in feats.items()}
    y1 = np.array([int(r["it"]["values"][1]) for r in rows]); y0 = np.array([int(r["it"]["values"][0]) for r in rows]); y2 = np.array([int(r["it"]["values"][2]) for r in rows])
    sigs = [r["it"]["sig"] for r in rows]; us = sorted(set(sigs)); perm = np.random.default_rng(0).permutation(len(us)); rank = {us[j]: k for k, j in enumerate(perm)}
    rr = np.array([rank[s] for s in sigs]) / len(us); tr, va, te = [np.where(c)[0] for c in (rr < 0.6, (rr >= 0.6) & (rr < 0.8), rr >= 0.8)]
    pr = {}
    for nm, y in (("u1", y1), ("u0", y1), ("uhalf", y1), ("s1", y1), ("name", y1), ("l0", y1), ("v0ctl", y0), ("v0ctl_u0", y0), ("iv_u1", y2)):
        t1 = time.time(); r_ = fit_probe(X[nm], y, tr, va, te); r_["sec"] = round(time.time() - t1); tgt = "v0" if nm.startswith("v0ctl") else ("v2" if nm == "iv_u1" else "v1")
        pr["%s|%s" % (nm, tgt)] = r_; print(nm, tgt, "test %.3f (chance %.3f, wd %g)" % (r_["test_acc"], r_["chance_majority"], r_["wd"]), flush=True)
    beh1 = np.concatenate(beh1); behiv = np.concatenate(behiv)
    u1v = pr["u1|v1"]["test_acc"]; ctl_ok = pr["l0|v1"]["test_acc"] <= pr["l0|v1"]["chance_majority"] + 0.05 and pr["v0ctl|v0"]["test_acc"] >= 0.80 and read_max == 0.0
    verdict = ("UNREADABLE (control)" if not ctl_ok else ("PASS" if u1v >= 0.80 else "FAIL")) if not a.smoke else "SMOKE (plumbing only; not a reading)"
    res = {"gate": "G-AW0 (AW_draft sec. 2)", "init": a.init, "init_path": INITS[a.init], "init_sha256": sha, "n_items": n, "skipped": skipped, "seed_items": SEED_ITEMS,
           "split_sizes": [len(tr), len(va), len(te)], "test_item_sig": [sigs[k] for k in te], "test_y_v1": y1[te].tolist(), "read_max_abs": read_max,
           "behaviour": {"x1_argmax_acc_v1": float((beh1 == y1).mean()), "iv_argmax_acc_v2": float((behiv == y2).mean())}, "probes": pr,
           "rule": "PASS iff u1|v1 >= 0.80 and l0|v1 <= majority + 0.05 and v0ctl|v0 >= 0.80 and read == 0; a control fails -> UNREADABLE", "verdict": verdict,
           "test2_sha": sets["shas"]["test2_sha"], "sec": round(time.time() - t0)}
    with open(out, "w", newline="\n") as f:
        json.dump(res, f); f.write("\n")
    print("VERDICT", verdict, "->", out, res["sec"], "s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
