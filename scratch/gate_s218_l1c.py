"""§218 G-L1c OVERFIT-256 (Note 218-8 b97b410d + Audit's addition): A1's recipe and code path on a FIXED set of 256 SYM v1 items at D 3-4.
  Items: prosqa_sym.Stream(21160) with the test + GATE signatures excluded, D from np rng 21161 (U{3..4}); fixed for the run.
  Model / loss: LangSLVA (pretrained GPT-2), calibrate() on 16 items (rng / Stream 21110, the trainer's calibration law; 16 not 64: the 8 GB eval card),
  lvb_loss with every registered term live (R1 + rec_w 1.0 R2), AdamW lr 1e-4 wd 0.01, langvb's cosine (fsq120.lr_at over 24,000 steps), c = vbank.c_at(step,
  1000, 8000), A1's autocast (bf16 on capability >= 8), grad clip 1.0; each step 16 items drawn from the 256 (np rng 21162, with replacement), K = min(D +
  U{0..4}, 12) (np rng 21163); micro = the smallest of {1, 2, 4, 8} under --mem_budget GiB at the 16 longest items, K 12.
  Step 1 (before the optimizer step): per-term gradient norms ALONE - R1 (lvb_loss at rec_w 0), R2-changed and R2-unchanged (the halves recomputed
  from the kept rollout with lvb_loss's targets / weights, as G-L2). cand_ce is a no-grad statistic (lvb_loss computes it under torch.no_grad; it is not
  a loss term), so it has no gradient by design: recorded as such.
  Every 100 steps: plain_cand_ce (mean of the 100 batch statistics), host candidate accuracy on all 256 at K' = D + 2 (A1's autocast), the first step with
  plain_cand_ce < ln 2. Stop at accuracy >= 0.95 or 3,000 steps. PASS = accuracy >= 0.95 and every per-term grad > 0.
    python -B scratch/gate_s218_l1c.py --device cuda:0
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "D:/Latent"); os.chdir("D:/Latent")
import boxes.langvb as LV  # noqa: E402
import boxes.prosqa_sym as PS  # noqa: E402
import boxes.vbank as VB  # noqa: E402
from boxes import s5  # noqa: E402
from boxes.fsq120 import lr_at  # noqa: E402


def gnorm(m):
    return float(torch.sqrt(sum((p.grad.float() ** 2).sum() for p in m.parameters() if p.grad is not None)))


def term_grads(m, items, Ks, dev, ac):
    out = {}
    m.zero_grad(set_to_none=True); L, _, _ = LV.lvb_loss(m, items, Ks, dev, ac, rec_w=0.0); L.backward(); out["R1"] = gnorm(m)
    m.zero_grad(set_to_none=True); _, _, r = LV.lvb_loss(m, items, Ks, dev, ac, rec_w=1.0, keep=True)
    x, pad, lens, si, sp, dR, dP = LV.lvb_batch(items, dev); Kb = max(Ks); Kt = torch.tensor(Ks, device=dev); E = LV.reach_targets(items, Kb, si.shape[1], dev)
    half = {"chg": [], "unc": []}
    for t in range(1, Kb + 1):
        Et, Ep = E[:, t], E[:, t - 1]; valid = sp & (t <= Kt)[:, None] & (Et >= 0)
        ce = F.cross_entropy(m.rec_logits(r["V"][t]).transpose(1, 2), Et.clamp_min(0), reduction="none").float(); w = 1.0 + (Et == LV.IN).float()
        for k_, msk in (("chg", (Et != Ep) & valid), ("unc", (Et == Ep) & valid)):
            if bool(msk.any()):
                half[k_].append((w * ce * msk.float()).sum() / (w * msk.float()).sum())
    for k_ in ("chg", "unc"):
        m.zero_grad(set_to_none=True)
        if half[k_]:
            torch.stack(half[k_]).mean().backward(retain_graph=True); out["R2_" + k_] = gnorm(m)
        else:
            out["R2_" + k_] = None
    m.zero_grad(set_to_none=True); del r, half
    out["cand_ce"] = "no gradient by design (a torch.no_grad statistic in lvb_loss, read by CurGate; not a loss term)"
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--max_steps", type=int, default=3000)
    ap.add_argument("--n_items", type=int, default=256); ap.add_argument("--bs", type=int, default=16); ap.add_argument("--mem_budget", type=float, default=6.5)
    ap.add_argument("--eval_bs", type=int, default=16); ap.add_argument("--eval_every", type=int, default=100); ap.add_argument("--out", default="scratch/gate_s218/g_l1c.json"); a = ap.parse_args(); dev = a.device
    torch.manual_seed(1); t0 = time.time()
    amp = None if (not dev.startswith("cuda") or torch.cuda.get_device_capability(dev)[0] < 8) else torch.bfloat16
    ac = lambda: torch.autocast("cuda", dtype=amp, enabled=amp is not None)  # noqa: E731
    EX = PS.exclusion_sigs(); s = PS.Stream(21160, EX); rD = np.random.default_rng(21161)
    items = [LV.lvb_item(s.draw(int(D))) for D in rD.integers(3, 5, size=a.n_items)]
    m = LV.make_model(s5.load_gpt2(dev)).to(dev); m.c_anneal.fill_(1e4)
    sc = PS.Stream(21110, EX); r_c = np.random.default_rng(21110); cal_its = [LV.lvb_item(sc.draw(int(r_c.integers(3, 9)))) for _ in range(16)]
    x_, p_, l_, si_, sp_, dR_, _ = LV.lvb_batch(cal_its, dev); cal = m.calibrate(x_, p_, si_, sp_, dR_, l_); del x_, p_, l_, si_, sp_, dR_
    opt = torch.optim.AdamW([p_ for p_ in m.parameters() if p_.requires_grad], lr=1e-4, weight_decay=0.01)
    big = sorted(items, key=lambda it: (-len(it["ids"]), -len(it["slot_idx"])))[:a.bs]; micro = None; pk = {}
    if dev.startswith("cuda"):
        for mic in (1, 2, 4, 8):
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(dev); ok = True; ch = a.bs // mic
            try:
                for k_ in range(mic):
                    L_, _, _ = LV.lvb_loss(m, big[k_ * ch:(k_ + 1) * ch], [12] * ch, dev, ac); (L_ / mic).backward(); del L_
            except torch.OutOfMemoryError:
                ok = False
            m.zero_grad(set_to_none=True); p = torch.cuda.max_memory_allocated(dev) / 2 ** 30; pk[str(mic)] = round(p, 2) if ok else "OOM"
            if ok and p <= a.mem_budget:
                micro = mic; break
        torch.cuda.empty_cache()
    micro = micro or 4
    rb = np.random.default_rng(21162); rk = np.random.default_rng(21163)

    def host_acc():
        m.eval(); ok = []
        with torch.no_grad():
            for D in (3, 4):
                sub = [it for it in items if it["D"] == D]
                for i in range(0, len(sub), a.eval_bs):
                    sR, sP = LV.cand_scores(m, sub[i:i + a.eval_bs], D + 2, dev, ac); ok += (sR > sP).tolist()
        m.train(); return float(np.mean(ok))
    res = {"items_seed": "Stream 21160 / D rng 21161 (U{3..4})", "n_items": len(items), "items_sha": PS.set_sha([{"text": str(it["ids"])} for it in items]),
           "bs": a.bs, "micro": micro, "peak_gib_by_micro": pk, "amp": str(amp), "card": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu",
           "cal_n": 16, "cal_max_m99": cal["max_m99"], "curve": [], "first_step_pcc_below_ln2": None}
    pcc_win = []; acc = None
    for step in range(1, a.max_steps + 1):
        for gr in opt.param_groups:
            gr["lr"] = lr_at(1e-4, {}, step, 24000)
        m.c_anneal.fill_(VB.c_at(step, 1000, 8000)); m.train()
        bt = [items[j] for j in rb.integers(0, len(items), a.bs)]; Ks = [min(it["D"] + int(rk.integers(0, 5)), 12) for it in bt]
        if step == 1:
            res["step1_term_grads"] = term_grads(m, bt[:a.bs // micro], Ks[:a.bs // micro], dev, ac)
            print("STEP1", res["step1_term_grads"], flush=True)
        opt.zero_grad(set_to_none=True); ch = a.bs // micro; pc = []
        for k_ in range(micro):
            L, st, _ = LV.lvb_loss(m, bt[k_ * ch:(k_ + 1) * ch], Ks[k_ * ch:(k_ + 1) * ch], dev, ac); (L / micro).backward()
            if st["plain_cand_ce"] is not None:
                pc.append(st["plain_cand_ce"])
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); pcc_win.append(float(np.mean(pc)) if pc else float("nan"))
        if step % a.eval_every == 0:
            pcc = float(np.nanmean(pcc_win)); pcc_win = []; acc = host_acc()
            res["curve"].append({"step": step, "plain_cand_ce": pcc, "host_acc_256": acc, "c": float(m.c_anneal), "sec": round(time.time() - t0)})
            if res["first_step_pcc_below_ln2"] is None and pcc < math.log(2):
                res["first_step_pcc_below_ln2"] = step
            print(res["curve"][-1], flush=True); json.dump(res, open(a.out, "w", newline="\n"), indent=1)
            if acc >= 0.95:
                break
    g = res["step1_term_grads"]
    res["final_host_acc_256"] = acc; res["steps_run"] = step
    res["grads_ok"] = all(g.get(k) is not None and g[k] > 0 for k in ("R1", "R2_chg", "R2_unc"))
    res["pass"] = bool(acc is not None and acc >= 0.95 and res["grads_ok"]); res["sec"] = round(time.time() - t0)
    json.dump(res, open(a.out, "w", newline="\n"), indent=1); print("G-L1c", "PASS" if res["pass"] else "FAIL", acc, step, flush=True)


if __name__ == "__main__":
    main()
