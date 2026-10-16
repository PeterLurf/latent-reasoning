"""§211 VERSION-BANK Stage 0 gates (Theory pin 6: diff E_t against the red team's reference). -> scratch/gate_s211/<gate>.json
    python scratch/gate_s211.py et      # boxes.vbank.recode_targets vs scratch/s211_ref_slot_targets.slot_targets on 100k items, n 1-16"""
import json
import os
import sys

sys.path.insert(0, "D:/Latent"); sys.path.insert(0, "D:/Latent/scratch"); os.chdir("D:/Latent")
import numpy as np  # noqa: E402
import torch  # noqa: E402

import boxes.nested170 as N  # noqa: E402
import boxes.vbank as VB  # noqa: E402


def gate_et(n_items=100_000, seed=20902, n_max=16):
    import s211_ref_slot_targets as R
    rng = np.random.default_rng(seed); bad = 0; vers = 0; first = None
    for i in range(n_items):
        root, _ = N.generate(rng, int(rng.integers(1, n_max + 1))); ref = R.slot_targets(root)
        mine = [["V" if c == VB.VOID else VB.CLS[c] for c in e] for e in VB.recode_targets(root)]
        vers += len(ref)
        if mine != ref:
            bad += 1; first = first if first is not None else i
    return {"items": n_items, "seed": seed, "n_range": [1, n_max], "versions": vers, "diffs": bad, "first_diff_item": first, "pass": bad == 0,
            "reference": "scratch/s211_ref_slot_targets.py (red team, sha256 ebf25d3b... of the source file)"}



# ---- Part 3 gates (VBankN): G-FP function preservation, c = 1e4 == CARRY bitwise, masked-K (batched/padded == per item), G1-cal + ratio test
def _items(n_items, seed, n_lo=1, n_hi=8):
    rng = np.random.default_rng(seed); out = []
    for _ in range(n_items):
        r, y = N.generate(rng, int(rng.integers(n_lo, n_hi + 1))); out.append({"syms": N.render(r), "t": y, "root": r, "n": None})
    return out


def _model(mode, dt, dev="cpu", seed=21100):
    from boxes import s5
    torch.manual_seed(seed); g = s5.load_gpt2(dev); m = VB.VBankN(g, mode).to(dev)
    return m.double() if dt == torch.float64 else m


def gate_fp(dev="cpu"):
    """draft §5.5: CARRY (and PA at c = 1e4) at K = 1 with the bank masked == GPT-2's logits at ' =' (fp64, <= 1e-9 rel)."""
    out = {}
    for mode in ("carry", "pa"):
        m = _model(mode, torch.float64, dev); m.bank_off = True; its = _items(6, 21101); x, pad, lens = VB.batch_x(its, dev)
        with torch.no_grad():
            r = m.rollout(x, pad, 1, coda_rows={1: torch.arange(len(its))}); z = m.answer_logits(r["coda"][1][1], lens)
            h, _ = m.g(m.g.embed(x)); zr = m.g.logits(h[torch.arange(len(its)), lens - 1])
        out[mode] = float((z - zr).abs().max() / zr.abs().max())
    out["pass"] = all(v <= 1e-9 for v in out.values()); return out


def gate_carry_bitwise(dev="cpu"):
    """pin 1: PURE-ANNEAL at c = 1e4 == CARRY bitwise (every stored version V_1..V_12 and the coda logits at every t), bank ON (beta -5, ALiBi gamma),
    fp32 and fp64; plus c = 0 differs (the gather is live)."""
    out = {}
    for dt in (torch.float32, torch.float64):
        m = _model("pa", dt, dev); its = _items(6, 21102); x, pad, lens = VB.batch_x(its, dev); rows = {t: torch.arange(len(its)) for t in range(13)}
        with torch.no_grad():
            m.c_anneal.fill_(1e4); ra = m.rollout(x, pad, 12, coda_rows=rows)
            m.mode = "carry"; rc = m.rollout(x, pad, 12, coda_rows=rows); m.mode = "pa"
            m.c_anneal.fill_(0.0); r0 = m.rollout(x, pad, 12, coda_rows=rows); m.c_anneal.fill_(1e4)
        eqV = all(torch.equal(a[pad], c[pad]) for a, c in zip(ra["V"], rc["V"])); eqZ = all(torch.equal(ra["coda"][t][1][pad], rc["coda"][t][1][pad]) for t in rows)
        d0 = max(float((a[pad] - c[pad]).abs().max()) for a, c in zip(r0["V"][2:], rc["V"][2:]))
        out[str(dt)] = {"V_bitwise": eqV, "coda_bitwise": eqZ, "c0_maxdiff_V2plus": d0}
    out["pass"] = all(v["V_bitwise"] and v["coda_bitwise"] and v["c0_maxdiff_V2plus"] > 1e-6 for v in out.values()); return out


def gate_maskk(dev="cpu"):
    """draft §9 item 5: the batched, right-padded path (per-row coda subsets) == each item alone, unpadded (fp64, real slots; V_1..V_8 and coda logits
    at t in its window), both modes, bank ON with c = 0 (PA) so padding enters every softmax."""
    out = {}
    for mode in ("carry", "pa"):
        m = _model(mode, torch.float64, dev); m.c_anneal.fill_(0.0); its = _items(5, 21103); x, pad, lens = VB.batch_x(its, dev)
        rows = {t: torch.tensor([i for i in range(len(its)) if (i + t) % 2 == 0]) for t in range(1, 9)}; worst = 0.0
        with torch.no_grad():
            rb = m.rollout(x, pad, 8, coda_rows=rows)
            for i in range(len(its)):
                xi, pi, li = VB.batch_x([its[i]], dev); ri = m.rollout(xi, pi, 8, coda_rows={t: torch.tensor([0]) for t in range(1, 9)}); L_ = int(li[0])
                for t in range(9):
                    worst = max(worst, float((rb["V"][t][i, :L_] - ri["V"][t][0, :L_]).abs().max() / ri["V"][t][0, :L_].abs().max()))
                for t, rr in rows.items():
                    if i in rr.tolist():
                        k_ = rr.tolist().index(i); za = m.answer_logits(rb["coda"][t][1][k_:k_ + 1], lens[i:i + 1]); zb = m.answer_logits(ri["coda"][t][1], li)
                        worst = max(worst, float((za - zb).abs().max() / zb.abs().max()))
        out[mode] = worst
    out["pass"] = all(v <= 1e-9 for k, v in out.items()); return out


def gate_cal(dev="cuda:0"):
    """pin 3 G1-cal: calibrate beta0 then beta on 64 items n <= 8, K = 12 (fp32), then the ratio test."""
    m = _model("pa", torch.float32, dev); its = _items(64, 21104); x, pad, lens = VB.batch_x(its, dev)
    rep = m.calibrate(x, pad, lens); rt = m.ratio_test(x, pad, lens)
    return {"calibration": rep, "ratio_test": rt, "pass": rep["max_m99"] <= 0.02 + 1e-6 and rt["pass"]}


if __name__ == "__main__":
    g = sys.argv[1]; r = {"et": gate_et, "fp": gate_fp, "carry_bitwise": gate_carry_bitwise, "maskk": gate_maskk, "cal": gate_cal}[g](); json.dump(r, open("scratch/gate_s211/%s.json" % g, "w", newline="\n"), indent=1); print(json.dumps(r))
