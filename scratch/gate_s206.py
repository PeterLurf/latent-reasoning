"""§206 JOINT-COT-L1 gates (docs/THEORY_JOINTCOT_DRAFT_2026-10-14.md §9, registered a005c7c7 / de56ca30; boxes/gsm8k_jcot.py + the gsm8k_stepaux hook):
  g2 JOINT-EXACT  : on 4 items, the two-backward update (teacher gamma_T L_T, then student L_S + beta L_KD) equals one backward of the sum. deltamem
                    casts the memory branch to float32 internally (as §204 gate 9), so this runs in fp32 CPU: max |dgrad| / max |grad| <= 1e-5.
  g3 gradflow     : fade 1, gamma_w 1, W_o ~ N(0, 1e-3), C's trunk: L_T1 / L_T2 alone -> grad(g) != 0 and grad(mem.*) None or 0; L_KD alone ->
                    grad(mem.wo) != 0 and grad(mem.qkv k / v rows) != 0, z.requires_grad False (the teacher gets NO grad from L_KD); L_KD restricted
                    to layers 1-6 -> grad(mem.*) == 0 (the read enters block 6); ast: no decorator on gsm8k_jcot's functions.
  g4 formats      : 200 Aug-train items: T2 == T1 minus exactly its last line; the '###' positions assert in both teacher sequences and the decoder.
  g_bits          : dec_layers' CE == ProsMem.dec_loss bitwise; fwd_plain_layers' final hidden == GPT2KV.forward bitwise (plain path, fp32 CPU).
-> scratch/gate_s206/gate.json        python scratch/gate_s206.py"""
import ast
import json
import os
import sys

ROOT = "D:/Latent"; sys.path.insert(0, ROOT); os.chdir(ROOT)
os.environ.setdefault("BOXES_NB", "5")
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(3)
import boxes.deltamem as DM  # noqa: E402
import boxes.gsm8k_data as G  # noqa: E402
import boxes.gsm8k_jcot as GJ  # noqa: E402
from boxes import s5  # noqa: E402
from boxes.prosqa168s2 import ENC, EOS, ProsMem  # noqa: E402

DEV = "cpu"; ac = lambda: torch.autocast("cpu", enabled=False)  # noqa: E731
C_PT = "boxes/runs_fsq/gsm_txt_aug_s1/model.pt"


def build():
    g = s5.load_gpt2(DEV); g.load_state_dict(torch.load(C_PT, map_location=DEV, weights_only=False)); m = ProsMem(g).to(DEV); m.train()
    with torch.no_grad():
        m.mem.fade.fill_(1.0); m.mem.gamma.fill_(1.0); m.mem.wo.weight.copy_(torch.randn(768, 768, generator=torch.Generator().manual_seed(5)) * 1e-3)
    DM.FAST_ADD = True; DM.CHUNK_DELTA = 64
    return g, m


def think(m, its, K=3):
    x, msk = m.think_inputs(its, DEV); S = m.mem.zero_state(len(its), DEV)
    for _ in range(K):
        S = m.write(S, x, msk)
    return S


def gn(m, pref):
    v = [p.grad.double().pow(2).sum() for n, p in m.named_parameters() if n.startswith(pref) and p.grad is not None]
    return float(torch.stack(v).sum().sqrt()) if v else 0.0


def main():
    OUT = {}; R, _ = G.load_aug("train"); rng = np.random.default_rng(20600); idx = rng.permutation(len(R))
    enc = lambda r: {**G.encode(r, ENC, EOS), "_q": r["q"]}  # noqa: E731
    its = [enc(R[i]) for i in idx[:60] if 2 <= len(R[i]["eqs"]) <= 4][:4]; L = GJ.parse_layers("7-12")
    g, m = build()
    # g_bits
    with torch.no_grad():
        S = think(m, its); ce0 = m.dec_loss(S, its, DEV); ce1, _ = GJ.dec_layers(m, S, its, DEV, L)
        x = torch.tensor(GJ.pad([it["pre"] for it in its])[0]); h0 = g(g.embed(x), None)[0]; h1, _ = GJ.fwd_plain_layers(g, x, torch.zeros(len(its), dtype=torch.long), L)
    OUT["g_bits"] = {"dec_ce_equal": bool(torch.equal(ce0, ce1)), "plain_final_equal": bool(torch.equal(h0, h1))}; OUT["g_bits"]["pass"] = all(OUT["g_bits"].values())
    # g2 JOINT-EXACT
    def grads(two):
        m.zero_grad(set_to_none=True); L1_, L2_, z, rows = GJ.teacher(g, its, DEV, ac, L, "drop", 2); lt = 2.0 * (L1_ + L2_)
        if two:
            lt.backward()
        S_ = think(m, its); la, cap = GJ.dec_layers(m, S_, its, DEV, L); kd, _, _, _ = GJ.kd_loss(z, cap, rows); ls = la + 1.0 * kd
        (ls if two else ls + lt).backward(); return {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}
    ga, gb = grads(True), grads(False); mx = max(float((ga[n] - gb[n]).abs().max()) for n in ga); scale = max(float(v.abs().max()) for v in ga.values())
    OUT["g2_joint_exact"] = {"max_abs_diff": mx, "max_abs_grad": scale, "rel": mx / scale, "same_keys": sorted(ga) == sorted(gb), "dtype": "float32 (deltamem casts the memory branch to fp32)",
                             "pass": sorted(ga) == sorted(gb) and mx / scale <= 1e-5}
    print("g_bits", OUT["g_bits"], "\ng2", OUT["g2_joint_exact"], flush=True)
    # g3 gradflow per term
    res = {}
    for term in ("T1", "T2", "KD", "KD_low"):
        m.zero_grad(set_to_none=True); lay = L if term != "KD_low" else GJ.parse_layers("1-6")
        L1_, L2_, z, rows = GJ.teacher(g, its, DEV, ac, lay, "drop", 2)
        if term == "T1":
            loss = L1_
        elif term == "T2":
            loss = L2_
        else:
            S_ = think(m, its); _, cap = GJ.dec_layers(m, S_, its, DEV, lay); loss, _, _, _ = GJ.kd_loss(z, cap, rows)
        loss.backward(); gq = m.mem.qkv.weight.grad
        res[term] = {"g": gn(m, "g."), "mem": gn(m, "mem."), "mem.wo": gn(m, "mem.wo"), "qkv_kv": float(gq[768:].norm()) if gq is not None else 0.0,
                     "z_requires_grad": any(v.requires_grad for v in z.values()) if z else None}
    ok = (res["T1"]["g"] > 0 and res["T1"]["mem"] == 0 and res["T2"]["g"] > 0 and res["T2"]["mem"] == 0 and res["KD"]["mem.wo"] > 0 and res["KD"]["qkv_kv"] > 0
          and res["KD"]["z_requires_grad"] is False and res["KD_low"]["mem"] == 0)
    tree = ast.parse(open("boxes/gsm8k_jcot.py", encoding="utf-8").read()); decs = {f.name: [ast.unparse(d) for d in f.decorator_list] for f in tree.body if isinstance(f, ast.FunctionDef)}
    OUT["g3_gradflow"] = {"by_term": res, "decorators": decs, "pass": ok and not any(decs.values())}; print("g3", OUT["g3_gradflow"], flush=True)
    # g4 formats
    bad = 0; n = 0
    for i in idx[1000:1400]:
        it = enc(R[i])
        if it["m"] < 2:
            continue
        n += 1; ln = GJ.lines_of(it); t1 = it["pre"] + it["cot"] + it["a1"] + [EOS]; t2 = it["pre"] + [t for l_ in ln[:-1] for t in l_] + it["a1"] + [EOS]
        bad += t2 != t1[:len(it["pre"]) + len(it["cot"]) - len(ln[-1])] + it["a1"] + [EOS]
        if n >= 200:
            break
    with torch.no_grad():
        GJ.teacher(g, [enc(R[i]) for i in idx[1000:1200]], DEV, ac, L, "drop", 2); GJ.teacher(g, [enc(R[i]) for i in idx[1000:1100]], DEV, ac, L, "full", 2)
    OUT["g4_formats"] = {"n_items": n, "t2_mismatch": int(bad), "positions_asserted": True, "pass": bad == 0}; print("g4", OUT["g4_formats"], flush=True)
    OUT["pass_all"] = all(OUT[k]["pass"] for k in ("g_bits", "g2_joint_exact", "g3_gradflow", "g4_formats"))
    os.makedirs("scratch/gate_s206", exist_ok=True); json.dump(OUT, open("scratch/gate_s206/gate.json", "w", newline="\n"), indent=1, default=str); print("GATE_S206", OUT["pass_all"], flush=True)


if __name__ == "__main__":
    main()
