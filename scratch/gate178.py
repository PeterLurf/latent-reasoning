"""THEORY §178 FROM-SCRATCH pre-launch gates (b2b4634f), boxes/nested170.py --scratch "6,384,6" (+ deltamem / core generalised to any width):
  (1) no pretrained weights: make_trunk on the scratch path never calls s5.load_gpt2 (it is made to raise here) and its wte is uncorrelated with GPT-2's
  (2) memory placement: after block L // 2 = 3, 3 additive + 3 delta heads (DH 64, D 384), state (B, 6, 64, 64), LN_w from the last block (5)
  (3) gradflow (fade 0.5, W_o perturbed, gamma 0.5, K = 3): every DeltaMem parameter, the random trunk (block 0 / 5), the memory after passes 1 and 2
  (4) the decoder writes nothing (S bitwise unchanged by dec_logits)
  (5) the default path is unchanged: GPT-2 small's DeltaMem (6 + 6, D 768) read_fixed / write_seq outputs equal the pre-§178 deltamem's bitwise
      (scratchpad copy deltamem_pre178.py, with and without a write mask), and core.GPT2KV(pretrained) has D 768 / H 12
  (6) smoke_plain on the scratch path: 3 CPU steps + eval of nested170 inject K2 and direct with --scratch 6,384,6
    python scratch/gate178.py <pre-178 deltamem copy>   -> boxes/runs_fsq/s178_gates/gate178.json"""
import importlib.util
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
os.environ.setdefault("BOXES_NB", "5")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

import boxes.deltamem as DM  # noqa: E402
import boxes.nested170 as N  # noqa: E402
from boxes import s5  # noqa: E402


def main():
    torch.set_num_threads(3); out = {}; ok = True; a = {"scratch": "6,384,6", "seed": 1}
    real = s5.load_gpt2; s5.load_gpt2 = lambda *x, **k: (_ for _ in ()).throw(RuntimeError("pretrained load on the scratch path"))
    try:
        g = N.make_trunk(a, "cpu"); loaded = False
    except RuntimeError:
        loaded = True
    finally:
        s5.load_gpt2 = real
    gp = s5.load_gpt2("cpu"); w0 = g.w("wte.weight")[:, :384].flatten(); w1 = gp.w("wte.weight")[:, :384].flatten()
    corr = float(torch.corrcoef(torch.stack([w0, w1]))[0, 1])
    out["g1"] = {"pretrained_load_called": loaded, "wte_corr_with_gpt2": corr, "L": g.L, "D": g.D, "H": g.H}
    out["g1_pass"] = (not loaded) and abs(corr) < 0.01 and (g.L, g.D, g.H) == (6, 384, 6); ok &= out["g1_pass"]
    ml, ha, hd = N.mem_geometry(a); torch.manual_seed(1); m = N.InjectN(g, ml, "top", ha, hd)
    S0 = m.mem.zero_state(2, "cpu"); out["g2"] = {"layer": m.mem.layer, "h_add": m.mem.h_add, "h_del": m.mem.h_del, "D": m.mem.D, "state_shape": list(S0.shape),
                                                  "ln_w_from_last_block": bool(torch.equal(m.mem.ln_w.weight, g.w("h.5.ln_1.weight").float()))}
    out["g2_pass"] = (out["g2"]["layer"], out["g2"]["h_add"], out["g2"]["h_del"], out["g2"]["D"]) == (3, 3, 3, 384) and out["g2"]["state_shape"] == [2, 6, 64, 64] and out["g2"]["ln_w_from_last_block"]
    ok &= out["g2_pass"]
    with torch.no_grad():
        m.mem.wo.weight.normal_(0, 0.02)
    m.mem.fade.fill_(0.5); m.mem.gamma.fill_(0.5); m.train(); its = N.train_items(np.random.default_rng(3), 4, 8, set()); kept = []; w_ = N.InjectN.write

    def wr(self, S, x, mask):
        S2 = w_(self, S, x, mask); S2.retain_grad(); kept.append(S2); return S2
    N.InjectN.write = wr
    try:
        x, mk = m.think_inputs(its, "cpu"); S = m.think(m.mem.zero_state(4, "cpu"), x, mk, 3); lg = m.dec_logits(S, its, "cpu")
        Fn.cross_entropy(lg, torch.tensor(N.NUM_IDS[[it["t"] for it in its]])).backward()
    finally:
        N.InjectN.write = w_
    mz = [n_ for n_, p_ in m.mem.named_parameters() if p_.grad is None or float(p_.grad.norm()) == 0]
    tg = {k: float(g.w(k).grad.norm()) if g.w(k).grad is not None else 0.0 for k in ("h.0.attn.c_attn.weight", "h.5.mlp.c_fc.weight", "wte.weight")}
    out["g3"] = {"mem_zero": mz, "trunk_grads": tg, "S_grad": [float(k_.grad.norm()) if k_.grad is not None else 0.0 for k_ in kept]}
    out["g3_pass"] = not mz and min(tg.values()) > 0 and min(out["g3"]["S_grad"][:2]) > 0; ok &= out["g3_pass"]
    m.eval()
    with torch.no_grad():
        S = m.think(m.mem.zero_state(4, "cpu"), x, mk, 2); Sc = S.clone(); m.dec_logits(S, its, "cpu")
    out["g4_pass"] = bool(torch.equal(S, Sc)); ok &= out["g4_pass"]
    spec = importlib.util.spec_from_file_location("dm_pre", sys.argv[1]); P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
    torch.manual_seed(0); mem = DM.DeltaMem(gp, 6, "top")
    with torch.no_grad():
        mem.gamma.fill_(0.5); mem.fade.fill_(1.0); mem.wo.weight.normal_(0, 0.02); mem.gate.weight.normal_(0, 0.01)
    h6, hw, S5 = torch.randn(3, 9, 768), torch.randn(3, 9, 768), torch.randn(3, 12, 64, 64); msk = torch.ones(3, 9, dtype=torch.bool); msk[1, 5:] = False
    eq = {"write": bool(torch.equal(DM.write_seq(mem, h6, hw, S5), P.write_seq(mem, h6, hw, S5))),
          "write_mask": bool(torch.equal(DM.write_seq(mem, h6, hw, S5, msk), P.write_seq(mem, h6, hw, S5, msk))),
          "read": bool(torch.equal(DM.read_fixed(mem, h6, S5), P.read_fixed(mem, h6, S5))), "gpt2_D_H": [gp.D, gp.H], "mem_D_heads": [mem.D, mem.h_add, mem.h_del]}
    out["g5"] = eq; out["g5_pass"] = eq["write"] and eq["write_mask"] and eq["read"] and eq["gpt2_D_H"] == [768, 12] and eq["mem_D_heads"] == [768, 6, 6]; ok &= out["g5_pass"]
    sp = os.environ.get("GATE178_OUT", os.path.join(ROOT, "boxes", "runs_fsq", "s178_gates")); env = dict(os.environ, OMP_NUM_THREADS="3"); g6 = {}
    for arm, extra in (("inject", ["--think_k", "2"]), ("direct", [])):
        od = os.path.join(sp, "smoke_" + arm)
        r_ = subprocess.run([sys.executable, "-m", "boxes.nested170", "--arm", arm, "--scratch", "6,384,6", "--warmup", "1000", "--lr", "0.0005", "--seed", "1", "--steps", "3",
                             "--sched_scale", "0.002", "--log_every", "1", "--bs", "4", "--per_n", "2", "--eval_bs", "8", "--serial_pool", "40", "--device", "cpu", "--out", od] + extra,
                            capture_output=True, text=True, env=env)
        e_ = json.load(open(os.path.join(od, "eval.json"))) if r_.returncode == 0 else {}; ar = json.load(open(os.path.join(od, "args.json"))) if r_.returncode == 0 else {}
        g6[arm] = {"rc": r_.returncode, "err": r_.stderr[-400:] if r_.returncode else "", "acc_by_n_keys": len(e_.get("acc_by_n", {})), "scratch": ar.get("scratch"), "warmup": ar.get("warmup")}
    out["g6"] = g6; out["g6_pass"] = all(v["rc"] == 0 and v["acc_by_n_keys"] == 16 and v["scratch"] == "6,384,6" for v in g6.values()); ok &= out["g6_pass"]
    out["all_pass"] = bool(ok); gd = os.path.join(ROOT, "boxes", "runs_fsq", "s178_gates"); os.makedirs(gd, exist_ok=True)
    json.dump(out, open(os.path.join(gd, "gate178.json"), "w", newline="\n"), indent=1); print("GATE178", json.dumps(out), flush=True); sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
