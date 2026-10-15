"""IIT-SERIAL gates (docs/THEORY_IITSERIAL_DRAFT_2026-10-14.md §5 / §7; boxes/iit_serial.py + the flag-gated nested170 hook), python-level part:
  G1 / G2 gradflow per term: on s185_rms_s1's final model (gamma 1, fade 1), L_IIT alone (w_S = 0) and L_SIIT alone (w_I = 0) each give nonzero
          gradient on mem.qkv q rows and the delta heads' k / v rows, mem.gate, mem.wo, mem.ln_w, block 6 and block 11 attn / mlp weights, and on S_tau.
  G5 ast: no decorator on iit_loss / make_pairs (the gradient path is never under no_grad).
  G-HYB: on 10k kept pairs from train_items (the run's exclusion set) the tree's y_hyb equals eval_tokens on the rendered hybrid, and y / y_d / y_hyb
          match eval_tokens on their renders; donors / SIIT texts render to the recipient's length.
  MEM: the IIT graph's peak at B_I = 16, n = 8 rows, K = 8 (this card; the 3090 G-MEM is the G-TIME host's).
The shell part (G-PLAIN bitwise vs the pre-edit nested170, G3 gamma = 0 => no call and bitwise logs) is scratch/_tmp/iit_gate/run.sh.
-> scratch/gate_iit/gate.json
    python scratch/gate_iit.py [device cuda:2]"""
import ast
import json
import os
import sys

ROOT = "D:/Latent"; sys.path.insert(0, ROOT); os.chdir(ROOT)
os.environ.setdefault("BOXES_NB", "5")
import numpy as np  # noqa: E402
import torch  # noqa: E402

import boxes.deltamem as DM  # noqa: E402
import boxes.halt173 as H  # noqa: E402
import boxes.iit_serial as IIT  # noqa: E402
import boxes.nested170 as N  # noqa: E402


def main():
    dev = sys.argv[1] if len(sys.argv) > 1 else "cuda:2"; out = {"device": dev}
    kind, m, ra = H.load_model(os.path.join(ROOT, "boxes", "runs_fsq", "s185_rms_s1"), dev); DM.FAST_ADD = True; DM.CHUNK_DELTA = 0; K = int(ra["think_k"])
    m.train()
    with torch.no_grad():
        m.mem.gamma.fill_(1.0); m.mem.fade.fill_(1.0)
    amp = None if torch.cuda.get_device_capability(dev)[0] < 8 else torch.bfloat16; ac = lambda: torch.autocast("cuda", dtype=amp, enabled=amp is not None)  # noqa: E731
    out["dtype"] = str(amp); TE = N.test_set(int(ra.get("per_n") or 1000)); excl = {tuple(it["syms"]) for it in TE}
    items = [it for it in N.train_items(np.random.default_rng(20500), 256, 8, excl)]
    HA, HD_ = m.mem.h_add, m.mem.h_del; D = 768; dh = DM.DH
    sel = {"qkv_q": lambda gq: gq[:D], "qkv_k_delta": lambda gq: gq[D + HA * dh:2 * D], "qkv_v_delta": lambda gq: gq[2 * D + HA * dh:]}
    for term, (wI, wS) in (("IIT", (1.0, 0.0)), ("SIIT", (0.0, 1.0))):
        m.zero_grad(set_to_none=True); rec = []
        loss, lg = IIT.iit_loss(m, items, dev, ac, K, np.random.default_rng(7), 1, 0, 4, wI, wS, rec_S=rec); loss.backward()   # B_I 4: fits fp32 on 8 GB
        gq = m.mem.qkv.weight.grad; g = {k_: float(f_(gq).norm()) for k_, f_ in sel.items()}
        for nm in ("gate.weight", "wo.weight", "ln_w.weight"):
            p_ = dict(m.mem.named_parameters())[nm]; g["mem." + nm] = float(p_.grad.norm()) if p_.grad is not None else 0.0
        for nm, p_ in m.g.named_parameters():
            if any(s_ in nm for s_ in ("h_6_attn_c_attn_weight", "h_6_mlp_c_fc_weight", "h_11_attn_c_attn_weight", "h_11_mlp_c_fc_weight")):
                g[nm] = float(p_.grad.norm()) if p_.grad is not None else 0.0
        S_rows = rec[0].grad[:lg["iit_pairs"]] if term == "IIT" else rec[0].grad[lg["iit_pairs"]:]
        g["dL_dS_tau"] = float(S_rows.norm()) if rec[0].grad is not None else 0.0
        out["G_" + term] = {"grad_norms": g, "pairs": lg, "pass": all(v > 0 for v in g.values())}; print(term, out["G_" + term], flush=True)
    tree = ast.parse(open(os.path.join(ROOT, "boxes", "iit_serial.py"), encoding="utf-8").read())
    decs = {f.name: [ast.unparse(d) for d in f.decorator_list] for f in tree.body if isinstance(f, ast.FunctionDef)}
    out["G5_ast"] = {"decorators": decs, "pass": not decs.get("iit_loss") and not decs.get("make_pairs")}
    rng = np.random.default_rng(20501); src = np.random.default_rng(20502); bad = n = 0; lens_ok = True
    while n < 10000:
        its = N.train_items(src, 256, 8, excl); hy = []; rec, don, yh, sx, yy, _ = IIT.make_pairs(its, rng, 1, 0, 256, hyb_out=hy)
        for r_, d_, h_, s_, a_, b_, y_ in zip(rec, don, hy, sx, yh, yy, [x["t"] for x in don]):
            n += 1; bad += N.eval_tokens(N.render(h_)) != a_ or N.eval_tokens(r_["syms"]) != b_ or N.eval_tokens(d_["syms"]) != y_ or N.eval_tokens(s_["syms"]) != s_["t"]
            lens_ok &= len(d_["syms"]) == len(r_["syms"]) == len(s_["syms"])
    out["G_HYB"] = {"n_pairs": n, "mismatch": int(bad), "lengths_match": bool(lens_ok), "pass": bad == 0 and lens_ok}; print("G_HYB", out["G_HYB"], flush=True)
    big = [it for it in N.train_items(np.random.default_rng(20503), 2000, 8, excl) if it["n"] == 8]
    out["MEM"] = {"rows_n": 8, "K": K, "card": torch.cuda.get_device_name(dev), "dtype": str(amp), "note": "8 GB fp32 card: B_I 4 / 8 measured; the 3090 bf16 G-MEM is read on the G-TIME host"}
    for bi in (4, 8):
        m.zero_grad(set_to_none=True); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(dev); base = torch.cuda.memory_allocated(dev)
        try:
            loss, lg = IIT.iit_loss(m, big, dev, ac, K, np.random.default_rng(8), 1, 0, bi); loss.backward(); pk = (torch.cuda.max_memory_allocated(dev) - base) / 2 ** 30
        except torch.OutOfMemoryError:
            pk = float("inf")
        out["MEM"]["iit_graph_peak_gib_BI%d" % bi] = pk
    out["pass_all"] = all(out[k_]["pass"] for k_ in ("G_IIT", "G_SIIT", "G5_ast", "G_HYB"))
    os.makedirs("scratch/gate_iit", exist_ok=True); json.dump(out, open("scratch/gate_iit/gate.json", "w", newline="\n"), indent=1); print("GATE_IIT", out["pass_all"], out["MEM"], flush=True)


if __name__ == "__main__":
    main()
