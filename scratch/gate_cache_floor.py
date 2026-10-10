"""--cache_lower reorder noise floor (Theory 2026-10-10): the UNCACHED path at K = 10 (CPU fp32, gate_cache_lower's fixture) with each pass's lower blocks
computed fresh (as uncached) but kept addressable: the loss's gradient at each pass's block-6 input h6_t is backpropagated through that pass's own lower
blocks separately (torch.autograd.grad), giving the 10 per-pass contributions to the lower-block parameters; summed in forward vs reversed order (fp32).
floor = max over those parameters of max|sum_fwd - sum_rev| / max|sum_fwd|. Ruling (Theory): cached vs uncached accepted iff its max rel diff
(gate_cache_lower.json g2, K10) <= max(1e-6, 2 x floor).
    python scratch/gate_cache_floor.py   -> adds g4 / ruling to boxes/runs_fsq/cache_lower_gates/gate_cache_lower.json"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
os.environ.setdefault("BOXES_NB", "5")
import torch  # noqa: E402

import boxes.deltamem as DM  # noqa: E402
import boxes.prosqa168s2 as P  # noqa: E402
from boxes import s5  # noqa: E402


def main():
    torch.set_num_threads(3); f = os.path.join(ROOT, "boxes", "runs_fsq", "cache_lower_gates", "gate_cache_lower.json"); G = json.load(open(f))
    g = s5.load_gpt2("cpu"); torch.manual_seed(1); m = P.ProsMem(g)
    with torch.no_grad():
        m.mem.wo.weight.normal_(0, 0.02); m.mem.fade.fill_(1.0); m.mem.gamma.fill_(0.5)
    D = P.load(os.path.join(ROOT, "prosqa", "p8data", "p8_na_train.json"))[:2]; x, msk = m.think_inputs(D, "cpu")
    lower_names = ["wte_weight", "wpe_weight"] + [k for k in g.p.keys() if any(k.startswith("h_%d_" % i) for i in range(m.mem.layer))]
    lower = [g.p[k] for k in lower_names]
    S = m.mem.zero_state(2, "cpu"); h6s = []
    for _ in range(10):
        h6 = DM.gpt_lower(g, g.embed(x), m.mem.layer); h6s.append(h6)    # uncached: a fresh lower-block computation every pass
        m._hl_src = x; S = m.write(S, x, msk, h6)
    loss = m.dec_loss(S, D, "cpu"); gh = torch.autograd.grad(loss, h6s, retain_graph=True)   # dL / dh6_t per pass
    contrib = [torch.autograd.grad(h6s[t], lower, grad_outputs=gh[t], retain_graph=True) for t in range(10)]
    floor = 0.0
    for j in range(len(lower)):
        fw = contrib[0][j].clone()
        for t in range(1, 10):
            fw = fw + contrib[t][j]
        rv = contrib[9][j].clone()
        for t in range(8, -1, -1):
            rv = rv + contrib[t][j]
        floor = max(floor, float((fw - rv).abs().max() / (fw.abs().max() + 1e-12)))
    obs = G["g2"]["K10_ckpt0"]["grad_max_rel_diff"]; bound = max(1e-6, 2 * floor)
    G["g4"] = {"floor_fwd_vs_rev_max_rel": floor, "n_lower_params": len(lower), "observed_cached_vs_uncached_K10": obs, "bound": bound,
               "def": "the 10 uncached passes' lower-block gradient contributions summed forward vs reversed (fp32); Theory: accept iff observed <= max(1e-6, 2 x floor)"}
    G["g4_pass"] = obs <= bound; G["g2_pass_by_ruling"] = G["g4_pass"]; G["all_pass"] = bool(G["g1_pass"] and G["g3_pass"] and G["g4_pass"])
    json.dump(G, open(f, "w", newline="\n"), indent=1); print("FLOOR", floor, "observed", obs, "bound", bound, "pass", G["g4_pass"], flush=True)


if __name__ == "__main__":
    main()
