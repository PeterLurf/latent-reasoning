"""--cache_lower gates (Theory / Audit 2026-10-10; boxes/deltamem.gpt_lower + gpt_forward_mem(h_low=) + prosqa168s2 --cache_lower), CPU fp32:
  (1) flags off: gpt_forward_mem with no h_low == the pre-change deltamem copy (output + state, bitwise; mask none / some)
  (2) K = 1 and K = 10 (ckpt off / on): the cached path's main-decoder logits, margins and final state == the uncached path (bitwise), and every
      parameter gradient (trunk + memory) within 1e-6 relative
  (3) the identical-input assert: a pass whose input differs from the cached one is refused
    python scratch/gate_cache_lower.py <pre-change deltamem>   -> boxes/runs_fsq/cache_lower_gates/gate_cache_lower.json"""
import importlib.util
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
os.environ.setdefault("BOXES_NB", "5")
import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402
from torch.utils.checkpoint import checkpoint  # noqa: E402

import boxes.deltamem as DM  # noqa: E402
import boxes.prosqa168s2 as P  # noqa: E402
from boxes import s5  # noqa: E402

GD = os.path.join(ROOT, "boxes", "runs_fsq", "cache_lower_gates"); OUT = {}


def main():
    torch.set_num_threads(3); ok = True
    spec = importlib.util.spec_from_file_location("dm_pre", sys.argv[1]); PRE = importlib.util.module_from_spec(spec); spec.loader.exec_module(PRE)
    g = s5.load_gpt2("cpu"); torch.manual_seed(1); m = P.ProsMem(g)
    with torch.no_grad():
        m.mem.wo.weight.normal_(0, 0.02); m.mem.fade.fill_(1.0); m.mem.gamma.fill_(0.5)
    D = P.load(os.path.join(ROOT, "prosqa", "p8data", "p8_na_train.json"))[:2]; x, msk = m.think_inputs(D, "cpu"); S0 = m.mem.zero_state(2, "cpu") + 0.01 * torch.randn(2, 12, 64, 64)
    with torch.no_grad():
        o1 = DM.gpt_forward_mem(g, g.embed(x), m.mem, S0, mask=msk); o0 = PRE.gpt_forward_mem(g, g.embed(x), m.mem, S0, mask=msk)
        n1 = DM.gpt_forward_mem(g, g.embed(x), m.mem, S0); n0 = PRE.gpt_forward_mem(g, g.embed(x), m.mem, S0)
    OUT["g1"] = {"mask_out_equal": bool(torch.equal(o1[0], o0[0])), "mask_state_equal": bool(torch.equal(o1[1], o0[1])),
                 "nomask_out_equal": bool(torch.equal(n1[0], n0[0])), "nomask_state_equal": bool(torch.equal(n1[1], n0[1]))}
    OUT["g1_pass"] = all(OUT["g1"].values()); ok &= OUT["g1_pass"]
    g2 = {}
    for K in (1, 10):
        for ck in (False, True):
            res = {}
            for cache in (False, True):
                m.zero_grad(set_to_none=True); S = m.mem.zero_state(2, "cpu"); hl = m.lower(x) if cache else None
                for _ in range(K):
                    S = checkpoint(m.write, S, x, msk, hl, use_reentrant=False) if ck else m.write(S, x, msk, hl)
                loss = m.dec_loss(S, D, "cpu"); loss.backward()
                res[cache] = {"S": S.detach().clone(), "loss": loss.detach().clone(), "marg": m.margins(S.detach(), D, "cpu"),
                              "grads": {n_: p_.grad.detach().clone() for n_, p_ in m.named_parameters() if p_.grad is not None}}
            a_, b_ = res[False], res[True]; gk = set(a_["grads"]) | set(b_["grads"])
            rel = max(float((a_["grads"][k] - b_["grads"][k]).abs().max() / (a_["grads"][k].abs().max() + 1e-12)) for k in gk)
            g2["K%d_ckpt%d" % (K, ck)] = {"state_equal": bool(torch.equal(a_["S"], b_["S"])), "loss_equal": bool(torch.equal(a_["loss"], b_["loss"])),
                                         "margins_equal": bool((a_["marg"] == b_["marg"]).all()), "grad_max_rel_diff": rel, "n_grad_params": len(gk),
                                         "same_grad_param_set": set(a_["grads"]) == set(b_["grads"])}
    OUT["g2"] = g2; OUT["g2_pass"] = all(v["state_equal"] and v["loss_equal"] and v["margins_equal"] and v["grad_max_rel_diff"] <= 1e-6 and v["same_grad_param_set"] for v in g2.values())
    ok &= OUT["g2_pass"]
    hl = m.lower(x); x2 = x.clone()
    try:
        m.write(m.mem.zero_state(2, "cpu"), x2, msk, hl); OUT["g3"] = {"refused_other_input": False}
    except AssertionError:
        OUT["g3"] = {"refused_other_input": True}
    OUT["g3_pass"] = OUT["g3"]["refused_other_input"]; ok &= OUT["g3_pass"]
    OUT["all_pass"] = bool(ok); os.makedirs(GD, exist_ok=True); json.dump(OUT, open(os.path.join(GD, "gate_cache_lower.json"), "w", newline="\n"), indent=1)
    print("GATE_CACHE", json.dumps(OUT)[:1500], flush=True); sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
