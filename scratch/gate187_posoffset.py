"""note 187-5 gate: DM.POS_OFFSET (deltamem.wpe_slice used by gpt_lower / gpt_forward_mem / gpt_forward_mem_r, deltas5.fwd_read, nested170.dec_reads).
  (1) POS_OFFSET 0 == the pre-change modules bitwise: gpt_forward_mem (FAST_ADD off / on, mask none / some; output + state), gpt_lower, fwd_read,
      nested170 InjectN write / dec_logits / aux_logits / dec_reads on a nested fixture
  (2) POS_OFFSET 3 changes every one of those outputs (the offset reaches each path)
    python scratch/gate187_posoffset.py <pre deltamem.py> <pre deltas5.py> <pre nested170.py>   -> boxes/runs_fsq/s187_gates/gate187_posoffset.json"""
import importlib.util
import json
import os
import sys
from contextlib import nullcontext

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
os.environ.setdefault("BOXES_NB", "5")
import torch  # noqa: E402

torch.set_num_threads(3)
import boxes.deltamem as DM  # noqa: E402
import boxes.deltas5 as D5  # noqa: E402
import boxes.nested170 as N  # noqa: E402
import boxes.halt173 as H  # noqa: E402


def load_pre(name, path, alias):
    spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec); sys.modules[alias] = mod; spec.loader.exec_module(mod); return mod


def outputs(DMm, D5m, Nm, m, items):
    o = {}; x, msk = m.think_inputs(items, "cpu"); S0 = m.mem.zero_state(len(items), "cpu") + 0.01 * torch.randn(len(items), 12, 64, 64, generator=torch.Generator().manual_seed(0))
    for fa in (False, True):
        DMm.FAST_ADD = fa
        for mn, mk in (("mask", msk), ("nomask", None)):
            a_ = DMm.gpt_forward_mem(m.g, m.g.embed(x), m.mem, S0, mask=mk); o["fwd_fast%d_%s_out" % (fa, mn)] = a_[0]; o["fwd_fast%d_%s_state" % (fa, mn)] = a_[1]
    DMm.FAST_ADD = True
    o["lower"] = DMm.gpt_lower(m.g, m.g.embed(x), m.mem.layer)
    o["fwd_read"] = D5m.fwd_read(m.g, m.g.embed(x), m.mem, S0)
    o["dec_reads"] = torch.tensor(Nm.dec_reads(m, items, 2, "cpu", nullcontext, 8))
    return o


def main():
    OUT = {}; DM.FAST_ADD = True
    kind, m, a = H.load_model("boxes/runs_fsq/s184_reweight_s1", "cpu"); TE = N.test_set(1000); items = [it for it in TE if it["n"] == 6][:8]
    with torch.no_grad():
        new0 = outputs(DM, D5, N, m, items)
        DM.POS_OFFSET = 3; new3 = outputs(DM, D5, N, m, items); DM.POS_OFFSET = 0
        pdm = load_pre("pre_dm", sys.argv[1], "boxes.deltamem"); pd5 = load_pre("pre_d5", sys.argv[2], "boxes.deltas5"); pn = load_pre("pre_n", sys.argv[3], "boxes.nested170")
        m.DM = pdm; m.mem.__class__ = pdm.DeltaMem                          # the same weights through the pre-change code
        pre = outputs(pdm, pd5, pn, m, items)
        sys.modules["boxes.deltamem"], sys.modules["boxes.deltas5"], sys.modules["boxes.nested170"] = DM, D5, N
    OUT["g1"] = {k: bool(torch.equal(new0[k], pre[k])) for k in new0}; OUT["g1_pass"] = all(OUT["g1"].values())
    OUT["g2"] = {k: (not torch.equal(new0[k], new3[k])) for k in new0}; OUT["g2_pass"] = all(OUT["g2"].values())
    OUT["all_pass"] = OUT["g1_pass"] and OUT["g2_pass"]
    os.makedirs("boxes/runs_fsq/s187_gates", exist_ok=True); json.dump(OUT, open("boxes/runs_fsq/s187_gates/gate187_posoffset.json", "w", newline="\n"), indent=1); print(json.dumps(OUT))


if __name__ == "__main__":
    main()
