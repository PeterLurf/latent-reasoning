"""note 187-5a KEEP-SINK gates (deltamem.POS_KEEP0 + halt173 --posoffset_keep0):
  (1) offset 0 (KEEP0 off and on) == the pre-change modules bitwise on every path (gate187_posoffset.outputs: FAST off / on, mask / none, lower,
      fwd_read, dec_reads)
  (2) offset 3 with KEEP0: the layer-0 input (x + wpe_slice) at position 0 equals offset 0's; positions >= 1 equal x + wpe[p + 3] and differ from offset 0
  (3) inside posoffset_fields(keep0=True) at offset 3, every aux_logits call runs at POS_OFFSET 0 (so the aux logits equal offset 0's for the same S),
      while the think writes and the main decoder run at offset 3 with KEEP0 (calls recorded on a 6-item fixture)
    python scratch/gate187_keep0.py <pre deltamem.py> <pre deltas5.py> <pre nested170.py>   -> boxes/runs_fsq/s187_gates/gate187_keep0.json"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scratch")); os.chdir(ROOT)
os.environ.setdefault("BOXES_NB", "5")
import torch  # noqa: E402

torch.set_num_threads(3)
import gate187_posoffset as GP  # noqa: E402
import boxes.deltamem as DM  # noqa: E402
import boxes.deltas5 as D5  # noqa: E402
import boxes.nested170 as N  # noqa: E402
import boxes.halt173 as H  # noqa: E402


def main():
    OUT = {}; DM.FAST_ADD = True
    kind, m, a = H.load_model("boxes/runs_fsq/s184_reweight_s1", "cpu"); TE = N.test_set(1000); items = [it for it in TE if it["n"] == 6][:8]
    with torch.no_grad():
        DM.POS_KEEP0 = False; o0 = GP.outputs(DM, D5, N, m, items)
        DM.POS_KEEP0 = True; o0k = GP.outputs(DM, D5, N, m, items); DM.POS_KEEP0 = False
        pdm = GP.load_pre("pre_dm", sys.argv[1], "boxes.deltamem"); pd5 = GP.load_pre("pre_d5", sys.argv[2], "boxes.deltas5"); pn = GP.load_pre("pre_n", sys.argv[3], "boxes.nested170")
        dm_cls = m.mem.__class__; m.DM = pdm; m.mem.__class__ = pdm.DeltaMem; pre = GP.outputs(pdm, pd5, pn, m, items)
        m.DM = DM; m.mem.__class__ = dm_cls; sys.modules["boxes.deltamem"], sys.modules["boxes.deltas5"], sys.modules["boxes.nested170"] = DM, D5, N
    OUT["g1"] = {k: bool(torch.equal(o0[k], pre[k]) and torch.equal(o0k[k], pre[k])) for k in o0}; OUT["g1_pass"] = all(OUT["g1"].values())
    w = m.g.w("wpe.weight"); T = 20
    DM.POS_OFFSET = 3; DM.POS_KEEP0 = True; s3 = DM.wpe_slice(m.g, T); DM.POS_OFFSET = 0; DM.POS_KEEP0 = False; s0 = DM.wpe_slice(m.g, T)
    OUT["g2"] = {"pos0_equal": bool(torch.equal(s3[0], s0[0])), "pos_ge1_are_shifted": bool(torch.equal(s3[1:], w[4:T + 3])),
                 "pos_ge1_differ": bool(all(not torch.equal(s3[p], s0[p]) for p in range(1, T)))}
    OUT["g2_pass"] = all(OUT["g2"].values())
    rec = {"aux": [], "write": [], "dec": []}; aw, ww, dw = m.aux_logits, m.write, m.dec_logits
    m.aux_logits = lambda *a_, **k_: (rec["aux"].append((DM.POS_OFFSET, DM.POS_KEEP0)), aw(*a_, **k_))[1]
    m.write = lambda *a_, **k_: (rec["write"].append((DM.POS_OFFSET, DM.POS_KEEP0)), ww(*a_, **k_))[1]
    m.dec_logits = lambda *a_, **k_: (rec["dec"].append((DM.POS_OFFSET, DM.POS_KEEP0)), dw(*a_, **k_))[1]
    N.test_set = lambda per_n=1000: [it for it in TE if it["n"] == 3][:6]
    H.posoffset_fields(m, a, "cpu", None, bs=6, offs=(3,), ns=(3,), keep0=True)
    OUT["g3"] = {"aux_offsets": sorted(set(rec["aux"])), "write_offsets": sorted(set(rec["write"])), "dec_offsets": sorted(set(rec["dec"])),
                 "reset_after": [DM.POS_OFFSET, DM.POS_KEEP0]}
    OUT["g3_pass"] = OUT["g3"]["aux_offsets"] == [(0, True)] and OUT["g3"]["write_offsets"] == [(3, True)] and OUT["g3"]["dec_offsets"] == [(3, True)] and OUT["g3"]["reset_after"] == [0, False]
    OUT["all_pass"] = bool(OUT["g1_pass"] and OUT["g2_pass"] and OUT["g3_pass"])
    json.dump(OUT, open("boxes/runs_fsq/s187_gates/gate187_keep0.json", "w", newline="\n"), indent=1); print(json.dumps(OUT, default=str))


if __name__ == "__main__":
    main()
