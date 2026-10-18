"""§218 LANG-VB model gates (LANG_v1 §7 at 83f2dea4) for boxes/langvb.py; each writes scratch/gate_s218/<gate>.json.
  l1a  G-L1(a): all-position slots, K = 1, bank masked: the coda's slot rows == GPT-2's final hidden (ln_f) at every prelude position, fp64 <= 1e-9 rel
       (the decoder rows follow the slots and are causally invisible to them).
  l1b  G-L1(b): LangSLVA with S = all positions on nested inputs, D3-T's weights (s213_d3t_s1/model.pt; LangSLVA's 2-class R_P left out) == D3-T's own
       SLVA rollout (vbank.VBankN.rollout with SLVA's gather): every stored version V_0..V_K and the coda's slot rows after pass K, bitwise (fp32, CPU).
    python scratch/gate_s218_lvb.py l1a --device cpu"""
import argparse
import json
import os
import sys

sys.path.insert(0, "D:/Latent"); os.chdir("D:/Latent")
import torch  # noqa: E402

import boxes.langvb as LV  # noqa: E402
import boxes.nested170 as N  # noqa: E402
import boxes.vbank as VB  # noqa: E402
import boxes.slva as SL  # noqa: E402
from boxes import s5  # noqa: E402

OUT = "scratch/gate_s218"


def all_slots(pad):
    B, T = pad.shape; si = torch.arange(T)[None].expand(B, T).clone(); return si, pad.clone()


def g_l1a(dev):
    g = s5.load_gpt2(dev).double(); m = LV.LangSLVA(g).to(dev).double(); m.c_anneal.fill_(1e4)
    items = [it for it in N.test_set(1000) if it["n"] == 5][:4]; x, pad, lens = VB.batch_x(items, dev); si, sp = all_slots(pad)
    dec = torch.tensor([LV.dec_ids("alpha")] * len(items), device=dev)
    m.bank_off = True; r = m.rollout_ls(x, pad, si, sp, 1, coda_rows={1: torch.arange(len(items))}, dec=dec, lens=lens)
    h = r["coda"][1][1][:, :x.shape[1]]
    ref, _ = g(g.embed(x)); rel = float(((h - ref).abs() * pad[..., None]).max() / ref.abs().max())
    return {"rel_max": rel, "pass": rel <= 1e-9, "items": len(items)}


def g_l1b(dev):
    torch.manual_seed(0); g1 = s5.load_gpt2(dev); g2 = s5.load_gpt2(dev)
    sd = torch.load("boxes/runs_fsq/s213_d3t_s1/model.pt", map_location=dev, weights_only=True)
    ref = SL.SLVA(g1).to(dev); ref.load_state_dict(sd, strict=True); ref.eval()
    m = LV.LangSLVA(g2).to(dev); mk = m.load_state_dict({k: v for k, v in sd.items() if not k.startswith("rec.")}, strict=False); m.eval()
    assert sorted(mk.missing_keys) == ["rec.bias", "rec.weight"] and not mk.unexpected_keys, mk
    for mm in (ref, m):
        mm.c_anneal.fill_(0.0)                                               # the trained regime (c = 0 from step 8000)
    items = [it for it in N.test_set(1000) if it["n"] == 6][:8]; x, pad, lens = VB.batch_x(items, dev); si, sp = all_slots(pad); K = 8
    dec = torch.tensor([LV.dec_ids("alpha")] * len(items), device=dev)
    with torch.no_grad():
        ra = ref.rollout(x, pad, K, coda_rows={K: torch.arange(len(items))})
        rb = m.rollout_ls(x, pad, si, sp, K, coda_rows={K: torch.arange(len(items))}, dec=dec, lens=lens)
    eqV = [bool(torch.equal(a_ * pad[..., None], b_ * pad[..., None])) for a_, b_ in zip(ra["S"], rb["S"])]
    ca = ra["coda"][K][1]; cb = rb["coda"][K][1][:, :x.shape[1]]
    eq_coda = bool(torch.equal(ca * pad[..., None], cb * pad[..., None])); md = float(((ca - cb).abs() * pad[..., None]).max())
    return {"versions_equal": eqV, "coda_slot_rows_equal": eq_coda, "coda_max_abs_diff": md, "K": K, "items": len(items), "pass": all(eqV) and eq_coda}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("gate"); ap.add_argument("--device", default="cpu"); a = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    r = {"l1a": g_l1a, "l1b": g_l1b}[a.gate](a.device); r["device"] = a.device
    json.dump(r, open(os.path.join(OUT, "g_%s.json" % a.gate), "w", newline="\n"), indent=1); print(a.gate, "PASS" if r["pass"] else "FAIL", json.dumps(r)[:600], flush=True)


if __name__ == "__main__":
    main()
