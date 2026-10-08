"""THEORY §156 SEP-SLOT pre-launch gates (37ac38d1):
  (1) tokenisation: '|' is one token (91); [' =', '|', ' d', '|', ' d', ...] re-encodes to exactly those ids (no ' d|' merge)
  (2) PAUSE length = SEP length (loop148.layout), and the prediction positions are the 8 slots (interleave: the '|' before each digit)
  (3) no loss on separators: the targets are exactly the 8 answer digits per row
  (4) gradflow on the new paths: loop148 plain --answer_sep interleave / pause (CE -> nonzero grad into P, h.0, h.11, wte); multrow --carrier
      deltamem --reader --answer_sep interleave (reader CE + final CE -> nonzero grad into every DeltaMem parameter)
    python scratch/gate156.py [device]   -> boxes/runs_fsq/s156_gates/gate156.json"""
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
os.environ["BOXES_NB"] = "5"
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

import boxes.loop148 as LP  # noqa: E402
import boxes.multrow as MR  # noqa: E402
from boxes import core as bc  # noqa: E402
from boxes import s5  # noqa: E402
from boxes.multcol import DTOK, gen_operands  # noqa: E402


def main():
    dev = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"; out = {"device": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu"}; ok = True
    amp = torch.bfloat16 if dev.startswith("cuda") and torch.cuda.get_device_capability(dev)[0] >= 8 else None; out["amp"] = str(amp)
    sep = bc.tok("|"); smp = []
    for d in range(10):
        smp += [sep[0], DTOK[d]]
    out["g1_sep_ids"] = sep; out["g1_roundtrip"] = bc.ENC.encode(bc.ENC.decode(bc.tok(" =") + smp)) == bc.tok(" =") + smp
    out["g1_pass"] = sep == [91] and out["g1_roundtrip"]; ok &= out["g1_pass"]
    rng = np.random.default_rng(0); A, B = gen_operands(rng, 8, 4), gen_operands(rng, 8, 4); P = LP.prompt_tokens(A, B); Y = np.array(DTOK)[LP.answer_digits(A, B)]
    lens = {}; posd = {}
    for md in ("interleave", "pause"):
        LP.SEP.update({"tok": 91, "mode": md}); x, pos = LP.layout(P, Y); lens[md] = int(x.shape[1]); pl = list(range(x.shape[1]))[pos]; posd[md] = pl
        if md == "interleave":
            out["g2_interleave_slots_are_sep"] = bool((x[:, pl] == 91).all()) and bool((x[:, [p_ + 1 for p_ in pl[:-1]]] == Y[:, :-1]).all())
        else:
            out["g2_pause_last_sep_then_digits"] = bool((x[:, P.shape[1]:P.shape[1] + 8] == 91).all()) and bool((x[:, pl[0] + 1:] == Y[:, :-1]).all())
    out["g2_lengths"] = lens; out["g2_positions"] = posd
    out["g2_pass"] = lens["interleave"] == lens["pause"] and out["g2_interleave_slots_are_sep"] and out["g2_pause_last_sep_then_digits"]; ok &= out["g2_pass"]
    out["g3_targets_per_row"] = int(Y.shape[1]); out["g3_pass"] = Y.shape[1] == 8 and all(len(v) == 8 for v in posd.values()); ok &= out["g3_pass"]
    gr = {}
    for md in ("interleave", "pause"):
        LP.SEP.update({"tok": 91, "mode": md}); m = LP.LoopGPT(s5.load_gpt2(dev)).to(dev); m.train()
        with torch.no_grad():
            m.P.weight.normal_(0, 0.02)
        x, pos = LP.layout(P, Y)
        with LP.amp_ctx(amp):
            o_, _, _ = m.run(torch.tensor(x, device=dev), 1, want={1})
        loss = Fn.cross_entropy(o_[1][:, pos].float().reshape(-1, o_[1].shape[-1]), torch.tensor(Y, device=dev).reshape(-1)); loss.backward()
        gr[md] = {k: float(v.grad.float().norm()) if v.grad is not None else 0.0 for k, v in (("P", m.P.weight), ("h.0", m.g.w("h.0.attn.c_attn.weight")),
                                                                                             ("h.11", m.g.w("h.11.mlp.c_fc.weight")), ("wte", m.g.w("wte.weight")))}
        del m
    LP.SEP.update({"tok": None, "mode": None})
    m = MR.MultRowDelta(s5.load_gpt2(dev), MR.RowFam(), 6).to(dev); m.sep = 91; m.sep_mode = "interleave"; m.train(); MR.set_ops(m, A, B, dev)
    _, med0 = MR.read_ratio_drow(m, A, B, dev, amp, n=8); m.mem.s_add.data.fill_(math.log(med0))
    with torch.no_grad():
        m.mem.wo.weight.normal_(0, 0.02)
    m.mem.fade.fill_(0.5); X = torch.tensor(np.tile(np.arange(4), (8, 1)), device=dev); lab = MR.sj_values(A, B)
    rtok = torch.tensor(np.array(DTOK)[MR.digits8(lab[:, :3])], device=dev); rmask = torch.ones(8, 3, dtype=torch.bool, device=dev)
    for term in ("reader", "final"):
        m.zero_grad(set_to_none=True)
        with MR.amp_ctx(amp):
            fin, rd, _ = m.chain_read(m.start(8, dev), X, lab, rtok, rmask)
        (rd if term == "reader" else fin).backward()
        gr["multrow_" + term] = {n_: float(p_.grad.float().norm()) if p_.grad is not None else 0.0 for n_, p_ in m.mem.named_parameters()}
    ids = m.reader_ids(2, dev, torch.tensor(np.array(DTOK)[MR.digits8(lab[:2, 3])], device=dev)).cpu().numpy()
    out["g4_reader_ids_row0"] = ids[0].tolist(); out["g4_reader_layout_ok"] = bool((ids[:, 2::2] == 91).all()) and ids.shape[1] == 2 + 2 * 8 - 1
    out["g4_grad"] = gr; out["g4_zero"] = {k: [n_ for n_, v in d.items() if v == 0.0] for k, d in gr.items()}
    out["g4_pass"] = all(not z for z in out["g4_zero"].values()) and out["g4_reader_layout_ok"]; ok &= out["g4_pass"]
    out["all_pass"] = bool(ok); d = os.path.join(ROOT, "boxes", "runs_fsq", "s156_gates"); os.makedirs(d, exist_ok=True)
    json.dump(out, open(os.path.join(d, "gate156.json"), "w", newline="\n"), indent=1)
    print("GATE156", json.dumps({k: v for k, v in out.items() if k not in ("g4_grad", "g2_positions")}), flush=True); sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
