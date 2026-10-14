"""§201 T2 on ProsQA (note 201-6; DESCRIPTIVE): the CLEAN per-pass dump for s191_stepaux_relabel_s1 in the format scratch/s201_t2.py reads, so the
same SHIRYAEV-HEAD primary / descriptive rules run unchanged. fp32 (no autocast), the exact delta loop, passes t = 0..24 (ProsQA's T1 cap).
Items: test = the first 200 p8_na_test items per depth 3..14 (the ProsQA T1 scope); train = 2,000 p8_na_train items (rng 19900 permutation[:2000]).
ProsQA adaptations (the nested dump's keys, same shapes):
  logits      : 2-class [m/2, -m/2], m = ProsMem.margins (log p(correct) - log p(negative)); the target class t = 0, so softmax max-prob / entropy /
                top-2 margin are the 2-way posterior, and correct = (m > 0) - the candidate accuracy;
  n           : the depth (the label 1[t >= depth]);
  aux_bit     : prosqa168s2's AUX-DONE rule - the greedy aux step's last word (24 tokens) is either option; aux_prob = aux_bit (no probability on ProsQA);
  beta / alpha / sla / ret / ri_add / ri_del / dinc: as s201_t1 (ri_* at the decoder's query = the last prefix token, the correct candidate's row).
    python scratch/s201_prosqa_dump.py [run s191_stepaux_relabel_s1] [device cuda:1] [--dump_dir D]   -> <dump>/<run>_{test,train}.npz"""
import json
import os
import sys
import time

ROOT = "D:/Latent"; sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scratch")); os.chdir(ROOT)
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_grad_enabled(False)
import boxes.deltamem as DM  # noqa: E402
import boxes.halt173 as H  # noqa: E402
import boxes.prosqa168s2 as P  # noqa: E402
import s201_t1 as T1  # noqa: E402  (installs the write_seq recorder; retention())
from boxes import s5  # noqa: E402

TMAX = 24; NPER = 200; NTR = 2000; BS = 16


def run_batch(m, ch, dev):
    mem = m.mem; HA, HD = mem.h_add, mem.h_del; B = len(ch); x, msk = m.think_inputs(ch, dev); qpos = torch.tensor([len(it["pre"]) for it in ch], device=dev)
    out = {k: [] for k in ("logits", "aux_bit", "aux_prob", "beta", "alpha", "sla", "ret", "ri_add", "ri_del", "dinc")}; S = mem.zero_state(B, dev); rprev = None
    nanh = torch.full((B, HD), float("nan"), device=dev); nanb = torch.full((B,), float("nan"), device=dev)
    with H.ReadTap() as tap:
        for t in range(TMAX + 1):
            if t:
                T1.REC["on"] = True; Sp = S
                try:
                    S = m.write(S, x, msk)
                finally:
                    T1.REC["on"] = False
                d = T1.REC["d"]; T1.REC["d"] = None; mf = d["mask"].float()[..., None]; L_ = mf.sum(1)
                out["beta"].append((d["be"] * mf).sum(1) / L_); out["alpha"].append((d["al"] * mf).sum(1) / L_); out["sla"].append((torch.log(d["al"]) * mf).sum(1))
                out["ret"].append(T1.retention(Sp[:, HA:HA + HD], d)); out["dinc"].append((S[:, HA:HA + HD] - Sp[:, HA:HA + HD]).float().flatten(1).norm(dim=1))
            else:
                for k_ in ("beta", "alpha", "sla", "ret"):
                    out[k_].append(nanh)
                out["dinc"].append(nanb)
            tap.arm(); mg = torch.tensor(m.margins(S, ch, dev), device=dev, dtype=torch.float32); tap.on = False
            out["logits"].append(torch.stack([mg / 2, -mg / 2], 1))
            xq = tap.x[0::2][torch.arange(B, device=dev), qpos]; r = H.read_vec(mem, xq, S); ra, rd = r[:, :HA * DM.DH], r[:, HA * DM.DH:]
            if rprev is None:
                out["ri_add"].append(nanb); out["ri_del"].append(nanb)
            else:
                out["ri_add"].append((ra - rprev[0]).norm(dim=1) / rprev[0].norm(dim=1).clamp_min(1e-30)); out["ri_del"].append((rd - rprev[1]).norm(dim=1) / rprev[1].norm(dim=1).clamp_min(1e-30))
            rprev = (ra, rd)
            if t:                                                           # prosqa168s2's AUX-DONE: the greedy aux step's object (last word) is an option
                gen = m.aux_greedy(S, t, 24); bit = []
                for r_, it in enumerate(ch):
                    w_ = P.ENC.decode([int(v) for v in gen[r_] if int(v) != P.EOT][:24]).strip().split(".")[0].split(); obj = w_[-1] if w_ else ""
                    bit.append(obj in (it["opt_true"], it["opt_false"]))
                b_ = torch.tensor(bit, device=dev); out["aux_bit"].append(b_); out["aux_prob"].append(b_.float())
            else:
                out["aux_bit"].append(torch.zeros(B, dtype=torch.bool, device=dev)); out["aux_prob"].append(nanb)
    return {k: torch.stack(v, 1).cpu() for k, v in out.items()}


def main():
    run = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "s191_stepaux_relabel_s1"
    dev = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else "cuda:1"
    dump = sys.argv[sys.argv.index("--dump_dir") + 1] if "--dump_dir" in sys.argv else "C:/Users/peter/AppData/Local/Temp/claude/D--/6af9c751-a423-5604-9170-0c177c13d67f/scratchpad/s201_dump"
    os.makedirs(dump, exist_ok=True); rd = os.path.join(ROOT, "boxes", "runs_fsq", run); a = json.load(open(os.path.join(rd, "args.json"))); DM.FAST_ADD = bool(a.get("fast_add")); DM.CHUNK_DELTA = 0
    g = s5.load_gpt2(dev); m = P.ProsMem(g).to(dev); m.load_state_dict(torch.load(os.path.join(rd, "model.pt"), map_location=dev, weights_only=False)); m.eval(); t0 = time.time()
    pr = os.path.join(ROOT, "prosqa", "p8data"); Dte = P.load(os.path.join(pr, "p8_na_test.json")); Dtr = P.load(os.path.join(pr, "p8_na_train.json")); byd = {}
    for it in Dte:
        if len(byd.setdefault(it["depth"], [])) < NPER:
            byd[it["depth"]].append(it)
    sets = {"test": [it for d in sorted(byd) for it in byd[d]], "train": [Dtr[i] for i in np.random.default_rng(19900).permutation(len(Dtr))[:NTR]]}
    for nm, its in sets.items():
        R = [run_batch(m, its[i:i + BS], dev) for i in range(0, len(its), BS)]; R = {k: torch.cat([r[k] for r in R]) for k in R[0]}
        np.savez_compressed(os.path.join(dump, run + "_%s.npz" % nm), n=np.array([it["depth"] for it in its]), t=np.zeros(len(its), dtype=np.int64),
                            **{k: v.numpy().astype(np.float32) if v.dtype.is_floating_point else v.numpy() for k, v in R.items()})
        print(run, nm, "dump", len(its), "%.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
