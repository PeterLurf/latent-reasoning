"""THEORY note 202-1 (b0d6e0d9; descriptive, before any §202 output): acc_by_m_kprime for a gsm8k_stepaux STEPAUX model - test accuracy by m (the gold
equation count, as eval.json's acc_by_m / n_by_m) at K' in {0, 1, 10}, plus the pooled acc per K'. Eval-only, a separate file (registered eval.json fields
untouched); the answer path is gsm8k_stepaux's own (think_states: m.think from the zero state under no_grad + autocast, cache = the run's --cache_lower;
greedy_after(Fwd(g, m.mem, S)) on [EOT] + pre + "###"), with the run's DeltaMem flags (fast_add, chunk_delta) from args.json. Written as a standalone
script, not a new --eval_only flag, because §202's deploys ship the working tree (no boxes/ edits while one is in flight); same functions, same order.
GATE (202-1): the pooled K'10 / K'0 / K'1 accuracy must equal the SAME card's own acc_by_kprime at the same --eval_bs (the eval-only record
<run>/eval_local<card>.json when present: §200 A / B have eval_local3070.json at eval_bs 64), and sum(n_by_m) == 1319.
-> <run>/eval_m_kprime.json
    python scratch/s202_mkprime.py <run> [device cuda:1] [--eval_bs 64]"""
import json
import os
import sys

ROOT = "D:/Latent"; sys.path.insert(0, ROOT); os.chdir(ROOT)
os.environ.setdefault("BOXES_NB", "5")
import numpy as np  # noqa: E402
import torch  # noqa: E402

import boxes.deltamem as DM  # noqa: E402
import boxes.gsm8k_data as G  # noqa: E402
import boxes.gsm8k_stepaux as GS  # noqa: E402
from boxes import s5  # noqa: E402
from boxes.prosqa168s2 import ANS, ENC, EOS, EOT, ProsMem  # noqa: E402

KPS = (0, 1, 10)


def main():
    run = sys.argv[1]; dev = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else "cuda:1"
    bs = int(sys.argv[sys.argv.index("--eval_bs") + 1]) if "--eval_bs" in sys.argv else 64; rd = os.path.join(ROOT, "boxes", "runs_fsq", run)
    a = json.load(open(os.path.join(rd, "args.json"))); assert a["arm"] == "stepaux", "202-1: stepaux models only"
    DM.FAST_ADD = bool(a.get("fast_add")); DM.CHUNK_DELTA = int(a.get("chunk_delta") or 0); cache = bool(a.get("cache_lower"))
    amp = None if (not dev.startswith("cuda") or torch.cuda.get_device_capability(dev)[0] < 8) else torch.bfloat16
    ac = (lambda: torch.autocast("cuda", dtype=amp)) if amp is not None else (lambda: torch.autocast("cpu", enabled=False))
    g = s5.load_gpt2(dev); m = ProsMem(g).to(dev); m.load_state_dict(torch.load(os.path.join(rd, "model.pt"), map_location=dev, weights_only=False)); m.eval()
    Rte, _ = G.load_aug("test", filter_bad=False); D = [{**G.encode(r, ENC, EOS), "_q": r["q"]} for r in Rte]; assert len(D) == 1319
    mt = np.array([it["m"] for it in D]); seqs = [[EOT] + it["pre"] + [ANS] for it in D]; out = {}; pooled = {}
    for kp in KPS:
        Ss = []
        for i in range(0, len(D), bs):
            ch = D[i:i + bs]; x, msk = m.think_inputs(ch, dev)
            with torch.no_grad(), ac():
                Ss.append(m.think(m.mem.zero_state(len(ch), dev), x, msk, kp, cache=cache))
        P = GS.greedy_after(GS.Fwd(g, m.mem, torch.cat(Ss)), seqs, dev, ac, bs); C = np.array([GS.ans_ok(p_, it["ans"]) for p_, it in zip(P, D)])
        out[str(kp)] = {str(k_): float(C[mt == k_].mean()) for k_ in sorted(set(mt.tolist()))}; pooled[str(kp)] = float(C.mean())
        print(run, "K'", kp, "pooled", pooled[str(kp)], out[str(kp)], flush=True)
    n_by_m = {str(k_): int((mt == k_).sum()) for k_ in sorted(set(mt.tolist()))}
    card = torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu"; tag = card.split()[-1] if dev.startswith("cuda") else "cpu"
    ref_fp = os.path.join(rd, "eval_local%s.json" % tag); ref = json.load(open(ref_fp)) if os.path.exists(ref_fp) else None
    gate = {"sum_n_by_m": sum(n_by_m.values()), "ref": os.path.basename(ref_fp) if ref else None}
    if ref is not None:
        gate.update({"ref_eval_bs_note": "§200 A / B eval_local3070.json came from --eval_only --eval_bs 64 (01fcde1d); a W reference must use the same bs",
                     "match": {kp: {"this": pooled[kp], "ref": ref["acc_by_kprime"][kp], "equal": pooled[kp] == ref["acc_by_kprime"][kp]} for kp in pooled}})
        gate["pass"] = bool(gate["sum_n_by_m"] == 1319 and all(v["equal"] for v in gate["match"].values()))
    else:
        gate["pass"] = None; gate["note"] = "no same-card eval-only record; the pooled values are this pass's own acc_by_kprime for a later same-card comparison"
    res = {"acc_by_m_kprime": out, "n_by_m": n_by_m, "pooled_by_kprime": pooled, "gate": gate, "card": card, "eval_bs": bs, "eval_dtype": str(amp),
           "dm_flags": {"FAST_ADD": DM.FAST_ADD, "CHUNK_DELTA": DM.CHUNK_DELTA, "cache_lower": cache},
           "def": "note 202-1: test acc by m (gold equation count) at K' in {0,1,10}; gsm8k_stepaux's answer path (think from zero state, greedy after [EOT]+pre+'###')"}
    json.dump(res, open(os.path.join(rd, "eval_m_kprime.json"), "w", newline="\n"), indent=1); print(run, "GATE", gate, flush=True)


if __name__ == "__main__":
    main()
