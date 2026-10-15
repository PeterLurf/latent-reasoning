"""§207 GSM-SWAP (THEORY.md §207, pre-stated 2026-10-15 09:57 EDT; Audit's lead; descriptive, $0 eval-only on kept final models). No boxes/*.py edit:
the think path is re-implemented here from ProsMem.think / gsm8k_stepaux.think_states (same calls), with per-pass inputs.
  clean    : gsm8k_stepaux's answers() path on all 1,319 GSM8K test items (eval_bs batches in test order, K = think_k): recipients = clean-correct items.
  gates    : G-PLAIN: this script's swap function with x_d = x_r on the clean batches == the clean path (states and answers bitwise).
             G-ID: the same on the recipient batches used by the arms (IDENTITY arm; OWN-KEPT clean on the arm layout).
  donors   : per recipient (test order), rng 20701 draws from the model's clean-correct items until y_d != y_r (norm_number).
  arms     : STATE-SWAP(t) t in {1, 2}: passes 1..t on x_d, t+1..K on x_r, decoder on x_r -> P(y_r) OWN-KEPT, P(y_d)
             ZERO-STATE: K' = 0, decoder on x_r -> P(y_r)
             INPUT-SWAP(k) k in {1, 2}: passes 1..k on x_r, k+1..K on x_d, decoder on x_d -> P(y_d) INPUT-FOLLOW, P(y_r) CARRIED
             INPUT-SWAP-THINK(k): as INPUT-SWAP, decoder keeps x_r -> P(y_r), P(y_d)
  every arm overall and by m <= 2 / m >= 3 with counts.     -> <run>/eval_gswap.json
    python scratch/gsm_swap.py <run dir> [--device cuda:1] [--eval_bs 64] [--fp32]"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("BOXES_NB", "5")
sys.path.insert(0, "D:/Latent")
import numpy as np  # noqa: E402
import torch  # noqa: E402

from boxes import s5  # noqa: E402
import boxes.deltamem as DM  # noqa: E402
import boxes.gsm8k_data as G  # noqa: E402
import boxes.gsm8k_factor as GF  # noqa: E402
from boxes.gsm8k_stepaux import Fwd, ans_ok, greedy_after  # noqa: E402
from boxes.prosqa168s2 import ANS, ENC, EOS, EOT, ProsMem  # noqa: E402
from bench.tasks import norm_number  # noqa: E402


def build(run, dev):
    a_ = json.load(open(os.path.join(run, "args.json"))); g = s5.load_gpt2(dev)
    new = (a_.get("aux_target") or "string") != "string" or bool(a_.get("aux_nopass")) or bool(a_.get("add_norm"))
    m = (GF.FactorMem(g, a_.get("add_norm") or None, bool(a_.get("aux_nopass")), a_.get("aux_target") == "factor", a_.get("seed", 1)) if new else ProsMem(g)).to(dev)
    m.load_state_dict(torch.load(os.path.join(run, "model.pt"), map_location=dev, weights_only=False)); m.eval(); return g, m, a_


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--device", default="cuda:1"); ap.add_argument("--eval_bs", type=int, default=64)
    ap.add_argument("--fp32", action="store_true"); a = ap.parse_args(); dev = a.device; bs = a.eval_bs; T0 = time.time()
    g, m, A = build(a.run, dev); K = int(A["think_k"]); cache = bool(A.get("cache_lower")); DM.FAST_ADD = bool(A.get("fast_add")); DM.CHUNK_DELTA = int(A.get("chunk_delta", 64))
    amp = None if (a.fp32 or not dev.startswith("cuda") or torch.cuda.get_device_capability(dev)[0] < 8) else torch.bfloat16
    ac = (lambda: torch.autocast("cuda", dtype=amp)) if amp is not None else (lambda: torch.autocast("cpu", enabled=False))
    Rte, _ = G.load_aug("test", filter_bad=False); assert len(Rte) == 1319
    D = [{**G.encode(r, ENC, EOS), "_q": r["q"]} for r in Rte]

    @torch.no_grad()
    def think_clean(ch, Kp):                                                # gsm8k_stepaux.think_states (one chunk)
        x, msk = m.think_inputs(ch, dev)
        with ac():
            return m.think(m.mem.zero_state(len(ch), dev), x, msk, Kp, cache=cache)

    @torch.no_grad()
    def think_swap(c1, c2, t1, Kp):
        """passes 1..t1 on chunk c1's inputs, t1+1..Kp on c2's (row-aligned); lower blocks cached per input as ProsMem.think does."""
        S = m.mem.zero_state(len(c1), dev)
        with ac():
            for ch, n_ in ((c1, t1), (c2, Kp - t1)):
                if n_ <= 0:
                    continue
                x, msk = m.think_inputs(ch, dev); hl = m.lower(x) if cache else None
                for _ in range(n_):
                    S = m.write(S, x, msk, hl)
        return S

    @torch.no_grad()
    def decode(S, dec_items):
        return greedy_after(Fwd(g, m.mem, S), [[EOT] + it["pre"] + [ANS] for it in dec_items], dev, ac, bs)

    # clean (the answers() path) + G-PLAIN on the same batches
    P0, gate_s, gate_a = [], True, True
    for i in range(0, len(D), bs):
        ch = D[i:i + bs]; Sc = think_clean(ch, K); pc = decode(Sc, ch); Ss = think_swap(ch, ch, 0, K); ps = decode(Ss, ch)
        gate_s &= bool(torch.equal(Sc, Ss)); gate_a &= pc == ps; P0 += pc
    C0 = np.array([ans_ok(p, it["ans"]) for p, it in zip(P0, D)]); rec = [i for i in range(len(D)) if C0[i]]
    landed = json.load(open(os.path.join(a.run, "eval.json")))
    print("clean acc", C0.mean(), "landed", landed.get("acc"), "G-PLAIN states", gate_s, "answers", gate_a, round(time.time() - T0), flush=True)
    # donors
    rng = np.random.default_rng(20701); yv = {i: norm_number(D[i]["ans"]) for i in rec}; don = {}
    for i in rec:
        while True:
            j = int(rec[rng.integers(0, len(rec))])
            if yv[j] != yv[i]:
                don[i] = j; break
    mt = np.array([D[i]["m"] for i in rec])

    def strata(hit):
        hit = np.asarray(hit, bool); out = {"all": {"p": float(hit.mean()), "n": int(len(hit))}}
        for nm, mk in (("m_le2", mt <= 2), ("m_ge3", mt >= 3)):
            out[nm] = {"p": float(hit[mk].mean()) if mk.any() else None, "n": int(mk.sum())}
        return out

    def run_arm(first, then, t1, Kp, dec):
        P = []
        for s in range(0, len(rec), bs):
            ri = rec[s:s + bs]; R_ = [D[i] for i in ri]; Dn = [D[don[i]] for i in ri]; pick = {"r": R_, "d": Dn}
            S = think_swap(pick[first], pick[then], t1, Kp); P += decode(S, pick[dec])
        return {"own": strata([ans_ok(p, D[i]["ans"]) for p, i in zip(P, rec)]), "donor": strata([ans_ok(p, D[don[i]]["ans"]) for p, i in zip(P, rec)])}
    arms = {"IDENTITY": run_arm("r", "r", 0, K, "r"), "ZERO_STATE": run_arm("r", "r", 0, 0, "r")}
    for t in (1, 2):
        arms["STATE_SWAP_%d" % t] = run_arm("d", "r", t, K, "r")
        arms["INPUT_SWAP_%d" % t] = run_arm("r", "d", t, K, "d")
        arms["INPUT_SWAP_THINK_%d" % t] = run_arm("r", "d", t, K, "r")
        print("t", t, {k: (v["own"]["all"]["p"], v["donor"]["all"]["p"]) for k, v in arms.items()}, round(time.time() - T0), flush=True)
    rd = {"OWN_KEPT_2": arms["STATE_SWAP_2"]["own"]["all"]["p"], "INPUT_FOLLOW_1": arms["INPUT_SWAP_1"]["donor"]["all"]["p"],
          "CARRIED_1": arms["INPUT_SWAP_1"]["own"]["all"]["p"], "readable": len(rec) >= 100}
    rd["reading"] = ("RECOMPUTES-FROM-INPUT" if rd["OWN_KEPT_2"] >= 0.8 and rd["INPUT_FOLLOW_1"] >= 0.8 else
                     "CARRIES-STATE" if rd["OWN_KEPT_2"] <= 0.5 and rd["CARRIED_1"] >= 0.3 else "MIXED") if rd["readable"] else "UNREADABLE (< 100 recipients)"
    res = {"section": "§207 GSM-SWAP", "model": os.path.join(a.run, "model.pt"), "card": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu",
           "dtype": str(amp) if amp else "fp32", "eval_bs": bs, "K": K, "cache_lower": cache, "n_test": len(D), "clean_acc": float(C0.mean()), "landed_acc": landed.get("acc"),
           "clean_agree_landed_items": None, "n_recipients": len(rec), "n_by_m": {"m_le2": int((mt <= 2).sum()), "m_ge3": int((mt >= 3).sum())},
           "gates": {"G_PLAIN_states_bitwise": gate_s, "G_PLAIN_answers_equal": gate_a, "G_ID_own_kept": arms["IDENTITY"]["own"]["all"]["p"]},
           "arms": arms, "readings": rd, "donor_rng": 20701, "sec": round(time.time() - T0),
           "def": "own = P(answer = y_r), donor = P(answer = y_d); STATE_SWAP_t: passes 1..t on x_d then x_r, decoder x_r; INPUT_SWAP_k: passes 1..k on x_r then x_d, "
                  "decoder x_d; INPUT_SWAP_THINK_k: same passes, decoder x_r; ZERO_STATE: K' = 0, decoder x_r; IDENTITY: x_d = x_r on the arm batches (recipients in test order, eval_bs)"}
    try:
        li = json.load(open(os.path.join(a.run, "eval_items.json"))); res["clean_agree_landed_items"] = float(np.mean([bool(x["ok"]) == bool(c) for x, c in zip(li, C0)]))
    except Exception:
        pass
    json.dump(res, open(os.path.join(a.run, "eval_gswap.json"), "w", newline="\n"), indent=1); print("GSWAP", rd, res["gates"], res["sec"], flush=True)


if __name__ == "__main__":
    main()
