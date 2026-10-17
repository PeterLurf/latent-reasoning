"""PROOFS_SELFDEPTH check 5 ($0, CPU, 1 thread): the cost / memory projection with the corrections found in the rigour pass:
  * T reference: c = 0.0258 was FITTED to the 11.28 GiB mem_check (PROOFS_D4 l.231), whose padded T is 31 (compose_memT_checks.json), not 33;
  * AdamW state (~8 B / trainable param) is allocated at the first opt.step, AFTER the first backward, so a first-step mem_check peak omits it
    (D4 v1 adds ~1.0 GiB; the SD draft does not);
  * the typical padded T of a training batch (batch max over 64 items) for plain (n <= 8, trainer law) and self (n 9..12 / 9..16) batches,
    simulated, since s/step and typical memory scale with the padded T, while micro is chosen on the forced (longest-item) shape.
Model forms are the draft's [ASSUMED]; only their inputs are corrected here."""
import os, sys, json
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, "D:/Latent")
import numpy as np
import torch
torch.set_num_threads(1)
import boxes.nested170 as N
import boxes.vbank as VB

rng = np.random.default_rng(5)
def padded_T(ns, B=64, reps=300):
    out = []
    for _ in range(reps):
        out.append(max(len(N.render(N.generate(rng, int(rng.choice(ns)))[0])) + 2 for _ in range(B)))
    return float(np.mean(out)), int(np.percentile(out, 95))
plain_w = np.array([1 - 0.8852, 1 - 0.0284, 1 - 0.0002, 1, 1, 1, 1, 1]); plain_w /= plain_w.sum()
def padded_T_plain(B=64, reps=300):
    out = []
    for _ in range(reps):
        out.append(max(len(N.render(N.generate(rng, int(rng.choice(np.arange(1, 9), p=plain_w)))[0])) + 2 for _ in range(B)))
    return float(np.mean(out)), int(np.percentile(out, 95))
T_plain = padded_T_plain(); T_s1 = padded_T([9, 10, 11, 12]); T_s1A = padded_T([9, 10]); T_s2 = padded_T(list(range(9, 17)))
m = VB.make_model.__code__  # noqa: F841  (no model build: parameter count from the known GPT-2 small size + measured nothing)
print("padded T (mean, p95): plain", T_plain, "self R1 stage B", T_s1, "stage A", T_s1A, "self R2", T_s2, flush=True)

TREF = 31; ADAM = 0.93          # GiB: 124.4M GPT-2 small params x 8 B (exp_avg + exp_avg_sq, fp32); the VB/SLVA extras are small [ASSUMED]
def mem(Kb, T, tref=TREF, adam=ADAM):
    return 2 + 0.58 * Kb * T / tref + 0.0258 * (T / tref) ** 2 * sum(t + 1 for t in range(1, Kb + 1)) + adam
def micro_for(g, cap=21.5, fixed=2.0 + ADAM):
    for mic in (1, 2, 4, 8):
        if (g - fixed) / mic + fixed <= cap:
            return mic
rows = {}
for name, Kb, T_forced, T_typ in (("R1 self K16", 16, 47, T_s1[0]), ("R1 self kextra2 K14", 14, 47, T_s1[0]), ("R2 self K20", 20, 61, T_s2[0]), ("R2 self kextra2 K18", 18, 61, T_s2[0])):
    g_draft = 2 + 0.58 * Kb * T_forced / 33 + 0.0258 * (T_forced / 33) ** 2 * sum(t + 1 for t in range(1, Kb + 1))
    g = mem(Kb, T_forced); mic = micro_for(g); mic_d = 1 if g_draft <= 21.5 else 2 if (g_draft - 2) / 2 + 2 <= 21.5 else 4
    # s/step: launch-bound lo = 0.407 * Kb / 11.84 * micro; compute-bound hi = lo * T_typ / T_plain_typ (padded T ratio), and the draft's x T_forced/33
    lo = 0.407 * Kb / 11.84 * mic; hi = lo * T_typ / T_plain[0]; hi_forced = lo * T_forced / TREF
    rows[name] = {"draft_GiB": round(g_draft, 2), "draft_micro": mic_d, "corrected_GiB_forced": round(g, 2), "micro": mic, "s_lo": round(lo, 3),
                  "s_hi_typicalT": round(hi, 3), "s_hi_forcedT": round(hi_forced, 3)}
    print(name, rows[name], flush=True)
def hours(s_self, steps=12000, p=0.5, plain=0.407, over=0.4):
    return (p * s_self + (1 - p) * plain) * steps / 3600 + over
proj = {}
for name, over in (("R1 self K16", 0.4), ("R1 self kextra2 K14", 0.4), ("R2 self K20", 1.0), ("R2 self kextra2 K18", 1.0)):
    r = rows[name]; h_lo = hours(r["s_lo"], over=over); h_hi = hours(r["s_hi_forcedT"], over=over)
    proj[name] = {"hours": (round(h_lo, 2), round(h_hi, 2)), "usd": (round(0.18 * h_lo, 2), round(0.26 * h_hi, 2))}
ctrl = 12000 * 0.407 / 3600 + 0.3; ctrl_g = 12000 * 0.462 / 3600 + 0.3
proj["FT-CTRL"] = {"hours": round(ctrl, 2), "usd": (round(0.18 * ctrl, 2), round(0.26 * ctrl, 2)), "with G-TIME 0.462 s": (round(0.18 * ctrl_g, 2), round(0.26 * ctrl_g, 2))}
proj["pair S1-A"] = (round(proj["R1 self K16"]["usd"][0] + proj["FT-CTRL"]["usd"][0], 2), round(proj["R1 self K16"]["usd"][1] + proj["FT-CTRL"]["usd"][1], 2))
# G-TIME-rate variant: per-pass time at the forced plain shape 0.462 / 12 (T 31) instead of 0.407 / 11.84
pp = 0.462 / 12; lo_g = pp * 16 * rows["R1 self K16"]["micro"]; hi_g = lo_g * 47 / TREF
hg = hours(hi_g, plain=0.462); proj["pair S1-A, G-TIME rates (high end)"] = round(0.26 * hg + 0.26 * ctrl_g, 2)
print(json.dumps(proj, indent=1))
json.dump({"padded_T": {"plain": T_plain, "self_R1B": T_s1, "self_R1A": T_s1A, "self_R2": T_s2}, "rows": rows, "proj": proj, "TREF": TREF, "ADAM_GiB": ADAM},
          open(sys.argv[1], "w", newline="\n"), indent=1)
