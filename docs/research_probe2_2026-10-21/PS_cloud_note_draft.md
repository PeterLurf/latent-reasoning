# Draft Note 222-1 (committed only if Experiments' timing puts 4060-serial CERT past ~24 h; otherwise discarded and the registered 4060 order stands)

**Note 222-1 (<time> EDT; §222 compute placement, before any CERT or A1 §222 probe output; Audit-approved cap; no reading, bar, plant, stream or rule changes).**

- **Why.**
  - The 4060 timing run measured <s/step> for a lockstep group at the registered shapes, so 4060-serial CERT (4 groups) plus A1's §222 probe parts take ≈ <h> h.
  - On one card the order does not change the total (Audit). Parallelism is the only saving.
- **Placement.** Every §222 certification group **and** every A1 §222 probe group runs on **one cloud GPU class, RTX 3090**, fp32, tf32 off, from the same image and code sha. Certification and readings therefore share card class and dtype.
  - **Hosts:** one host per group. That is CERT untrained ×2, CERT reference ×2, and A1's §222 groups (PROBE(t) ×11 + plant_s lin / cexor ×3 t; the grouping is recorded).
  - Lockstep grouping is equivalent by construction (218-10: each group replays the batch stream; per-probe parameters and streams). The 3090's 24 GB allows any group size ≥ the 4060's; the chunk size stays 16, so rollout numerics are unchanged.
  - **A1's model:** if A1's training host is still alive at landing, A1's §222 groups may run there, and its model.pt does not travel. Otherwise A1's model.pt is shipped once, sha-checked. The G-L1c reference model.pt is shipped sha-checked; the untrained model is built on the host from the pinned GPT-2 sha.
- **Card note for PROBE-GAIN-S.** ACC(K′ D+2) comes from A1's §218 eval_lvb (4060, fp32). PROBE(1) comes from a 3090, fp32.
  - Cross-card fp32 differences are expected to be ≪ the 0.35 bar.
  - **Pre-stated check:** A1's acc_d2 at D 3–12 is re-run on one 3090 host (minutes). If max |Δ acc_d2| > 0.01 against the 4060, PROBE-GAIN-S is computed with the 3090 ACC instead, and the flag is reported. This choice is made by the check, not by the outcome.
- **Cost.**
  - Projection: <n> hosts × <h> h × ≈ $0.22 / h ≈ <$>, with per-host admission by G-TIME at the registered group shape.
  - **Family cap $15 under the user's $30 rule (Audit approves ≤ $30).** s222fam GUARD_STOP on realised spend; per-host guards = projection × 1.3. Off-host full-state mirrors every 2k steps for any host > 8 h, plus a tested resume. Drain waiters armed at launch.
  - Vast launches proceed under the user's standing OK for registered work (10-20).
- **Unchanged:** the plants, m (from m_rule.json, computed on the untrained model, card recorded), streams, bars, controls, the choice rule, the readings and the scorer. score_s222 checks `card` for "4060". Under this Note it accepts the class "3090" for every §222 certification and A1 probe file, and requires that one class across them all (a scorer constant changed with a fixture before any output).
