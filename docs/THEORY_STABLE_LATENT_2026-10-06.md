# A stable latent method: design synthesis (2026-10-06) — candidate 1 registered as §134; helper's data in

The user's directive (via Audit): "ask literature and helper to work together with theory to invent and synthesize a stable latent
method". Theory leads. Literature maps the prior art (4da2f672 / e0909d0b / ebc2a039); the Audit helper mined our raw runs (report 2026-10-06 15:10, `scratchpad/report_early_separators.md` in its session).
**Goal:** one task-agnostic method that is
(1) **stable to train** (enters without restarts),
(2) **stable to run** (contractive, exact over long chains; C6 / C2),
(3) **causally serial** (C1), and
(4) **expressive** (large or superposed states: the multiplication / ProsQA path),
with the same code and settings across families.

## What we know (one line each; details in THEORY)
* **The narrow FSQ-8 carrier works once entered.** §128 TASK-AGNOSTIC: 5 families; held-out families 12/12 slots with no
  restart; every learner has ρ₁ 0.03–0.27.
* **Entry is the weak point, on S5 only:** 2 restarts in 4 slots, and 1 accepted run escaped without learning.
* **Wide or continuous carriers drift.** RAW-H 768-d: ρ_ff 0.5–0.8, 0/4 exact. §119's published latent methods: 0/9 learned.
* **Entry fixes escape without learning** (§127: STD2 / LR10 / VAR / SKIP s2 all end at 0). §129: a fresh head escapes in 2/4 and learns
  in 0/4. SKIP did rescue one basin seed (s4).
* **Early markers fail** (§130: the code-variance collapse at about 650 is common and transient). The 4k flag catches escaped
  failures 4/4 so far (§131, interim).
* **A weak attractor suffices** (§101: β_min 0.01 passes with a given codebook).
* **Reading a given state is the strongest entry aid we have:**
  - §111 READ (a per-step reader, with labels): 5/5 reliable hand-off.
  - §124 COPY (label-free: decode the given start state back): acquisition at L5 in **4/4** (EXPOSE L5–L8; BASE 2/4); serial in 2/4,
    against 0/4.
* **In-context rules** (§132 pilot): near-exact (0.9865 on held-out tables), serial, with an absolute-state code. The k6 prototype holds
  the op history, not the state, at 2k.
* §133 (running) tests SOFT / NOISE / PIN / FSQ-W error correction on a wide carrier, plus an S5×4 capacity probe.

## What the helper's mining shows (n in brackets; hard-rounded S5 unless stated)
* **Nothing at or before 2k separates learners from escape-without-learning:**
  - the 2k ratio is 0.78 (exact, n 12) against 0.79 (escape-fail, n 18);
  - the 2k probe matches: the state is read (R²_state 0.25–0.47) and the current op is gone, in 5/5 learners and in 6/11 escape-fails;
  - the carried norm matches.
  - Only the **plateau** group shows at 2k: an unread code (abl about 0, R²_state ≤ 0.2) in 10/11, and no or tiny decoder pulls in the
    replays (no pull or dmax < 0.5 → flagged 26/26 on the 4060 and 3070).
* **The earliest separator is the 2.6k → 4k window: loss(4k) / loss(2.6k)** at 0.71 (exact) against 0.997 (escape-fail).
  - At 0.9: 0/12 learners against 14/18 escape-fails flagged (§131's FLAG4).
  - By 4.5k, every exact hard-rounded run is at ≤ 2.49 and every escape-fail at ≥ 3.45.
  - **That window is the soft → hard rounding ramp (p_hard 0 → 1 over 2k–4k) together with L_max 4,** two changes at once.
* **ρ₁ is necessary, not sufficient:** the exact band is 0.12–0.22 on S5, but 5 non-exact runs share it. ρ₁ = 0 marks the
  collapsed-code null (9 runs).
* **COPY (§124) gives the earliest, seed-uniform, sharpest acquisition event** (4/4 at the same round, jumps 0.99, no precursor), against
  BASE (2/4) and EXPOSE (precursors, later).
* **Partial outcomes:** a state-bearing code at 2k is not enough (3070 s17: R²_next 0.86 → the 3.2 basin).
* **End-of-training probes** (§128 S5, kept models):
  - learners hold the state at every length (R² 0.98–1.00);
  - the escape-fail s3 **reverts**: state R² 0.44 at 2k, then about 0 at L 2 / 4 / 8 by 24k, with the current op back at L 8 (0.83).
  - So F2 looks like an unanchored state code decaying (n = 1 probed), not a short-chain op-history shortcut. That fits candidates 1
    and 3.

## Failure taxonomy (what a method must fix)
* **F1, entry basin:** the first decoder pull fails and the code's variance collapses (127-7). Card-dependent, and often transient
  (§130).
* **F2, escape without learning (dominant; it shows only in the 2.6k → 4k window, during the rounding ramp at L 4):** the run passes the 2k flag but never composes. The loss rises at the L_max 2 → 4 → 8
  switches. The code is read by the decoder, but its content is not the state (VAR: R²_state −0.05; LR10 / SKIP s2: 0.25–0.28).
* **F3, drift:** a continuous or wide carrier has no attractor (RAW-H, CONT).
* **F4, capacity:** 8⁸ codes (24 bits) cannot hold large states (S5×4, multiplication, graph sets).

## Candidate 1 (rank 1): COPY-ANCHOR, a label-free state-reading auxiliary from step 0 (targets F1 + F2)
* **Mechanism.** A fraction of each batch (copy rows, e.g. 10 %) is **autoencoding through the carrier:** [EOT, state text] →
  segment → code → answer decoder → the same state text (CE).
  - The text is either the problem's given start state (S5), or **the final answer text of another problem**. That is a
    final-answer-only target, so it is task-agnostic and C4-compatible for every family, including FF / CYC / MOD / RDFA, whose start is
    a code.
  - It gives the decoder a direct gradient to read the state from the code from step 0, with no failed first pull to wait for (F1).
  - It anchors the code's content to **state identity**, not op identity or op history (F2: the code that escaped-but-failed runs
    carry is not the state).
* **Evidence it can work:** §124 COPY (uniform L5 acquisition, 4/4, in self-training); §111 READ (reliable hand-off); 127-7 (failure
  starts with a failed first pull).
* **Falsifiable predictions** (S5, §128 recipe, RTX 3090):
  - (a) The known plain failures become learners: seed 2 (plateau), seed 3 (escaped then failed), seed 4 (plateau). Plain: 0/3.
  - (b) R²_state at 2k is above R²_item in every seed (plain basin seeds: below).
  - (c) No escaped-and-failed run.
  - (d) FAMILY-PASS on all five families is preserved: copy rows do not hurt CYC / MOD / RDFA / FF.
* **Cheapest decisive test** (about $5): a COPY arm on S5 seeds 1–4 (the s2 / s3 / s4 failures plus the s1 learner), with no restarts
  allowed.
  - **PASS = 4/4 exact at 2,048 and SERIAL, with no restart.** The plain recipe is 1/4 without restarts.
  - Then FF / CYC / MOD / RDFA, 1 seed each, as the no-harm guard.
* **Composes with §133:** COPY is carrier-agnostic, so it adds onto whichever carrier §133 shows is stable.

## Candidate 2 (rank 2): ROUND-TRIP ATTRACTOR, a label-free auto-snap toward the model's own decoded state (targets F3 + F4)
* **Mechanism.** At every hand-off, s = (1 − β)·z + β·sg(enc(dec(z))):
  - dec is the answer decoder's greedy text for the current code;
  - enc is Candidate 1's text → code path;
  - the attractor's fixed points are exactly **the codes of decodable states**: the "text manifold", not a grid.
* **What it buys:**
  - It works on a **wide continuous** carrier (F3, an error-correcting pull).
  - Its capacity is the space of decodable texts (F4): multi-token states such as S5×4's four registers, partial products or node sets
    are not bounded by 8⁸.
  - It is the user's "auto snap without hard-coded design": no schedule, no template, only the model's own decoder.
* **Risks:**
  - It locks onto a wrong or constant decode at init, so β ramps from 0 after entry, or is gated by decoder confidence.
  - Cost: one extra decode and encode per hand-off, about 2–3× compute.
* **Falsifiable predictions:**
  - (a) CONT-64 + RT reaches ρ ≤ 0.28 and is exact at 2,048 where CONT-64 drifts (§133's control).
  - (b) On S5×4 it CAP-PASSES where FSQ-8 fails by capacity (§133's reading).
* **Cheapest decisive test:** as a §133-style arm (CONT-64 + RT, β 0.1 from 2k) on the S5 proof seed, then S5×4. Its comparator is §133's
  CONT-64 / SOFT-64.
* **Composes with Candidate 1** (enc is Candidate 1's copy path). Candidate 1 alone fixes entry; Candidates 1 and 2 together are the
  proposed full method: a **COPY-anchored, round-trip-stabilised wide carrier.**

## Candidate 3 (new, from the helper's data): ROUNDING-AWARE ENTRY (targets F2)
* **Hypothesis.** Escape without learning happens when FSQ rounding hardens (2k–4k) before the code has separated the states. The soft code
  that carried state at 2k (R²_state 0.3–0.4) loses it when it is quantised.
  - Rival: the L_max 2 → 4 step (composition demand) is the cause. The two are confounded in §120's schedule.
* **$0 disentangling prototype first** (LOCAL_BACKLOG): the same seeds, plain vs **delayed rounding ramp** (p_hard 0 → 1 over 6k–8k, L_max
  schedule unchanged), short runs to 6k, reading loss(4k) / loss(2.6k) and loss(7k) / loss(5.6k).
  - If the failure window moves with the ramp, rounding is the cause.
* **If so, the fix candidates:**
  - a soft-snap carrier (§133 SOFT, a constant weak pull instead of a hard ramp);
  - or a label-free gated ramp, which hardens only when the code is separable: grid margin, active units (1509.00519), or the decoder's
    agreement between soft and rounded codes.
  - Registered only after the prototype.

## Not proposed
* More entry levers of §127's kind (escape without learning).
* A global contraction penalty (it fights S5's isometric carry; Literature 7dbdb426).
* Learned codebooks (§101's SELF-SNAP premise failed).

## Status
* **Candidate 1 is registered as §134** (26bc216e; scorer `theory/score_s134.py`), with Literature's design checks:
  - constant f;
  - copy decoded from the code alone;
  - a "two dialects" identity probe and a copy-code donor injection (134-1);
  - no copy → one-op rows (they would label transitions);
  - the FF caveat (copy anchors only 2 codes there).
* **Candidate 2** waits for §133's stage 1. Literature's guards apply:
  - dec reads the code only;
  - β ≈ 0.01, with β = 0 until decodes are valid;
  - decode "the answer if queried now" (= the state for S5 / CYC / MOD / RDFA, not FF);
  - arm order COPY → COPY + RT → COPY + RT + SKIP.
* Literature's $0 test (does escape without learning carry the op history?) is queued on the local 3070 (LOCAL_BACKLOG).

## Order
* Register Candidate 1 first (cheap, decisive, targets the dominant failure).
* Register Candidate 2 after §133's stage 1 reads, so its comparators exist.
* Literature's check comes before each registration. Spending is staged with Audit's approval, cloud only.
